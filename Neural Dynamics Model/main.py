import numpy as np
import os
import time
from tqdm import tqdm
from numba import cuda
import config
from src.loader import DataLoader
from src.topography import ReceptorArray
from src.mechanics import StressProcessor
from src.neuron import LIFModel
from src.decoding import PopulationDecoder

try:
    import cupy as cp
except Exception:
    cp = None


def _is_gpu_array(x):
    return cp is not None and isinstance(x, cp.ndarray)


def _to_float(x):
    if _is_gpu_array(x):
        return float(x.item())
    return float(x)


def _to_int(x):
    if _is_gpu_array(x):
        return int(x.item())
    return int(x)


def _median_active_rate_hz(spike_counts, duration_s):
    active_counts = spike_counts[spike_counts > 0]
    if active_counts.size == 0:
        return 0.0
    rates = active_counts / duration_s
    if _is_gpu_array(rates):
        return _to_float(cp.median(rates))
    return float(np.median(rates))


def _combine_channel_rates(counts_x, counts_y, duration_s):
    rate_x = _median_active_rate_hz(counts_x, duration_s)
    rate_y = _median_active_rate_hz(counts_y, duration_s)
    if rate_x <= 0.0:
        return rate_y
    if rate_y <= 0.0:
        return rate_x
    return 0.5 * (rate_x + rate_y)


def _magnitude_stats(x, y):
    xp = cp if _is_gpu_array(x) or _is_gpu_array(y) else np
    mag = xp.sqrt(x * x + y * y)
    return (
        _to_float(xp.min(mag)),
        _to_float(xp.mean(mag)),
        _to_float(xp.max(mag)),
    )


def _complex_abs_max(x):
    if _is_gpu_array(x):
        return _to_float(cp.max(cp.abs(x)))
    return _to_float(np.max(np.abs(x)))


def _rank_methods_desc(results_by_method, metric_key, method_order):
    ranked_values = []
    for method_name in method_order:
        if method_name in results_by_method:
            ranked_values.append((method_name, float(results_by_method[method_name][metric_key])))
    ranked_values.sort(key=lambda item: (-item[1], item[0]))
    ranks = {}
    for idx, (method_name, _) in enumerate(ranked_values, start=1):
        ranks[method_name] = idx
    return ranks


def _print_results_table(title, results_summary, method_order):
    print(f"\n=== {title} ===")
    print(f"{'Method':<10} | {'Intensity':<10} | {'Clarity':<10}")
    print("-" * 40)
    for name in method_order:
        if name in results_summary:
            res = results_summary[name]
            print(f"{name:<10} | {res['Intensity']:<10.3f} | {res['SpatialClarity']:<10.3f}")


def _print_rank_table(title, rank_map, method_order, value_label):
    print(f"\n=== {title} ===")
    print(f"{'Method':<10} | {value_label}")
    print("-" * 32)
    for name in method_order:
        if name in rank_map:
            print(f"{name:<10} | {rank_map[name]:.3f}")


def _run_single_seed(seed, methods_data, input_dt, gpu_ready):
    print(f"\n{'=' * 24} Seed {seed} {'=' * 24}")
    print("Initializing model components...")
    receptor_array = ReceptorArray(
        roi_size_mm=config.ROI_SIZE_MM,
        n_receptors=config.N_RECEPTORS,
        min_dist_mm=config.MIN_DISTANCE_MM,
        seed=seed
    )
    receptor_coords = receptor_array.generate()
    print(f"Generated {len(receptor_coords)} receptors.")

    stress_processor = StressProcessor(
        fs=config.FS_MODEL,
        filter_order=config.FILTER_ORDER,
        f_low=config.F_LOW_HZ,
        f_high=config.F_HIGH_HZ,
        spatial_sigma=config.SPATIAL_SIGMA_MM,
        pca_window_ms=config.PCA_STEADY_STATE_WINDOW_MS,
        use_gpu=gpu_ready
    )

    lif = LIFModel(
        tau_m=config.TAU_M_MS,
        v_rest=config.V_REST,
        v_reset=config.V_RESET,
        v_thresh=config.V_THRESH,
        r_m=config.R_M,
        t_ref=config.T_REF_MS,
        dt=config.DT_MS,
        use_gpu=gpu_ready
    )

    decoder = PopulationDecoder(
        roi_area_mm2=config.ROI_SIZE_MM ** 2,
        density_sigma_mm=config.DENSITY_SIGMA_MM,
        density_grid_mm=config.DENSITY_GRID_MM,
        fidelity_freqs_hz=config.FIDELITY_FREQS_HZ,
        use_gpu=gpu_ready
    )

    print(f"Calibrating Global Gain using method: {config.CALIBRATION_METHOD}...")
    if config.CALIBRATION_METHOD not in methods_data:
        raise ValueError(f"Calibration method {config.CALIBRATION_METHOD} not found in data.")

    calib_data = methods_data[config.CALIBRATION_METHOD]
    calib_processed = stress_processor.process(
        calib_data['stress_xz'],
        calib_data['stress_yz'],
        calib_data['roi_x'],
        calib_data['roi_y'],
        receptor_coords,
        input_dt
    )
    s_calib_x = calib_processed['xz']
    s_calib_y = calib_processed['yz']
    t_calib = calib_processed['t']

    target_rate_per_ms = config.CALIBRATION_TARGET_RATE / config.CALIBRATION_CYCLE_MS
    target_hz = target_rate_per_ms * 1000.0
    gamma = float(config.GLOBAL_GAIN)
    duration_s = len(t_calib) / config.FS_MODEL

    initial_currents_x, initial_currents_y = lif.compute_currents(s_calib_x, s_calib_y, gamma)
    initial_counts_x, initial_counts_y = lif.run_counts(initial_currents_x, initial_currents_y)
    initial_rate = _combine_channel_rates(initial_counts_x, initial_counts_y, duration_s)
    print(f"Initial Gamma={gamma:.4f} -> Median Rate={initial_rate:.2f} Hz")

    if initial_rate < 1.0:
        gamma = 100.0

    for i in range(20):
        currents_x, currents_y = lif.compute_currents(s_calib_x, s_calib_y, gamma)
        spike_counts_x, spike_counts_y = lif.run_counts(currents_x, currents_y)
        median_rate = _combine_channel_rates(spike_counts_x, spike_counts_y, duration_s)

        print(f"Iter {i + 1}: Gamma={gamma:.4f} -> Median Rate={median_rate:.2f} Hz (Target: {target_hz:.2f} Hz)")

        if abs(median_rate - target_hz) < 10.0:
            break

        if median_rate == 0.0:
            gamma *= 2.0
        else:
            gamma *= (target_hz / (median_rate + 1e-5))

    print(f"Calibration Complete. Final Gamma = {gamma:.4f}")

    results_summary = {}

    print("\nRunning Simulation for all methods...")
    for method_name in tqdm(config.STIMULUS_METHODS, desc=f"Methods@seed={seed}"):
        if method_name not in methods_data:
            print(f"Skipping {method_name} (not in data)")
            continue

        t0 = time.time()
        print(f"\n>>> Processing Method: {method_name}")

        m_data = methods_data[method_name]

        processed = stress_processor.process(
            m_data['stress_xz'],
            m_data['stress_yz'],
            m_data['roi_x'],
            m_data['roi_y'],
            receptor_coords,
            input_dt
        )
        s_filt_x = processed['xz']
        s_filt_y = processed['yz']
        drive_x = processed['drive_xz']
        drive_y = processed['drive_yz']
        t_vec = processed['t']

        s_min, s_mean, max_signal = _magnitude_stats(s_filt_x, s_filt_y)
        print(f"\nMethod {method_name}: 2D filtered signal magnitude stats min={s_min:.4f}, mean={s_mean:.4f}, max={max_signal:.4f}")

        currents_x, currents_y = lif.compute_currents(s_filt_x, s_filt_y, gamma)
        c_min, c_mean, c_max = _magnitude_stats(currents_x, currents_y)
        print(f"DEBUG: 2D currents magnitude stats: min={c_min:.4f}, mean={c_mean:.4f}, max={c_max:.4f} (Gamma={gamma:.4f})")

        spikes_x, spikes_y = lif.run(currents_x, currents_y)

        t_end = _to_float(t_vec[-1])
        t_start_win = t_end - (config.DECODING_WINDOW_MS / 1000.0)
        if _is_gpu_array(t_vec):
            win_idx = cp.flatnonzero(t_vec >= t_start_win)
        else:
            win_idx = np.flatnonzero(t_vec >= t_start_win)

        print(f"DEBUG: Window analysis: start={t_start_win:.4f}, end={t_end:.4f}, num_samples={len(win_idx)}")

        spikes_x_win = spikes_x[:, win_idx]
        spikes_y_win = spikes_y[:, win_idx]
        drive_x_win = drive_x[:, win_idx]
        drive_y_win = drive_y[:, win_idx]
        t_win = t_vec[win_idx]

        fidelity_weights = stress_processor.compute_drive_frequency_fidelity(
            drive_x_win,
            drive_y_win,
            config.FIDELITY_FREQS_HZ,
            target_freq_hz=config.PHASE_LOCK_F0_HZ
        )
        qx_complex, qy_complex = decoder.build_vector_phase_field(
            spikes_x_win,
            spikes_y_win,
            t_win,
            fidelity_weights,
            f0=config.PHASE_LOCK_F0_HZ
        )
        metrics = decoder.compute_core_map_metrics(
            qx_complex,
            qy_complex,
            receptor_coords,
            fidelity_weights=fidelity_weights,
        )
        intensity = metrics['intensity']
        clarity = metrics['clarity']

        total_spikes_x = _to_int(cp.sum(spikes_x_win) if _is_gpu_array(spikes_x_win) else np.sum(spikes_x_win))
        total_spikes_y = _to_int(cp.sum(spikes_y_win) if _is_gpu_array(spikes_y_win) else np.sum(spikes_y_win))
        qx_max = _complex_abs_max(qx_complex)
        qy_max = _complex_abs_max(qy_complex)
        print(
            f"DEBUG: Spikes in window total_x={total_spikes_x}, total_y={total_spikes_y}, "
            f"max_|qx|={qx_max:.4f}, max_|qy|={qy_max:.4f}, intensity={intensity:.4f}, clarity={clarity:.4f}"
        )

        results_summary[method_name] = {
            'Intensity': float(intensity),
            'SpatialClarity': float(clarity),
        }

        t1 = time.time()
        print(f"DEBUG: Method {method_name} finished in {t1 - t0:.2f}s")

    _print_results_table(f"Seed {seed} Results", results_summary, config.STIMULUS_METHODS)
    intensity_ranks = _rank_methods_desc(results_summary, 'Intensity', config.STIMULUS_METHODS)
    clarity_ranks = _rank_methods_desc(results_summary, 'SpatialClarity', config.STIMULUS_METHODS)
    _print_rank_table(f"Seed {seed} Intensity Ranks", intensity_ranks, config.STIMULUS_METHODS, 'Rank')
    _print_rank_table(f"Seed {seed} Clarity Ranks", clarity_ranks, config.STIMULUS_METHODS, 'Rank')

    return {
        'seed': seed,
        'gamma': float(gamma),
        'results_summary': results_summary,
        'intensity_ranks': intensity_ranks,
        'clarity_ranks': clarity_ranks,
    }


def main():
    print("=== Computational Neural Dynamics Model ===")
    print(f"DEBUG: Config: FS={config.FS_MODEL}, DT={config.DT_MS}, GPU={config.USE_GPU}")
    use_gpu = bool(getattr(config, "USE_GPU", False))
    gpu_ready = bool(use_gpu and cp is not None and cuda.is_available())
    if gpu_ready:
        cuda.select_device(getattr(config, "GPU_DEVICE_ID", 0))
        print("Backend: GPU (CUDA)")
    else:
        if use_gpu:
            print("Backend: CPU (GPU requested but unavailable)")
        else:
            print("Backend: CPU")

    print(f"Loading data from {config.DATA_FILE}...")
    if not os.path.exists(config.DATA_FILE):
        print("Error: Data file not found. Please run the K-Wave simulation first.")
        return

    loader = DataLoader(config.DATA_FILE)
    data = loader.load()
    methods_data = data['methods']
    input_dt = data.get('dt', 0.1 / 1000)

    seed_run_count = int(getattr(config, 'SEED_RUN_COUNT', 1))
    seed_stride = int(getattr(config, 'SEED_STRIDE', 1))
    if seed_run_count <= 0:
        raise ValueError("SEED_RUN_COUNT must be >= 1")
    if seed_stride <= 0:
        raise ValueError("SEED_STRIDE must be >= 1")

    seeds = [int(config.RANDOM_SEED) + i * seed_stride for i in range(seed_run_count)]
    print(f"Running seed robustness evaluation across {len(seeds)} seeds: {seeds}")

    all_runs = []
    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n##### Seed Run {run_idx}/{len(seeds)} #####")
        run_result = _run_single_seed(seed, methods_data, input_dt, gpu_ready)
        all_runs.append(run_result)

    aggregated_results = {}
    for method_name in config.STIMULUS_METHODS:
        method_runs = [run['results_summary'][method_name] for run in all_runs if method_name in run['results_summary']]
        if not method_runs:
            continue
        aggregated_results[method_name] = {
            'Intensity': float(np.median([item['Intensity'] for item in method_runs])),
            'SpatialClarity': float(np.median([item['SpatialClarity'] for item in method_runs])),
        }

    median_intensity_ranks = {}
    median_clarity_ranks = {}
    for method_name in config.STIMULUS_METHODS:
        intensity_rank_values = [run['intensity_ranks'][method_name] for run in all_runs if method_name in run['intensity_ranks']]
        clarity_rank_values = [run['clarity_ranks'][method_name] for run in all_runs if method_name in run['clarity_ranks']]
        if intensity_rank_values:
            median_intensity_ranks[method_name] = float(np.median(intensity_rank_values))
        if clarity_rank_values:
            median_clarity_ranks[method_name] = float(np.median(clarity_rank_values))

    print("\n=== Seed-by-Seed Summary ===")
    for run in all_runs:
        print(f"Seed {run['seed']}: calibrated gamma={run['gamma']:.4f}")
        for method_name in config.STIMULUS_METHODS:
            if method_name not in run['results_summary']:
                continue
            res = run['results_summary'][method_name]
            intensity_rank = run['intensity_ranks'].get(method_name)
            clarity_rank = run['clarity_ranks'].get(method_name)
            print(
                f"  {method_name:<10} | Intensity={res['Intensity']:.3f} | Clarity={res['SpatialClarity']:.3f} | "
                f"IntensityRank={intensity_rank} | ClarityRank={clarity_rank}"
            )

    _print_results_table("Median Metrics Across Seeds", aggregated_results, config.STIMULUS_METHODS)
    _print_rank_table("Median Intensity Rank Across Seeds", median_intensity_ranks, config.STIMULUS_METHODS, 'MedianRank')
    _print_rank_table("Median Clarity Rank Across Seeds", median_clarity_ranks, config.STIMULUS_METHODS, 'MedianRank')

    print("\nDone.")


if __name__ == "__main__":
    main()
