import numpy as np
import os
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


def main():
    print("=== Computational Neural Dynamics Model ===")
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

    print("Initializing model components...")
    receptor_array = ReceptorArray(
        roi_size_mm=config.ROI_SIZE_MM,
        n_receptors=config.N_RECEPTORS,
        min_dist_mm=config.MIN_DISTANCE_MM,
        seed=config.RANDOM_SEED
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
        use_gpu=gpu_ready
    )

    print(f"Calibrating Global Gain using method: {config.CALIBRATION_METHOD}...")
    if config.CALIBRATION_METHOD not in methods_data:
        print(f"Error: Calibration method {config.CALIBRATION_METHOD} not found in data.")
        return

    calib_data = methods_data[config.CALIBRATION_METHOD]
    s_calib, t_calib, _ = stress_processor.process(
        calib_data['stress_xz'],
        calib_data['stress_yz'],
        calib_data['roi_x'],
        calib_data['roi_y'],
        receptor_coords,
        input_dt
    )

    target_rate_per_ms = config.CALIBRATION_TARGET_RATE / config.CALIBRATION_CYCLE_MS
    target_hz = target_rate_per_ms * 1000.0
    gamma = float(config.GLOBAL_GAIN)
    duration_s = len(t_calib) / config.FS_MODEL

    initial_currents = lif.compute_current(s_calib, gamma)
    initial_counts = lif.run_counts(initial_currents)
    initial_rate = _median_active_rate_hz(initial_counts, duration_s)
    print(f"Initial Gamma={gamma:.4f} -> Median Rate={initial_rate:.2f} Hz")

    if initial_rate < 1.0:
        gamma = 100.0

    for i in range(20):
        currents = lif.compute_current(s_calib, gamma)
        spike_counts = lif.run_counts(currents)
        median_rate = _median_active_rate_hz(spike_counts, duration_s)

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
    for method_name in tqdm(config.STIMULUS_METHODS, desc="Methods"):
        if method_name not in methods_data:
            print(f"Skipping {method_name} (not in data)")
            continue

        m_data = methods_data[method_name]

        s_filt, t_vec, s_raw = stress_processor.process(
            m_data['stress_xz'],
            m_data['stress_yz'],
            m_data['roi_x'],
            m_data['roi_y'],
            receptor_coords,
            input_dt
        )

        max_signal = _to_float(cp.max(s_filt) if _is_gpu_array(s_filt) else np.max(s_filt))
        s_min = _to_float(cp.min(s_filt) if _is_gpu_array(s_filt) else np.min(s_filt))
        s_mean = _to_float(cp.mean(s_filt) if _is_gpu_array(s_filt) else np.mean(s_filt))
        print(f"\nMethod {method_name}: Signed filtered signal stats min={s_min:.4f}, mean={s_mean:.4f}, max={max_signal:.4f}")

        currents = lif.compute_current(s_filt, gamma)
        c_max = _to_float(cp.max(currents) if _is_gpu_array(currents) else np.max(currents))
        c_mean = _to_float(cp.mean(currents) if _is_gpu_array(currents) else np.mean(currents))
        print(f"DEBUG: Currents stats: mean={c_mean:.4f}, max={c_max:.4f} (Gamma={gamma:.4f})")

        spikes = lif.run(currents)

        t_end = _to_float(t_vec[-1])
        t_start_win = t_end - (config.DECODING_WINDOW_MS / 1000.0)
        if _is_gpu_array(t_vec):
            win_idx = cp.flatnonzero(t_vec >= t_start_win)
        else:
            win_idx = np.flatnonzero(t_vec >= t_start_win)

        print(f"DEBUG: Window analysis: start={t_start_win:.4f}, end={t_end:.4f}, num_samples={len(win_idx)}")

        spikes_win = spikes[:, win_idx]
        t_win = t_vec[win_idx]
        phase_weights = decoder.compute_phase_locked_weight(
            spikes_win,
            t_win,
            f0=config.PHASE_LOCK_F0_HZ
        )
        intensity, clarity, density_map = decoder.compute_core_map_metrics(phase_weights, receptor_coords)

        total_spikes = _to_int(cp.sum(spikes_win) if _is_gpu_array(spikes_win) else np.sum(spikes_win))
        q_max = _to_float(cp.max(phase_weights) if _is_gpu_array(phase_weights) else np.max(phase_weights))
        rho_max = _to_float(cp.max(density_map) if _is_gpu_array(density_map) else np.max(density_map))
        print(f"DEBUG: Spikes in window total={total_spikes}, max_phase_weight={q_max:.4f}, rho_max={rho_max:.4f}")

        results_summary[method_name] = {
            'Intensity': intensity,
            'SpatialClarity': clarity,
        }

    print("\n=== Final Results ===")
    print(f"{'Method':<10} | {'Intensity':<10} | {'Clarity':<10}")
    print("-" * 50)
    for name in config.STIMULUS_METHODS:
        if name in results_summary:
            res = results_summary[name]
            print(f"{name:<10} | {res['Intensity']:<10.3f} | {res['SpatialClarity']:<10.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
