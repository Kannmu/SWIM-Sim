import itertools
import os
import time

import numpy as np
from numba import cuda

import config
from src.decoding import CoherentFieldDecoder
from src.loader import DataLoader
from src.mechanics import StressProcessor
from src.topography import ReceptorArray

try:
    import cupy as cp
except Exception:
    cp = None


receptor_coords = None


def _is_gpu_array(x):
    return cp is not None and isinstance(x, cp.ndarray)


def _to_numpy(x):
    if _is_gpu_array(x):
        return cp.asnumpy(x)
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy(v) for v in x)
    return np.asarray(x)


def _build_pair_list(methods):
    return list(itertools.combinations(methods, 2))


def _sigmoid(x):
    x = float(x)
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def _precompute_processed_cache(processed):
    use_gpu = _is_gpu_array(processed["xz"]) or _is_gpu_array(processed["yz"])
    xp = cp if use_gpu else np

    filtered_x = xp.asarray(processed["xz"], dtype=xp.float64)
    filtered_y = xp.asarray(processed["yz"], dtype=xp.float64)
    drive_x = xp.asarray(processed["drive_xz"], dtype=xp.float64)
    drive_y = xp.asarray(processed["drive_yz"], dtype=xp.float64)
    t_vec = xp.asarray(processed["t"], dtype=xp.float64)

    t_end = float(_to_numpy(t_vec[-1]))
    t_start = t_end - (config.STEADY_STATE_WINDOW_MS / 1000.0)
    win_idx = xp.flatnonzero(t_vec >= t_start)
    if int(win_idx.size) == 0:
        raise ValueError("Steady-state window is empty.")

    return {
        "filtered_x_win": xp.ascontiguousarray(filtered_x[:, win_idx]),
        "filtered_y_win": xp.ascontiguousarray(filtered_y[:, win_idx]),
        "drive_x_win": xp.ascontiguousarray(drive_x[:, win_idx]),
        "drive_y_win": xp.ascontiguousarray(drive_y[:, win_idx]),
        "t_win": xp.ascontiguousarray(t_vec[win_idx]),
        "n_receptors": int(filtered_x.shape[0]),
    }


def _estimate_drive_scale(center_caches):
    medians = []
    for cache in center_caches.values():
        fx = _to_numpy(cache["filtered_x_win"]).astype(np.float64, copy=False)
        fy = _to_numpy(cache["filtered_y_win"]).astype(np.float64, copy=False)
        rms = np.sqrt(np.mean(fx * fx + fy * fy, axis=1))
        medians.append(float(np.median(rms)))
    scale = float(np.median(medians))
    return max(scale, 1e-12)


def _normalize_processed_cache(cache, scale):
    use_gpu = _is_gpu_array(cache["filtered_x_win"]) or _is_gpu_array(cache["filtered_y_win"])
    xp = cp if use_gpu else np
    s = float(scale)
    return {
        "filtered_x_win": xp.asarray(cache["filtered_x_win"], dtype=xp.float64) / s,
        "filtered_y_win": xp.asarray(cache["filtered_y_win"], dtype=xp.float64) / s,
        "drive_x_win": xp.asarray(cache["drive_x_win"], dtype=xp.float64) / s,
        "drive_y_win": xp.asarray(cache["drive_y_win"], dtype=xp.float64) / s,
        "t_win": cache["t_win"],
        "n_receptors": int(cache["n_receptors"]),
    }


def _compute_condition_bridge(name, processed_cache, decoder):
    metrics = decoder.compute_bridge_metrics(
        filtered_x=processed_cache["filtered_x_win"],
        filtered_y=processed_cache["filtered_y_win"],
        raw_drive_x=processed_cache["drive_x_win"],
        raw_drive_y=processed_cache["drive_y_win"],
        t_vec=processed_cache["t_win"],
        receptor_coords=receptor_coords,
        target_freq_hz=config.BRIDGE_TARGET_FREQ_HZ,
    )
    print(
        f"[{name}] Intensity={metrics['intensity_score']:.6e}, "
        f"Clarity={metrics['clarity_score']:.6e}, "
        f"Area={metrics['effective_area_mm2']:.6f} mm^2, "
        f"FFI_raw={metrics['mean_ffi']:.6f}, "
        f"NDWCI={metrics['ndwci']:.6f}"
    )
    return metrics


def _standardize_log_scores(score_dict, method_order, eps):
    raw = np.asarray(
        [np.log(float(score_dict[m]) + float(eps)) for m in method_order],
        dtype=np.float64,
    )
    mu = float(np.mean(raw))
    sigma = float(np.std(raw))
    if sigma <= float(eps):
        return {m: 0.0 for m in method_order}
    z = (raw - mu) / (sigma + float(eps))
    return {m: float(v) for m, v in zip(method_order, z)}


def _compute_pairwise_from_scalar_scores(score_dict, method_order, eps):
    zscore_dict = _standardize_log_scores(score_dict, method_order, eps)
    pairwise_results = []
    for a, b in _build_pair_list(method_order):
        p = _sigmoid(zscore_dict[a] - zscore_dict[b])
        pairwise_results.append({"A": a, "B": b, "probability": float(p)})
    return pairwise_results, zscore_dict


def _print_method_summary(title, method_results, method_order):
    print(f"\n=== {title} ===")
    print(
        f"{'Method':<10} | {'Intensity':<13} | {'Clarity':<13} | {'EffArea(mm2)':<13} | {'FFI_raw':<8} | {'NDWCI':<8}"
    )
    print("-" * 86)

    def fmt(v, w):
        v = float(v)
        if abs(v) > 10000 or (abs(v) > 0 and abs(v) < 1e-3):
            return f"{v:<{w}.3e}"
        return f"{v:<{w}.6f}"

    for method in method_order:
        if method not in method_results:
            continue
        result = method_results[method]
        print(
            f"{method:<10} | {fmt(result['intensity_score'], 13)} | {fmt(result['clarity_score'], 13)} | "
            f"{fmt(result['effective_area_mm2'], 13)} | {fmt(result['mean_ffi'], 8)} | {fmt(result['ndwci'], 8)}"
        )


def _print_pairwise_table(title, pairwise_results):
    print(f"\n=== {title} ===")
    print(f"{'Pair':<24} | {'P(A>B)':<10}")
    print("-" * 40)
    for item in pairwise_results:
        pair_name = f"{item['A']} > {item['B']}"
        print(f"{pair_name:<24} | {item['probability']:<10.4f}")


def _aggregate_runs(all_runs):
    aggregated_methods = {}
    for method in config.STIMULUS_METHODS:
        method_runs = [
            run["method_results"][method]
            for run in all_runs
            if method in run["method_results"]
        ]
        if not method_runs:
            continue
        aggregated_methods[method] = {
            "intensity_score": float(
                np.median([item["intensity_score"] for item in method_runs])
            ),
            "clarity_score": float(
                np.median([item["clarity_score"] for item in method_runs])
            ),
            "effective_area_mm2": float(
                np.median([item["effective_area_mm2"] for item in method_runs])
            ),
            "mean_ffi": float(np.median([item["mean_ffi"] for item in method_runs])),
            "ndwci": float(np.median([item["ndwci"] for item in method_runs])),
        }

    def aggregate_pairwise(key):
        first = all_runs[0][key]
        out = []
        for idx, item in enumerate(first):
            vals = [run[key][idx]["probability"] for run in all_runs]
            out.append(
                {"A": item["A"], "B": item["B"], "probability": float(np.median(vals))}
            )
        return out

    return (
        aggregated_methods,
        aggregate_pairwise("intensity_pairwise"),
        aggregate_pairwise("clarity_pairwise"),
    )


def _run_single_seed(seed, methods_data, input_dt_value, gpu_ready):
    global receptor_coords

    print(f"\n{'=' * 24} Seed {seed} {'=' * 24}")
    receptor_array = ReceptorArray(
        roi_size_mm=config.ROI_SIZE_MM,
        n_receptors=config.N_RECEPTORS,
        min_dist_mm=config.MIN_DISTANCE_MM,
        seed=seed,
    )
    receptor_coords = receptor_array.generate()
    print(f"Generated {len(receptor_coords)} receptors.")

    stress_processor = StressProcessor(
        fs=config.FS_MODEL,
        filter_order=config.FILTER_ORDER,
        f_low=config.F_LOW_HZ,
        f_high=config.F_HIGH_HZ,
        spatial_sigma=config.SPATIAL_SIGMA_MM,
        use_gpu=gpu_ready,
        enable_temporal_tuning=config.ENABLE_TEMPORAL_TUNING,
        temporal_tuning_center_hz=config.TEMPORAL_TUNING_CENTER_HZ,
        temporal_tuning_sigma_oct=config.TEMPORAL_TUNING_SIGMA_OCT,
    )
    decoder = CoherentFieldDecoder(
        roi_area_mm2=config.ROI_SIZE_MM**2,
        density_sigma_mm=config.DENSITY_SIGMA_MM,
        density_grid_mm=config.DENSITY_GRID_MM,
        fidelity_freqs_hz=config.FIDELITY_FREQS_HZ,
        use_gpu=gpu_ready,
    )

    if config.ENABLE_TEMPORAL_TUNING:
        w200 = stress_processor.temporal_tuning_weight(200.0)
        w400 = stress_processor.temporal_tuning_weight(400.0)
        print(
            f"Temporal tuning: center={config.TEMPORAL_TUNING_CENTER_HZ:.1f} Hz, "
            f"sigma_oct={config.TEMPORAL_TUNING_SIGMA_OCT:.2f}, "
            f"w200={w200:.3f}, w400={w400:.3f}, "
            f"w400/w200={w400 / (w200 + 1e-12):.3f}"
        )

    raw_center_caches = {}
    for method in config.STIMULUS_METHODS:
        if method not in methods_data:
            continue
        method_data = methods_data[method]
        processed = stress_processor.process(
            method_data["stress_xz"],
            method_data["stress_yz"],
            method_data["roi_x"],
            method_data["roi_y"],
            receptor_coords,
            input_dt_value,
        )
        raw_center_caches[method] = _precompute_processed_cache(processed)

    drive_scale = _estimate_drive_scale(raw_center_caches)
    print(f"Drive normalization scale: {drive_scale:.6e}")

    center_caches = {
        method: _normalize_processed_cache(cache, drive_scale)
        for method, cache in raw_center_caches.items()
    }

    method_results = {}
    for method in config.STIMULUS_METHODS:
        if method not in center_caches:
            continue
        method_results[method] = _compute_condition_bridge(
            method,
            center_caches[method],
            decoder,
        )

    intensity_scores = {
        method: method_results[method]["intensity_score"]
        for method in config.PAIRWISE_METHODS
        if method in method_results
    }
    clarity_scores = {
        method: method_results[method]["clarity_score"]
        for method in config.PAIRWISE_METHODS
        if method in method_results
    }
    method_order = [m for m in config.PAIRWISE_METHODS if m in method_results]

    intensity_pairwise, intensity_z = _compute_pairwise_from_scalar_scores(
        intensity_scores,
        method_order,
        config.BRIDGE_SCORE_EPS,
    )
    clarity_pairwise, clarity_z = _compute_pairwise_from_scalar_scores(
        clarity_scores,
        method_order,
        config.BRIDGE_SCORE_EPS,
    )

    _print_method_summary(f"Seed {seed} method summary", method_results, config.STIMULUS_METHODS)
    _print_pairwise_table(f"Seed {seed} intensity 2-AFC", intensity_pairwise)
    _print_pairwise_table(f"Seed {seed} clarity 2-AFC", clarity_pairwise)

    return {
        "seed": seed,
        "method_results": _to_numpy(method_results),
        "intensity_pairwise": _to_numpy(intensity_pairwise),
        "clarity_pairwise": _to_numpy(clarity_pairwise),
        "intensity_z": _to_numpy(intensity_z),
        "clarity_z": _to_numpy(clarity_z),
    }


def main():
    print("=== Computational Neural Dynamics Model ===")
    print(f"Config: FS={config.FS_MODEL}, DT={config.DT_MS}, GPU={config.USE_GPU}")
    use_gpu = bool(getattr(config, "USE_GPU", False))
    gpu_ready = bool(use_gpu and cp is not None and cuda.is_available())
    if gpu_ready:
        cuda.select_device(getattr(config, "GPU_DEVICE_ID", 0))
        print("Backend: GPU preprocessing available")
    else:
        print("Backend: CPU")

    print(f"Loading data from {config.DATA_FILE}...")
    if not os.path.exists(config.DATA_FILE):
        print("Error: Data file not found. Please run the K-Wave simulation first.")
        return

    loader = DataLoader(config.DATA_FILE)
    data = loader.load()
    methods_data = data["methods"]
    input_dt_value = data.get("dt", 0.1 / 1000)

    seed_run_count = int(getattr(config, "SEED_RUN_COUNT", 1))
    seed_stride = int(getattr(config, "SEED_STRIDE", 1))
    if seed_run_count <= 0:
        raise ValueError("SEED_RUN_COUNT must be >= 1")
    if seed_stride <= 0:
        raise ValueError("SEED_STRIDE must be >= 1")

    seeds = [int(config.RANDOM_SEED) + i * seed_stride for i in range(seed_run_count)]
    print(f"Running seed robustness evaluation across {len(seeds)} seeds: {seeds}")

    all_runs = []
    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n##### Seed Run {run_idx}/{len(seeds)} #####")
        t0 = time.time()
        run_result = _run_single_seed(seed, methods_data, input_dt_value, gpu_ready)
        all_runs.append(run_result)
        print(f"Seed {seed} finished in {time.time() - t0:.2f}s")

    aggregated_methods, aggregated_intensity, aggregated_clarity = _aggregate_runs(all_runs)
    _print_method_summary(
        "Median metrics across seeds",
        aggregated_methods,
        config.STIMULUS_METHODS,
    )
    _print_pairwise_table("Median intensity 2-AFC across seeds", aggregated_intensity)
    _print_pairwise_table("Median clarity 2-AFC across seeds", aggregated_clarity)
    print("\nDone.")


if __name__ == "__main__":
    main()
