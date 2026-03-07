import itertools
import os
import time

import numpy as np
from numba import cuda
from tqdm import tqdm

import config
from src.decoding import PopulationDecoder
from src.loader import DataLoader
from src.mechanics import StressProcessor
from src.neuron import LIFModel
from src.topography import ReceptorArray

try:
    import cupy as cp
except Exception:
    cp = None


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


def _ensure_2d_cov(cov):
    if _is_gpu_array(cov):
        cov = cp.asarray(cov, dtype=cp.float64)
        if cov.ndim == 0:
            cov = cp.asarray([[float(cov.item())]], dtype=cp.float64)
        return cov
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    return cov


def _bin_sample_count():
    samples = int(round(config.TRIAL_BIN_MS / config.DT_MS))
    if samples <= 0:
        raise ValueError("TRIAL_BIN_MS / DT_MS must be >= 1 sample")
    return samples


def _mean_choice_probability(values_a, values_b):
    xp = cp if _is_gpu_array(values_a) or _is_gpu_array(values_b) else np
    values_a = xp.asarray(values_a, dtype=xp.float64)
    values_b = xp.asarray(values_b, dtype=xp.float64)
    if values_a.shape != values_b.shape:
        raise ValueError("Trial metric arrays must have the same shape")
    result = xp.mean(values_a > values_b) + 0.5 * xp.mean(values_a == values_b)
    return float(result.item() if xp is cp else result)


def _build_pair_list(methods):
    return list(itertools.combinations(methods, 2))


def _print_method_summary(title, method_results, method_order):
    print(f"\n=== {title} ===")
    print(f"{'Method':<10} | {'Detectability':<14} | {'Clarity':<14} | {'FFI':<8} | {'NDWCI':<8}")
    print("-" * 70)
    for method in method_order:
        if method not in method_results:
            continue
        result = method_results[method]
        mech = result['mechanistic']
        print(
            f"{method:<10} | {result['intensity_mean']:<14.4f} | {result['clarity_mean']:<14.4f} | "
            f"{mech['mean_ffi']:<8.4f} | {mech['ndwci']:<8.4f}"
        )


def _print_pairwise_table(title, pairwise_results):
    print(f"\n=== {title} ===")
    print(f"{'Pair':<24} | {'P(A>B)':<10}")
    print("-" * 40)
    for item in pairwise_results:
        pair_name = f"{item['A']} > {item['B']}"
        print(f"{pair_name:<24} | {item['probability']:<10.4f}")


def _generate_calibration_scales():
    scales = []
    for freq in config.CALIBRATION_AM_FREQUENCIES_HZ:
        for amp in config.CALIBRATION_AMPLITUDE_LEVELS:
            scales.append({
                'label': f'calib_{int(freq)}Hz_amp{amp:.2f}',
                'freq_hz': float(freq),
                'amp_scale': float(amp),
            })
    return scales


def _calibration_target_probability(freq_hz, amp_scale):
    amp_center = 0.55 if int(round(freq_hz)) == 200 else 0.75
    return float(1.0 / (1.0 + np.exp(-config.CALIBRATION_SIGMOID_SLOPE * (amp_scale - amp_center))))


def _pair_mahalanobis_scores(projected, mu0, sigma0_inv):
    xp = cp if _is_gpu_array(projected) or _is_gpu_array(mu0) or _is_gpu_array(sigma0_inv) else np
    delta = xp.asarray(projected) - xp.asarray(mu0)
    sigma0_inv_arr = xp.asarray(sigma0_inv)
    return xp.einsum('ij,jk,ik->i', delta, sigma0_inv_arr, delta, optimize=True)


def _make_response_matrix(response_vectors):
    if not response_vectors:
        raise ValueError("At least one response vector is required.")
    use_gpu = any(_is_gpu_array(item) for item in response_vectors)
    xp = cp if use_gpu else np
    return xp.asarray(response_vectors, dtype=xp.float32)


def _precompute_processed_cache(processed, samples_per_bin):
    use_gpu = _is_gpu_array(processed['xz']) or _is_gpu_array(processed['yz'])
    xp = cp if use_gpu else np
    filtered_x = xp.asarray(processed['xz'], dtype=xp.float64)
    filtered_y = xp.asarray(processed['yz'], dtype=xp.float64)
    drive_x = xp.asarray(processed['drive_xz'], dtype=xp.float64)
    drive_y = xp.asarray(processed['drive_yz'], dtype=xp.float64)
    t_vec = xp.asarray(processed['t'], dtype=xp.float64)
    t_end = float(_to_numpy(t_vec[-1]))
    t_start = t_end - (config.STEADY_STATE_WINDOW_MS / 1000.0)
    win_idx = xp.flatnonzero(t_vec >= t_start)
    win_idx_size = int(win_idx.size)
    if win_idx_size == 0:
        raise ValueError("Steady-state window is empty.")
    window_start_idx = int(_to_numpy(win_idx[0]))
    t_win = t_vec[win_idx]
    drive_x_win = drive_x[:, win_idx]
    drive_y_win = drive_y[:, win_idx]
    n_bins = win_idx_size // samples_per_bin
    if n_bins <= 0:
        raise ValueError("Steady-state window is shorter than one bin.")
    trimmed = n_bins * samples_per_bin
    return {
        'filtered_x': filtered_x,
        'filtered_y': filtered_y,
        'drive_x_win': xp.ascontiguousarray(drive_x_win[:, :trimmed]),
        'drive_y_win': xp.ascontiguousarray(drive_y_win[:, :trimmed]),
        't_win': xp.ascontiguousarray(t_win[:trimmed]),
        'window_start_idx': window_start_idx,
        'samples_per_bin': samples_per_bin,
        'n_bins': n_bins,
        'n_receptors': filtered_x.shape[0],
    }


def _sample_thresholds(rng, n_receptors, lif_model):
    if not config.ENABLE_THRESHOLD_NOISE:
        base = np.full(n_receptors, lif_model.v_thresh, dtype=np.float64)
        return base, base.copy()
    sigma = config.THRESHOLD_NOISE_FRACTION * lif_model.v_thresh
    tx = rng.normal(lif_model.v_thresh, sigma, size=n_receptors)
    ty = rng.normal(lif_model.v_thresh, sigma, size=n_receptors)
    floor = 1e-6
    return np.clip(tx, floor, None), np.clip(ty, floor, None)


def _sample_input_noise(rng, shape, sigma_n):
    if not config.ENABLE_INPUT_NOISE or sigma_n <= 0.0:
        return np.zeros(shape, dtype=np.float64)
    return rng.normal(0.0, sigma_n, size=shape)


def _build_trial_noise_bank(n_trials, processed_cache, lif_model, rng):
    n_receptors = processed_cache['n_receptors']
    shape = processed_cache['filtered_x'].shape
    sigma_thresh = config.THRESHOLD_NOISE_FRACTION * lif_model.v_thresh
    bank = []
    for _ in range(n_trials):
        if config.ENABLE_INPUT_NOISE:
            z_x = rng.normal(0.0, 1.0, size=shape)
            z_y = rng.normal(0.0, 1.0, size=shape)
        else:
            z_x = np.zeros(shape, dtype=np.float64)
            z_y = np.zeros(shape, dtype=np.float64)
        if config.ENABLE_THRESHOLD_NOISE:
            thresh_z_x = rng.normal(0.0, 1.0, size=n_receptors)
            thresh_z_y = rng.normal(0.0, 1.0, size=n_receptors)
        else:
            thresh_z_x = np.zeros(n_receptors, dtype=np.float64)
            thresh_z_y = np.zeros(n_receptors, dtype=np.float64)
        bank.append({
            'input_noise_x_unit': z_x,
            'input_noise_y_unit': z_y,
            'thresholds_x': np.clip(lif_model.v_thresh + sigma_thresh * thresh_z_x, 1e-6, None),
            'thresholds_y': np.clip(lif_model.v_thresh + sigma_thresh * thresh_z_y, 1e-6, None),
        })
    return bank


def _simulate_trial_response_cached(processed_cache, lif_model, gamma, sigma_n, trial_noise, include_mechanistic=False):
    use_gpu = _is_gpu_array(processed_cache['filtered_x']) or _is_gpu_array(processed_cache['filtered_y'])
    xp = cp if use_gpu else np
    current_x = gamma * processed_cache['filtered_x'] + sigma_n * xp.asarray(trial_noise['input_noise_x_unit'])
    current_y = gamma * processed_cache['filtered_y'] + sigma_n * xp.asarray(trial_noise['input_noise_y_unit'])
    thresholds_x = xp.asarray(trial_noise['thresholds_x']) if use_gpu else trial_noise['thresholds_x']
    thresholds_y = xp.asarray(trial_noise['thresholds_y']) if use_gpu else trial_noise['thresholds_y']
    binned_x, binned_y = lif_model.run_binned(
        current_x,
        current_y,
        processed_cache['window_start_idx'],
        processed_cache['samples_per_bin'],
        thresholds_x=thresholds_x,
        thresholds_y=thresholds_y,
    )
    response_vector = xp.stack((binned_x, binned_y), axis=1).reshape(-1)
    result = {'response_vector': response_vector}
    if include_mechanistic:
        spikes_x = xp.repeat(binned_x, processed_cache['samples_per_bin'], axis=1) / processed_cache['samples_per_bin']
        spikes_y = xp.repeat(binned_y, processed_cache['samples_per_bin'], axis=1) / processed_cache['samples_per_bin']
        mechanistic = decoder.compute_mechanistic_metrics(
            spikes_x,
            spikes_y,
            processed_cache['drive_x_win'],
            processed_cache['drive_y_win'],
            processed_cache['t_win'],
            receptor_coords,
            f0=config.PHASE_LOCK_F0_HZ,
        )
        result['mechanistic'] = mechanistic
    return result


def _build_baseline_processed(method_template):
    zeros_x = np.zeros_like(method_template['stress_xz'], dtype=np.float64)
    zeros_y = np.zeros_like(method_template['stress_yz'], dtype=np.float64)
    return stress_processor.process(
        zeros_x,
        zeros_y,
        method_template['roi_x'],
        method_template['roi_y'],
        receptor_coords,
        input_dt,
    )


def _build_calibration_processed(method_template):
    processed_variants = []
    for item in _generate_calibration_scales():
        processed_variants.append({
            'label': item['label'],
            'freq_hz': item['freq_hz'],
            'amp_scale': item['amp_scale'],
            'processed': stress_processor.process(
                np.asarray(method_template['stress_xz'], dtype=np.float64) * item['amp_scale'],
                np.asarray(method_template['stress_yz'], dtype=np.float64) * item['amp_scale'],
                method_template['roi_x'],
                method_template['roi_y'],
                receptor_coords,
                input_dt,
            ),
        })
    return processed_variants


def _calibration_objective(params, calibration_variants, baseline_cache, baseline_noise_bank, lif_model):
    gamma, sigma_n = params
    baseline_matrix = _make_response_matrix([
        _simulate_trial_response_cached(
            baseline_cache,
            lif_model,
            gamma,
            sigma_n,
            noise,
            include_mechanistic=False,
        )['response_vector']
        for noise in baseline_noise_bank
    ])
    baseline_pca_fit = decoder.fit_pca(baseline_matrix, variance_ratio=min(config.PCA_VARIANCE_RATIO, 0.99))
    baseline_projected = decoder.transform_pca(baseline_matrix)
    mu0 = baseline_projected.mean(axis=0)
    sigma0 = _ensure_2d_cov(decoder.compute_covariance(baseline_projected, reg_eps=config.COVARIANCE_REG_EPS))
    sigma0_inv = np.linalg.inv(sigma0)
    baseline_scores = _pair_mahalanobis_scores(baseline_projected, mu0, sigma0_inv)
    baseline_median = float(np.median(baseline_scores))

    monotonic_penalty = 0.0
    frequency_penalty = 0.0
    fit_penalty = 0.0
    by_freq = {}

    for item in calibration_variants:
        projected = decoder.transform_pca(_make_response_matrix([
            _simulate_trial_response_cached(
                item['cache'],
                lif_model,
                gamma,
                sigma_n,
                noise,
                include_mechanistic=False,
            )['response_vector']
            for noise in item['noise_bank']
        ]))
        scores = _pair_mahalanobis_scores(projected, mu0, sigma0_inv)
        detect_prob = float(np.mean(scores > baseline_median))
        target = _calibration_target_probability(item['freq_hz'], item['amp_scale'])
        fit_penalty += (detect_prob - target) ** 2
        by_freq.setdefault(item['freq_hz'], []).append((item['amp_scale'], detect_prob))

    for freq, seq in by_freq.items():
        seq = sorted(seq, key=lambda x: x[0])
        probs = [p for _, p in seq]
        for i in range(len(probs) - 1):
            if probs[i + 1] < probs[i]:
                monotonic_penalty += (probs[i] - probs[i + 1]) ** 2

    amps = sorted(set(config.CALIBRATION_AMPLITUDE_LEVELS))
    for amp in amps:
        p200 = next(p for a, p in by_freq[200.0] if abs(a - amp) < 1e-12)
        p400 = next(p for a, p in by_freq[400.0] if abs(a - amp) < 1e-12)
        if p400 > p200:
            frequency_penalty += (p400 - p200) ** 2

    objective = fit_penalty + 10.0 * monotonic_penalty + 10.0 * frequency_penalty
    return objective, {
        'gamma': float(gamma),
        'sigma_n': float(sigma_n),
        'baseline_pca': baseline_pca_fit,
    }


def _calibrate_global_parameters(calibration_variants, baseline_cache, lif_model, seed):
    gamma_min, gamma_max = config.CALIBRATION_GAMMA_BOUNDS
    sigma_min, sigma_max = config.CALIBRATION_SIGMA_BOUNDS
    max_iter = int(config.CALIBRATION_MAX_ITER)
    gamma_grid = np.geomspace(gamma_min, gamma_max, num=max_iter)
    sigma_grid = np.geomspace(max(sigma_min, 1e-4), sigma_max, num=max_iter)
    sigma_grid[0] = sigma_min

    rng = np.random.default_rng(seed)
    baseline_noise_bank = _build_trial_noise_bank(config.CALIBRATION_TRIALS_PER_LEVEL, baseline_cache, lif_model, rng)
    for item in calibration_variants:
        item['noise_bank'] = _build_trial_noise_bank(config.CALIBRATION_TRIALS_PER_LEVEL, item['cache'], lif_model, rng)

    coarse_to_fine = bool(getattr(config, 'CALIBRATION_COARSE_TO_FINE', False))
    topk = max(1, int(getattr(config, 'CALIBRATION_TOPK', 4)))
    if coarse_to_fine and max_iter >= 4:
        coarse_idx = np.unique(np.linspace(0, max_iter - 1, num=max(3, max_iter // 2), dtype=int))
        coarse_gamma_grid = gamma_grid[coarse_idx]
        coarse_sigma_grid = sigma_grid[coarse_idx]
    else:
        coarse_gamma_grid = gamma_grid
        coarse_sigma_grid = sigma_grid

    candidates = []
    best = None
    for gamma in coarse_gamma_grid:
        for sigma_n in coarse_sigma_grid:
            objective, details = _calibration_objective(
                (gamma, sigma_n),
                calibration_variants,
                baseline_cache,
                baseline_noise_bank,
                lif_model,
            )
            record = {
                'objective': float(objective),
                'gamma': float(gamma),
                'sigma_n': float(sigma_n),
                'details': details,
            }
            candidates.append(record)
            if best is None or objective < best['objective']:
                best = record

    if coarse_to_fine and len(candidates) > 0 and (len(coarse_gamma_grid) != len(gamma_grid) or len(coarse_sigma_grid) != len(sigma_grid)):
        candidate_keys = set()
        refined_points = []
        for record in sorted(candidates, key=lambda item: item['objective'])[:topk]:
            gamma_idx = int(np.argmin(np.abs(gamma_grid - record['gamma'])))
            sigma_idx = int(np.argmin(np.abs(sigma_grid - record['sigma_n'])))
            for gi in range(max(0, gamma_idx - 1), min(len(gamma_grid), gamma_idx + 2)):
                for si in range(max(0, sigma_idx - 1), min(len(sigma_grid), sigma_idx + 2)):
                    key = (gi, si)
                    if key in candidate_keys:
                        continue
                    candidate_keys.add(key)
                    refined_points.append((float(gamma_grid[gi]), float(sigma_grid[si])))

        for gamma, sigma_n in refined_points:
            objective, details = _calibration_objective(
                (gamma, sigma_n),
                calibration_variants,
                baseline_cache,
                baseline_noise_bank,
                lif_model,
            )
            if best is None or objective < best['objective']:
                best = {
                    'objective': float(objective),
                    'gamma': float(gamma),
                    'sigma_n': float(sigma_n),
                    'details': details,
                }

    print(
        f"Calibration complete: gamma={best['gamma']:.6f}, sigma_n={best['sigma_n']:.6f}, objective={best['objective']:.6f}"
    )
    print("All parameters were fixed before predicting Experiment 1 pairwise choices.")
    return best


def _collect_condition_trials(name, processed_cache, lif_model, gamma, sigma_n, n_trials, seed):
    rng = np.random.default_rng(seed)
    noise_bank = _build_trial_noise_bank(n_trials, processed_cache, lif_model, rng)
    response_vectors = []
    mechanistic_sum = {
        'mean_ffi': 0.0,
        'ndwci': 0.0,
        'dwci': 0.0,
        'rho_mean': 0.0,
        'rho_max': 0.0,
    }
    for noise in noise_bank:
        trial = _simulate_trial_response_cached(
            processed_cache,
            lif_model,
            gamma,
            sigma_n,
            noise,
            include_mechanistic=True,
        )
        response_vectors.append(trial['response_vector'])
        mech = trial['mechanistic']
        mechanistic_sum['mean_ffi'] += float(mech['mean_ffi'])
        mechanistic_sum['ndwci'] += float(mech['ndwci'])
        mechanistic_sum['dwci'] += float(mech['dwci'])
        mechanistic_sum['rho_mean'] += float(mech['rho_mean'])
        mechanistic_sum['rho_max'] += float(mech['rho_max'])
    trial_count = max(1, len(response_vectors))
    return {
        'name': name,
        'response_matrix': _make_response_matrix(response_vectors),
        'mechanistic': {
            'mean_ffi': mechanistic_sum['mean_ffi'] / trial_count,
            'ndwci': mechanistic_sum['ndwci'] / trial_count,
            'dwci': mechanistic_sum['dwci'] / trial_count,
            'rho_mean': mechanistic_sum['rho_mean'] / trial_count,
            'rho_max': mechanistic_sum['rho_max'] / trial_count,
        },
    }


def _project_all_conditions(condition_trials):
    all_vectors = [item['response_matrix'] for item in condition_trials.values()]
    pca_info = decoder.fit_pca(np.vstack(all_vectors), variance_ratio=config.PCA_VARIANCE_RATIO)
    print(
        f"PCA retained {pca_info['n_components']} components, explained variance={pca_info['explained_variance_ratio']:.4f}"
    )
    projected = {}
    for name, item in condition_trials.items():
        projected[name] = {
            'projected': decoder.transform_pca(item['response_matrix']),
            'mechanistic': item['mechanistic'],
        }
    return projected


def _compute_trial_metrics(projected_conditions):
    baseline = projected_conditions[config.BASELINE_CONDITION_NAME]['projected']
    xp = cp if _is_gpu_array(baseline) else np
    mu0 = baseline.mean(axis=0)
    sigma0 = _ensure_2d_cov(decoder.compute_covariance(baseline, reg_eps=config.COVARIANCE_REG_EPS))

    results = {}
    for method in config.STIMULUS_METHODS:
        base = projected_conditions[method]['projected']
        mu = base.mean(axis=0)
        sigma = _ensure_2d_cov(decoder.compute_covariance(base, reg_eps=config.COVARIANCE_REG_EPS))
        delta = base - mu0
        solved_delta = xp.linalg.solve(sigma0, delta.T).T
        intensity_trials = xp.sum(delta * solved_delta, axis=1)

        mu_px = projected_conditions[f'{method}__pos_x']['projected'].mean(axis=0)
        mu_nx = projected_conditions[f'{method}__neg_x']['projected'].mean(axis=0)
        mu_py = projected_conditions[f'{method}__pos_y']['projected'].mean(axis=0)
        mu_ny = projected_conditions[f'{method}__neg_y']['projected'].mean(axis=0)
        clarity_scalar, fisher = decoder.compute_fisher_clarity(
            mu_px,
            mu_nx,
            mu_py,
            mu_ny,
            sigma,
            config.POSITION_DELTA_MM,
        )
        clarity_trials = xp.full(base.shape[0], clarity_scalar, dtype=xp.float64)
        intensity_mean = decoder.compute_detectability(mu, mu0, sigma0)

        results[method] = {
            'intensity_trials': intensity_trials,
            'clarity_trials': clarity_trials,
            'intensity_mean': float(intensity_mean),
            'clarity_mean': float(clarity_scalar),
            'fisher_matrix': fisher,
            'mechanistic': projected_conditions[method]['mechanistic'],
        }
    return results


def _compute_pairwise_predictions(metric_results, metric_key):
    pairwise = []
    for a, b in _build_pair_list(config.PAIRWISE_METHODS):
        prob = _mean_choice_probability(metric_results[a][metric_key], metric_results[b][metric_key])
        pairwise.append({'A': a, 'B': b, 'probability': float(prob)})
    return _to_numpy(pairwise)


def _shift_stress_tensor(stress_tensor, roi_axis, delta_mm):
    roi_mm = np.asarray(roi_axis, dtype=np.float64) * 1000.0
    step_mm = float(np.mean(np.diff(roi_mm)))
    shift_bins = int(round(delta_mm / step_mm))
    shifted = np.roll(np.asarray(stress_tensor, dtype=np.float64), shift_bins, axis=1)
    if shift_bins > 0:
        shifted[:, :shift_bins, :] = 0.0
    elif shift_bins < 0:
        shifted[:, shift_bins:, :] = 0.0
    return shifted


def _shift_stress_tensor_y(stress_tensor, roi_axis, delta_mm):
    roi_mm = np.asarray(roi_axis, dtype=np.float64) * 1000.0
    step_mm = float(np.mean(np.diff(roi_mm)))
    shift_bins = int(round(delta_mm / step_mm))
    shifted = np.roll(np.asarray(stress_tensor, dtype=np.float64), shift_bins, axis=2)
    if shift_bins > 0:
        shifted[:, :, :shift_bins] = 0.0
    elif shift_bins < 0:
        shifted[:, :, shift_bins:] = 0.0
    return shifted


def _build_shifted_processed(method_data, delta_mm_x=0.0, delta_mm_y=0.0):
    stress_xz = np.asarray(method_data['stress_xz'], dtype=np.float64)
    stress_yz = np.asarray(method_data['stress_yz'], dtype=np.float64)
    if abs(delta_mm_x) > 0.0:
        stress_xz = _shift_stress_tensor(stress_xz, method_data['roi_x'], delta_mm_x)
        stress_yz = _shift_stress_tensor(stress_yz, method_data['roi_x'], delta_mm_x)
    if abs(delta_mm_y) > 0.0:
        stress_xz = _shift_stress_tensor_y(stress_xz, method_data['roi_y'], delta_mm_y)
        stress_yz = _shift_stress_tensor_y(stress_yz, method_data['roi_y'], delta_mm_y)
    return stress_processor.process(
        stress_xz,
        stress_yz,
        method_data['roi_x'],
        method_data['roi_y'],
        receptor_coords,
        input_dt,
    )


def _run_single_seed(seed, methods_data, input_dt_value, gpu_ready):
    global receptor_coords, stress_processor, decoder, input_dt
    input_dt = input_dt_value

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
        pca_window_ms=config.PCA_STEADY_STATE_WINDOW_MS,
        use_gpu=gpu_ready,
    )
    lif_model = LIFModel(
        tau_m=config.TAU_M_MS,
        v_rest=config.V_REST,
        v_reset=config.V_RESET,
        v_thresh=config.V_THRESH,
        r_m=config.R_M,
        t_ref=config.T_REF_MS,
        dt=config.DT_MS,
        use_gpu=gpu_ready,
    )
    decoder = PopulationDecoder(
        roi_area_mm2=config.ROI_SIZE_MM ** 2,
        density_sigma_mm=config.DENSITY_SIGMA_MM,
        density_grid_mm=config.DENSITY_GRID_MM,
        fidelity_freqs_hz=config.FIDELITY_FREQS_HZ,
        use_gpu=gpu_ready,
    )

    samples_per_bin = _bin_sample_count()
    reference_method = methods_data[config.STIMULUS_METHODS[0]]
    baseline_cache = _precompute_processed_cache(_build_baseline_processed(reference_method), samples_per_bin)
    calibration_variants = []
    for item in _build_calibration_processed(reference_method):
        calibration_variants.append({
            'label': item['label'],
            'freq_hz': item['freq_hz'],
            'amp_scale': item['amp_scale'],
            'cache': _precompute_processed_cache(item['processed'], samples_per_bin),
        })

    calibration = _calibrate_global_parameters(
        calibration_variants,
        baseline_cache,
        lif_model,
        seed + 17,
    )
    gamma = calibration['gamma']
    sigma_n = calibration['sigma_n']

    condition_trials = {
        config.BASELINE_CONDITION_NAME: _collect_condition_trials(
            config.BASELINE_CONDITION_NAME,
            baseline_cache,
            lif_model,
            gamma,
            sigma_n,
            config.TRIAL_COUNT,
            seed + 101,
        )
    }

    for idx, method in enumerate(tqdm(config.STIMULUS_METHODS, desc=f"Methods@seed={seed}")):
        if method not in methods_data:
            continue
        method_data = methods_data[method]
        center_cache = _precompute_processed_cache(
            stress_processor.process(
                method_data['stress_xz'],
                method_data['stress_yz'],
                method_data['roi_x'],
                method_data['roi_y'],
                receptor_coords,
                input_dt,
            ),
            samples_per_bin,
        )
        condition_trials[method] = _collect_condition_trials(
            method,
            center_cache,
            lif_model,
            gamma,
            sigma_n,
            config.TRIAL_COUNT,
            seed + 1000 + idx,
        )
        condition_trials[f'{method}__pos_x'] = _collect_condition_trials(
            f'{method}__pos_x',
            _precompute_processed_cache(_build_shifted_processed(method_data, delta_mm_x=+config.POSITION_DELTA_MM), samples_per_bin),
            lif_model,
            gamma,
            sigma_n,
            config.TRIAL_COUNT,
            seed + 2000 + idx,
        )
        condition_trials[f'{method}__neg_x'] = _collect_condition_trials(
            f'{method}__neg_x',
            _precompute_processed_cache(_build_shifted_processed(method_data, delta_mm_x=-config.POSITION_DELTA_MM), samples_per_bin),
            lif_model,
            gamma,
            sigma_n,
            config.TRIAL_COUNT,
            seed + 3000 + idx,
        )
        condition_trials[f'{method}__pos_y'] = _collect_condition_trials(
            f'{method}__pos_y',
            _precompute_processed_cache(_build_shifted_processed(method_data, delta_mm_y=+config.POSITION_DELTA_MM), samples_per_bin),
            lif_model,
            gamma,
            sigma_n,
            config.TRIAL_COUNT,
            seed + 4000 + idx,
        )
        condition_trials[f'{method}__neg_y'] = _collect_condition_trials(
            f'{method}__neg_y',
            _precompute_processed_cache(_build_shifted_processed(method_data, delta_mm_y=-config.POSITION_DELTA_MM), samples_per_bin),
            lif_model,
            gamma,
            sigma_n,
            config.TRIAL_COUNT,
            seed + 5000 + idx,
        )

    projected_conditions = _project_all_conditions(condition_trials)
    metric_results = _compute_trial_metrics(projected_conditions)
    intensity_pairwise = _compute_pairwise_predictions(metric_results, 'intensity_trials')
    clarity_pairwise = _compute_pairwise_predictions(metric_results, 'clarity_trials')

    _print_method_summary(f"Seed {seed} method summary", metric_results, config.STIMULUS_METHODS)
    _print_pairwise_table(f"Seed {seed} intensity 2-AFC", intensity_pairwise)
    _print_pairwise_table(f"Seed {seed} clarity 2-AFC", clarity_pairwise)

    return {
        'seed': seed,
        'gamma': float(gamma),
        'sigma_n': float(sigma_n),
        'method_results': _to_numpy(metric_results),
        'intensity_pairwise': _to_numpy(intensity_pairwise),
        'clarity_pairwise': _to_numpy(clarity_pairwise),
    }


def _aggregate_runs(all_runs):
    aggregated_methods = {}
    for method in config.STIMULUS_METHODS:
        method_runs = [run['method_results'][method] for run in all_runs if method in run['method_results']]
        if not method_runs:
            continue
        aggregated_methods[method] = {
            'intensity_mean': float(np.median([item['intensity_mean'] for item in method_runs])),
            'clarity_mean': float(np.median([item['clarity_mean'] for item in method_runs])),
            'mechanistic': {
                'mean_ffi': float(np.median([item['mechanistic']['mean_ffi'] for item in method_runs])),
                'ndwci': float(np.median([item['mechanistic']['ndwci'] for item in method_runs])),
            },
        }

    def aggregate_pairwise(key):
        first = all_runs[0][key]
        out = []
        for idx, item in enumerate(first):
            vals = [run[key][idx]['probability'] for run in all_runs]
            out.append({'A': item['A'], 'B': item['B'], 'probability': float(np.median(vals))})
        return out

    return aggregated_methods, aggregate_pairwise('intensity_pairwise'), aggregate_pairwise('clarity_pairwise')


def main():
    print("=== Computational Neural Dynamics Model ===")
    print(f"Config: FS={config.FS_MODEL}, DT={config.DT_MS}, GPU={config.USE_GPU}")
    use_gpu = bool(getattr(config, 'USE_GPU', False))
    gpu_ready = bool(use_gpu and cp is not None and cuda.is_available())
    if gpu_ready:
        cuda.select_device(getattr(config, 'GPU_DEVICE_ID', 0))
        print("Backend: GPU preprocessing available")
    else:
        print("Backend: CPU")

    print(f"Loading data from {config.DATA_FILE}...")
    if not os.path.exists(config.DATA_FILE):
        print("Error: Data file not found. Please run the K-Wave simulation first.")
        return

    loader = DataLoader(config.DATA_FILE)
    data = loader.load()
    methods_data = data['methods']
    input_dt_value = data.get('dt', 0.1 / 1000)

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
        t0 = time.time()
        run_result = _run_single_seed(seed, methods_data, input_dt_value, gpu_ready)
        all_runs.append(run_result)
        print(f"Seed {seed} finished in {time.time() - t0:.2f}s")

    aggregated_methods, aggregated_intensity, aggregated_clarity = _aggregate_runs(all_runs)

    print("\n=== Seed-by-seed parameter summary ===")
    for run in all_runs:
        print(f"Seed {run['seed']}: gamma={run['gamma']:.6f}, sigma_n={run['sigma_n']:.6f}")

    _print_method_summary("Median metrics across seeds", aggregated_methods, config.STIMULUS_METHODS)
    _print_pairwise_table("Median intensity 2-AFC across seeds", aggregated_intensity)
    _print_pairwise_table("Median clarity 2-AFC across seeds", aggregated_clarity)
    print("\nDone.")


if __name__ == '__main__':
    main()
