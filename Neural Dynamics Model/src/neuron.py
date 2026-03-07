import numpy as np
from numba import cuda, njit, prange

try:
    import cupy as cp
except Exception:
    cp = None


def _is_cupy_array(x):
    return cp is not None and isinstance(x, cp.ndarray)


class LIFModel:
    def __init__(self, tau_m, v_rest, v_reset, v_thresh, r_m, t_ref, dt, use_gpu=False):
        self.tau_m = float(tau_m)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = float(v_thresh)
        self.r_m = float(r_m)
        self.t_ref = float(t_ref)
        self.dt = float(dt)
        self.use_gpu = bool(use_gpu and cuda.is_available())
        self.ref_steps = int(round(self.t_ref / self.dt))

    def run_single(self, input_current, thresholds=None):
        if self.use_gpu and thresholds is None:
            return run_lif_cuda(
                input_current,
                self.v_rest,
                self.v_reset,
                self.v_thresh,
                self.r_m,
                self.tau_m,
                self.dt,
                self.ref_steps,
            )
        return run_lif_numba_parallel(
            np.asarray(input_current, dtype=np.float64),
            self.v_rest,
            self.v_reset,
            self.v_thresh,
            self.r_m,
            self.tau_m,
            self.dt,
            self.ref_steps,
            None if thresholds is None else np.asarray(thresholds, dtype=np.float64),
        )

    def run(self, currents_x, currents_y, thresholds_x=None, thresholds_y=None):
        spikes_x = self.run_single(currents_x, thresholds=thresholds_x)
        spikes_y = self.run_single(currents_y, thresholds=thresholds_y)
        return spikes_x, spikes_y

    def run_counts_single(self, input_current, thresholds=None):
        if self.use_gpu and thresholds is None:
            return run_lif_count_cuda(
                input_current,
                self.v_rest,
                self.v_reset,
                self.v_thresh,
                self.r_m,
                self.tau_m,
                self.dt,
                self.ref_steps,
            )
        return run_lif_count_numba_parallel(
            np.asarray(input_current, dtype=np.float64),
            self.v_rest,
            self.v_reset,
            self.v_thresh,
            self.r_m,
            self.tau_m,
            self.dt,
            self.ref_steps,
            None if thresholds is None else np.asarray(thresholds, dtype=np.float64),
        )

    def run_counts(self, currents_x, currents_y, thresholds_x=None, thresholds_y=None):
        counts_x = self.run_counts_single(currents_x, thresholds=thresholds_x)
        counts_y = self.run_counts_single(currents_y, thresholds=thresholds_y)
        return counts_x, counts_y

    def run_binned(self, currents_x, currents_y, window_start_idx, samples_per_bin, thresholds_x=None, thresholds_y=None):
        binned_x = self.run_binned_single(
            currents_x,
            window_start_idx,
            samples_per_bin,
            thresholds=thresholds_x,
        )
        binned_y = self.run_binned_single(
            currents_y,
            window_start_idx,
            samples_per_bin,
            thresholds=thresholds_y,
        )
        return binned_x, binned_y

    def run_binned_single(self, input_current, window_start_idx, samples_per_bin, thresholds=None):
        if self.use_gpu:
            return run_lif_binned_cuda(
                input_current,
                self.v_rest,
                self.v_reset,
                self.v_thresh,
                self.r_m,
                self.tau_m,
                self.dt,
                self.ref_steps,
                int(window_start_idx),
                int(samples_per_bin),
                thresholds,
            )
        return run_lif_binned_numba_parallel(
            np.asarray(input_current, dtype=np.float64),
            self.v_rest,
            self.v_reset,
            self.v_thresh,
            self.r_m,
            self.tau_m,
            self.dt,
            self.ref_steps,
            int(window_start_idx),
            int(samples_per_bin),
            None if thresholds is None else np.asarray(thresholds, dtype=np.float64),
        )

    @staticmethod
    def compute_current(filtered_stress, gain):
        if cp is not None and isinstance(filtered_stress, cp.ndarray):
            return filtered_stress * float(gain)
        return np.asarray(filtered_stress, dtype=np.float64) * float(gain)

    def compute_currents(self, filtered_xz, filtered_yz, gain):
        current_x = self.compute_current(filtered_xz, gain)
        current_y = self.compute_current(filtered_yz, gain)
        return current_x, current_y


@njit(cache=True, fastmath=False, parallel=True)
def run_lif_numba_parallel(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps, thresholds):
    n_receptors, n_time = input_current.shape
    spike_train = np.zeros((n_receptors, n_time), dtype=np.bool_)
    alpha = dt / tau_m

    for i in prange(n_receptors):
        v_i = v_rest
        ref_i = 0
        thresh_i = v_thresh if thresholds is None else thresholds[i]
        for t in range(n_time):
            if ref_i > 0:
                ref_i -= 1
                v_i = v_reset
            else:
                dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
                v_i += dv
                if v_i >= thresh_i:
                    spike_train[i, t] = True
                    v_i = v_reset
                    ref_i = ref_steps
    return spike_train


@njit(cache=True, fastmath=False, parallel=True)
def run_lif_count_numba_parallel(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps, thresholds):
    n_receptors, n_time = input_current.shape
    spike_counts = np.zeros(n_receptors, dtype=np.int64)
    alpha = dt / tau_m

    for i in prange(n_receptors):
        v_i = v_rest
        ref_i = 0
        count_i = 0
        thresh_i = v_thresh if thresholds is None else thresholds[i]
        for t in range(n_time):
            if ref_i > 0:
                ref_i -= 1
                v_i = v_reset
            else:
                dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
                v_i += dv
                if v_i >= thresh_i:
                    count_i += 1
                    v_i = v_reset
                    ref_i = ref_steps
        spike_counts[i] = count_i
    return spike_counts


@njit(cache=True, fastmath=False, parallel=True)
def run_lif_binned_numba_parallel(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps, window_start_idx, samples_per_bin, thresholds):
    n_receptors, n_time = input_current.shape
    n_window_samples = n_time - window_start_idx
    if n_window_samples <= 0:
        raise ValueError("window_start_idx must be smaller than the number of time samples")
    n_bins = n_window_samples // samples_per_bin
    if n_bins <= 0:
        raise ValueError("Window does not contain enough samples for the requested bin width")
    valid_window_samples = n_bins * samples_per_bin
    window_end_idx = window_start_idx + valid_window_samples
    binned_counts = np.zeros((n_receptors, n_bins), dtype=np.float64)
    alpha = dt / tau_m

    for i in prange(n_receptors):
        v_i = v_rest
        ref_i = 0
        thresh_i = v_thresh if thresholds is None else thresholds[i]
        for t in range(n_time):
            spiked = False
            if ref_i > 0:
                ref_i -= 1
                v_i = v_reset
            else:
                dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
                v_i += dv
                if v_i >= thresh_i:
                    spiked = True
                    v_i = v_reset
                    ref_i = ref_steps
            if spiked and t >= window_start_idx and t < window_end_idx:
                bin_idx = (t - window_start_idx) // samples_per_bin
                binned_counts[i, bin_idx] += 1.0
    return binned_counts


@cuda.jit
def _run_lif_cuda_kernel(input_current, spike_train, v_rest, v_reset, v_thresh, r_m, alpha, ref_steps):
    i = cuda.grid(1)
    n_receptors = input_current.shape[0]
    n_time = input_current.shape[1]
    if i >= n_receptors:
        return

    v_i = v_rest
    ref_i = 0
    for t in range(n_time):
        if ref_i > 0:
            ref_i -= 1
            v_i = v_reset
        else:
            dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
            v_i += dv
            if v_i >= v_thresh:
                spike_train[i, t] = 1
                v_i = v_reset
                ref_i = ref_steps


@cuda.jit
def _run_lif_count_cuda_kernel(input_current, spike_counts, v_rest, v_reset, v_thresh, r_m, alpha, ref_steps):
    i = cuda.grid(1)
    n_receptors = input_current.shape[0]
    n_time = input_current.shape[1]
    if i >= n_receptors:
        return

    v_i = v_rest
    ref_i = 0
    count_i = 0
    for t in range(n_time):
        if ref_i > 0:
            ref_i -= 1
            v_i = v_reset
        else:
            dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
            v_i += dv
            if v_i >= v_thresh:
                count_i += 1
                v_i = v_reset
                ref_i = ref_steps
    spike_counts[i] = count_i


@cuda.jit
def _run_lif_binned_cuda_kernel(
    input_current,
    thresholds,
    binned_counts,
    v_rest,
    v_reset,
    v_thresh,
    r_m,
    alpha,
    ref_steps,
    window_start_idx,
    window_end_idx,
    samples_per_bin,
    use_thresholds,
):
    i = cuda.grid(1)
    n_receptors = input_current.shape[0]
    n_time = input_current.shape[1]
    if i >= n_receptors:
        return

    v_i = v_rest
    ref_i = 0
    thresh_i = thresholds[i] if use_thresholds else v_thresh
    for t in range(n_time):
        spiked = False
        if ref_i > 0:
            ref_i -= 1
            v_i = v_reset
        else:
            dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
            v_i += dv
            if v_i >= thresh_i:
                spiked = True
                v_i = v_reset
                ref_i = ref_steps
        if spiked and t >= window_start_idx and t < window_end_idx:
            bin_idx = (t - window_start_idx) // samples_per_bin
            binned_counts[i, bin_idx] += 1.0


def run_lif_cuda(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps):
    alpha = dt / tau_m
    is_cupy_input = _is_cupy_array(input_current)

    if is_cupy_input:
        current_gpu = input_current
        spike_gpu = cp.zeros(current_gpu.shape, dtype=cp.uint8)
    else:
        current_gpu = cuda.to_device(np.asarray(input_current, dtype=np.float64))
        spike_gpu = cuda.to_device(np.zeros(current_gpu.shape, dtype=np.uint8))

    n_receptors, _ = current_gpu.shape
    threads_per_block = 256
    blocks = (n_receptors + threads_per_block - 1) // threads_per_block
    _run_lif_cuda_kernel[blocks, threads_per_block](
        current_gpu, spike_gpu, v_rest, v_reset, v_thresh, r_m, alpha, ref_steps
    )

    if is_cupy_input:
        return spike_gpu.astype(cp.bool_)
    return spike_gpu.copy_to_host().astype(np.bool_)


def run_lif_count_cuda(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps):
    alpha = dt / tau_m
    is_cupy_input = _is_cupy_array(input_current)
    current_gpu = input_current if is_cupy_input else cuda.to_device(np.asarray(input_current, dtype=np.float64))
    n_receptors = current_gpu.shape[0]
    counts_gpu = cuda.device_array(n_receptors, dtype=np.int64)
    threads_per_block = 256
    blocks = (n_receptors + threads_per_block - 1) // threads_per_block
    _run_lif_count_cuda_kernel[blocks, threads_per_block](
        current_gpu, counts_gpu, v_rest, v_reset, v_thresh, r_m, alpha, ref_steps
    )
    if is_cupy_input:
        return cp.asarray(counts_gpu)
    return counts_gpu.copy_to_host()


def run_lif_binned_cuda(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps, window_start_idx, samples_per_bin, thresholds=None):
    input_is_cupy = _is_cupy_array(input_current)
    thresholds_is_cupy = _is_cupy_array(thresholds)
    alpha = dt / tau_m

    current_gpu = input_current if input_is_cupy else cp.asarray(np.asarray(input_current, dtype=np.float64))
    n_receptors, n_time = current_gpu.shape
    n_window_samples = n_time - int(window_start_idx)
    if n_window_samples <= 0:
        raise ValueError("window_start_idx must be smaller than the number of time samples")
    n_bins = n_window_samples // int(samples_per_bin)
    if n_bins <= 0:
        raise ValueError("Window does not contain enough samples for the requested bin width")
    valid_window_samples = n_bins * int(samples_per_bin)
    window_end_idx = int(window_start_idx) + valid_window_samples

    thresholds_gpu = cp.empty((1,), dtype=cp.float64)
    use_thresholds = 0
    if thresholds is not None:
        thresholds_gpu = thresholds if thresholds_is_cupy else cp.asarray(np.asarray(thresholds, dtype=np.float64))
        use_thresholds = 1

    binned_gpu = cp.zeros((n_receptors, n_bins), dtype=cp.float64)
    threads_per_block = 256
    blocks = (n_receptors + threads_per_block - 1) // threads_per_block
    _run_lif_binned_cuda_kernel[blocks, threads_per_block](
        current_gpu,
        thresholds_gpu,
        binned_gpu,
        v_rest,
        v_reset,
        v_thresh,
        r_m,
        alpha,
        ref_steps,
        int(window_start_idx),
        int(window_end_idx),
        int(samples_per_bin),
        use_thresholds,
    )
    return binned_gpu if input_is_cupy or thresholds_is_cupy else cp.asnumpy(binned_gpu)
