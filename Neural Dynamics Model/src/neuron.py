import numpy as np
from numba import njit, prange
from numba import cuda
try:
    import cupy as cp
except Exception:
    cp = None


class LIFModel:
    def __init__(self, tau_m, v_rest, v_reset, v_thresh, r_m, t_ref, dt, use_gpu=False):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.r_m = r_m
        self.t_ref = t_ref
        self.dt = dt
        self.use_gpu = bool(use_gpu and cuda.is_available())

        self.ref_steps = int(t_ref / dt)
        print(f"DEBUG: LIFModel initialized. dt={dt}, t_ref={t_ref}, ref_steps={self.ref_steps}")

    def run(self, input_current):
        if self.use_gpu:
            return run_lif_cuda(
                input_current,
                self.v_rest,
                self.v_reset,
                self.v_thresh,
                self.r_m,
                self.tau_m,
                self.dt,
                self.ref_steps
            )
        return run_lif_numba_parallel(
            input_current,
            self.v_rest,
            self.v_reset,
            self.v_thresh,
            self.r_m,
            self.tau_m,
            self.dt,
            self.ref_steps
        )

    def run_counts(self, input_current):
        if self.use_gpu:
            return run_lif_count_cuda(
                input_current,
                self.v_rest,
                self.v_reset,
                self.v_thresh,
                self.r_m,
                self.tau_m,
                self.dt,
                self.ref_steps
            )
        return run_lif_count_numba_parallel(
            input_current,
            self.v_rest,
            self.v_reset,
            self.v_thresh,
            self.r_m,
            self.tau_m,
            self.dt,
            self.ref_steps
        )

    @staticmethod
    def compute_current(filtered_stress, gain):
        if cp is not None and isinstance(filtered_stress, cp.ndarray):
            return cp.where(filtered_stress > 0.0, filtered_stress, 0.0) * gain
        return np.where(filtered_stress > 0.0, filtered_stress, 0.0) * gain


def _is_cupy_array(x):
    return cp is not None and isinstance(x, cp.ndarray)


def _to_host_counts(counts):
    if _is_cupy_array(counts):
        return cp.asnumpy(counts)
    return counts


@njit(cache=True, fastmath=False, parallel=True)
def run_lif_numba_parallel(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps):
    N, T = input_current.shape
    spike_train = np.zeros((N, T), dtype=np.bool_)

    alpha = dt / tau_m

    # Parallelize across neurons; each neuron's temporal dynamics are independent.
    for i in prange(N):
        v_i = v_rest
        ref_i = 0
        for t in range(T):
            if ref_i > 0:
                ref_i -= 1
                v_i = v_reset
            else:
                dv = alpha * (-(v_i - v_rest) + r_m * input_current[i, t])
                v_i += dv

                if v_i >= v_thresh:
                    spike_train[i, t] = True
                    v_i = v_reset
                    ref_i = ref_steps

    return spike_train


@njit(cache=True, fastmath=False, parallel=True)
def run_lif_count_numba_parallel(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps):
    N, T = input_current.shape
    spike_counts = np.zeros(N, dtype=np.int64)

    alpha = dt / tau_m

    for i in prange(N):
        v_i = v_rest
        ref_i = 0
        count_i = 0
        for t in range(T):
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

    return spike_counts


@cuda.jit
def _run_lif_cuda_kernel(input_current, spike_train, v_rest, v_reset, v_thresh, r_m, alpha, ref_steps):
    i = cuda.grid(1)
    N = input_current.shape[0]
    T = input_current.shape[1]
    if i >= N:
        return

    v_i = v_rest
    ref_i = 0
    for t in range(T):
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
    N = input_current.shape[0]
    T = input_current.shape[1]
    if i >= N:
        return

    v_i = v_rest
    ref_i = 0
    count_i = 0
    for t in range(T):
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


def run_lif_cuda(input_current, v_rest, v_reset, v_thresh, r_m, tau_m, dt, ref_steps):
    alpha = dt / tau_m
    is_cupy_input = _is_cupy_array(input_current)
    
    if is_cupy_input:
        current_gpu = input_current
        # Initialize with zeros using cupy
        spike_gpu = cp.zeros(current_gpu.shape, dtype=cp.uint8)
    else:
        current_gpu = cuda.to_device(np.asarray(input_current, dtype=np.float64))
        # Initialize with zeros using numpy -> device to ensure clean memory
        spike_gpu = cuda.to_device(np.zeros(current_gpu.shape, dtype=np.uint8))

    N, T = current_gpu.shape
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
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
    N = current_gpu.shape[0]
    counts_gpu = cuda.device_array(N, dtype=np.int64)
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
    _run_lif_count_cuda_kernel[blocks, threads_per_block](
        current_gpu, counts_gpu, v_rest, v_reset, v_thresh, r_m, alpha, ref_steps
    )
    if is_cupy_input:
        return cp.asarray(counts_gpu)
    return counts_gpu.copy_to_host()
