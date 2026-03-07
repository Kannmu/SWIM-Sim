import numpy as np
import scipy.signal
import scipy.ndimage

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupyx.scipy.signal as cpx_signal
except Exception:
    cp = None
    cpx_ndimage = None
    cpx_signal = None


class StressProcessor:
    def __init__(self, fs, filter_order, f_low, f_high, spatial_sigma, pca_window_ms, use_gpu=False):
        self.fs = fs
        self.filter_order = filter_order
        self.f_low = f_low
        self.f_high = f_high
        self.spatial_sigma = spatial_sigma
        self.pca_window_ms = pca_window_ms
        self.use_gpu = bool(use_gpu and cp is not None)

        self.sos = scipy.signal.butter(
            N=self.filter_order,
            Wn=[self.f_low, self.f_high],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        self.sos_gpu = cp.asarray(self.sos) if self.use_gpu else None

        self._cache = {}

    def _get_interpolation_cache(self, T, x_vec, y_vec, receptor_coords):
        x_vec = np.asarray(x_vec)
        y_vec = np.asarray(y_vec)
        receptor_coords = np.asarray(receptor_coords)

        cache_key = (
            int(T),
            x_vec.shape,
            y_vec.shape,
            float(x_vec[0]),
            float(x_vec[-1]),
            float(y_vec[0]),
            float(y_vec[-1]),
            receptor_coords.shape,
            np.ascontiguousarray(receptor_coords).tobytes(),
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        n_receptors = receptor_coords.shape[0]
        x_vec_mm = (x_vec * 1000.0).astype(np.float64, copy=False)
        y_vec_mm = (y_vec * 1000.0).astype(np.float64, copy=False)

        x0_idx = np.searchsorted(x_vec_mm, receptor_coords[:, 0], side='right') - 1
        y0_idx = np.searchsorted(y_vec_mm, receptor_coords[:, 1], side='right') - 1
        x0_idx = np.clip(x0_idx, 0, x_vec_mm.size - 2)
        y0_idx = np.clip(y0_idx, 0, y_vec_mm.size - 2)
        x1_idx = x0_idx + 1
        y1_idx = y0_idx + 1

        x0 = x_vec_mm[x0_idx]
        x1 = x_vec_mm[x1_idx]
        y0 = y_vec_mm[y0_idx]
        y1 = y_vec_mm[y1_idx]
        wx = (receptor_coords[:, 0] - x0) / (x1 - x0 + 1e-15)
        wy = (receptor_coords[:, 1] - y0) / (y1 - y0 + 1e-15)
        wx = np.clip(wx, 0.0, 1.0)
        wy = np.clip(wy, 0.0, 1.0)

        cached = {
            "x0_idx": x0_idx.astype(np.int64, copy=False),
            "x1_idx": x1_idx.astype(np.int64, copy=False),
            "y0_idx": y0_idx.astype(np.int64, copy=False),
            "y1_idx": y1_idx.astype(np.int64, copy=False),
            "wx": wx.astype(np.float64, copy=False),
            "wy": wy.astype(np.float64, copy=False),
            "n_receptors": n_receptors,
        }
        self._cache[cache_key] = cached
        return cached

    @staticmethod
    def _build_sos_zi_array(zi, first_sample):
        zi = np.asarray(zi, dtype=np.float64)
        first_sample = np.asarray(first_sample, dtype=np.float64)
        n_sections = zi.shape[0]
        n_states = zi.shape[-1]
        n_receptors = first_sample.shape[0]
        zi_array = np.empty((n_sections, n_receptors, n_states), dtype=np.float64)
        for sec in range(n_sections):
            zi_array[sec, :, :] = zi[sec][None, :] * first_sample[:, None]
        return zi_array

    @staticmethod
    def _build_sos_zi_array_gpu(zi, first_sample):
        n_sections = zi.shape[0]
        n_states = zi.shape[-1]
        n_receptors = first_sample.shape[0]
        zi_array = cp.empty((n_sections, n_receptors, n_states), dtype=cp.float64)
        for sec in range(n_sections):
            zi_array[sec, :, :] = zi[sec][None, :] * first_sample[:, None]
        return zi_array

    def _interpolate_gpu(self, smoothed_stress, cache):
        x0 = cp.asarray(cache["x0_idx"])
        x1 = cp.asarray(cache["x1_idx"])
        y0 = cp.asarray(cache["y0_idx"])
        y1 = cp.asarray(cache["y1_idx"])
        wx = cp.asarray(cache["wx"])
        wy = cp.asarray(cache["wy"])

        v00 = smoothed_stress[:, x0, y0]
        v01 = smoothed_stress[:, x0, y1]
        v10 = smoothed_stress[:, x1, y0]
        v11 = smoothed_stress[:, x1, y1]

        w00 = (1.0 - wx) * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w10 = wx * (1.0 - wy)
        w11 = wx * wy

        raw_tn = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
        return cp.transpose(raw_tn, (1, 0))

    def _interpolate_cpu(self, smoothed_stress, cache):
        x0 = cache["x0_idx"]
        x1 = cache["x1_idx"]
        y0 = cache["y0_idx"]
        y1 = cache["y1_idx"]
        wx = cache["wx"]
        wy = cache["wy"]

        v00 = smoothed_stress[:, x0, y0]
        v01 = smoothed_stress[:, x0, y1]
        v10 = smoothed_stress[:, x1, y0]
        v11 = smoothed_stress[:, x1, y1]

        w00 = (1.0 - wx) * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w10 = wx * (1.0 - wy)
        w11 = wx * wy

        raw_tn = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
        return np.transpose(raw_tn, (1, 0))

    def _resample_signals(self, signals, input_fs, target_len=None):
        current_len = signals.shape[-1]
        if target_len is None:
            if abs(input_fs - self.fs) <= 1.0:
                return signals
            target_len = int(round(current_len * self.fs / input_fs))

        if self.use_gpu:
            return cpx_signal.resample(signals, target_len, axis=-1)
        return scipy.signal.resample(signals, target_len, axis=-1)

    def _filter_component(self, stress_component):
        if self.use_gpu:
            detrended = stress_component - cp.mean(stress_component, axis=1, keepdims=True)
            zi = cpx_signal.sosfilt_zi(self.sos_gpu)
            zi_array = self._build_sos_zi_array_gpu(zi, detrended[:, 0])
            filtered, _ = cpx_signal.sosfilt(self.sos_gpu, detrended, axis=1, zi=zi_array)
            return filtered

        detrended = stress_component - np.mean(stress_component, axis=1, keepdims=True)
        zi = scipy.signal.sosfilt_zi(self.sos)
        zi_array = self._build_sos_zi_array(zi, detrended[:, 0])
        filtered, _ = scipy.signal.sosfilt(self.sos, detrended, axis=1, zi=zi_array)
        return filtered

    def _compute_frequency_power(self, signal_component, freq_hz):
        xp = cp if self.use_gpu else np
        signal_arr = xp.asarray(signal_component, dtype=xp.float64)
        n_time = signal_arr.shape[1]
        if n_time <= 0:
            raise ValueError("Continuous receptor drive must contain at least one sample.")
        t_vec = xp.arange(n_time, dtype=xp.float64) / self.fs
        phase = xp.exp(-1j * 2.0 * xp.pi * float(freq_hz) * t_vec)
        coeff = xp.sum(signal_arr * phase[None, :], axis=1)
        return xp.abs(coeff) ** 2

    def compute_drive_frequency_fidelity(self, drive_x, drive_y, fidelity_freqs_hz, target_freq_hz=200.0):
        xp = cp if self.use_gpu else np
        power_sum = None
        target_power = None

        for freq in fidelity_freqs_hz:
            px = self._compute_frequency_power(drive_x, freq)
            py = self._compute_frequency_power(drive_y, freq)
            power = px + py
            if int(round(freq)) == int(round(target_freq_hz)):
                target_power = power
            power_sum = power if power_sum is None else (power_sum + power)

        if target_power is None:
            raise ValueError("Continuous-drive fidelity requires the target frequency in fidelity_freqs_hz.")

        return target_power / (power_sum + 1e-12)

    def process(self, stress_xz, stress_yz, x_vec, y_vec, receptor_coords, original_dt):
        if stress_xz is None or stress_yz is None:
            raise ValueError("Signed tangential stress inputs stress_xz and stress_yz are required.")

        if stress_xz.shape != stress_yz.shape:
            raise ValueError(f"stress_xz and stress_yz shape mismatch: {stress_xz.shape} vs {stress_yz.shape}")

        t_samples, _, _ = stress_xz.shape
        dx_mm = (x_vec[1] - x_vec[0]) * 1000.0
        dy_mm = (y_vec[1] - y_vec[0]) * 1000.0
        sigma_pixels_x = self.spatial_sigma / dx_mm
        sigma_pixels_y = self.spatial_sigma / dy_mm
        cache = self._get_interpolation_cache(t_samples, x_vec, y_vec, receptor_coords)
        input_fs = 1.0 / original_dt

        if self.use_gpu:
            xz_gpu = cp.asarray(stress_xz, dtype=cp.float64)
            yz_gpu = cp.asarray(stress_yz, dtype=cp.float64)

            xz_smooth = cpx_ndimage.gaussian_filter(xz_gpu, sigma=[0, sigma_pixels_x, sigma_pixels_y], mode='nearest')
            yz_smooth = cpx_ndimage.gaussian_filter(yz_gpu, sigma=[0, sigma_pixels_x, sigma_pixels_y], mode='nearest')

            xz_raw = self._interpolate_gpu(xz_smooth, cache)
            yz_raw = self._interpolate_gpu(yz_smooth, cache)

            target_len = None if abs(input_fs - self.fs) <= 1.0 else int(round(t_samples * self.fs / input_fs))
            xz_drive = self._resample_signals(xz_raw, input_fs, target_len=target_len)
            yz_drive = self._resample_signals(yz_raw, input_fs, target_len=target_len)
            xz_filtered = self._filter_component(xz_drive)
            yz_filtered = self._filter_component(yz_drive)
            n_time = xz_filtered.shape[1]
            t_vec_new = cp.arange(n_time, dtype=cp.float64) / self.fs

            if cp.any(cp.isnan(xz_filtered)) or cp.any(cp.isinf(xz_filtered)):
                raise ValueError("NaNs or Infs detected in xz_filtered (GPU)")
            if cp.any(cp.isnan(yz_filtered)) or cp.any(cp.isinf(yz_filtered)):
                raise ValueError("NaNs or Infs detected in yz_filtered (GPU)")

            return {
                'xz': xz_filtered,
                'yz': yz_filtered,
                'drive_xz': xz_drive,
                'drive_yz': yz_drive,
                't': t_vec_new,
            }

        xz_smooth = scipy.ndimage.gaussian_filter(stress_xz, sigma=[0, sigma_pixels_x, sigma_pixels_y], mode='nearest')
        yz_smooth = scipy.ndimage.gaussian_filter(stress_yz, sigma=[0, sigma_pixels_x, sigma_pixels_y], mode='nearest')

        xz_raw = self._interpolate_cpu(xz_smooth, cache)
        yz_raw = self._interpolate_cpu(yz_smooth, cache)

        target_len = None if abs(input_fs - self.fs) <= 1.0 else int(round(t_samples * self.fs / input_fs))
        xz_drive = self._resample_signals(xz_raw, input_fs, target_len=target_len)
        yz_drive = self._resample_signals(yz_raw, input_fs, target_len=target_len)

        xz_filtered = self._filter_component(xz_drive)
        yz_filtered = self._filter_component(yz_drive)
        n_time = xz_filtered.shape[1]
        t_vec_new = np.arange(n_time, dtype=np.float64) / self.fs

        if np.any(np.isnan(xz_filtered)) or np.any(np.isinf(xz_filtered)):
            raise ValueError("NaNs or Infs detected in xz_filtered (CPU)")
        if np.any(np.isnan(yz_filtered)) or np.any(np.isinf(yz_filtered)):
            raise ValueError("NaNs or Infs detected in yz_filtered (CPU)")

        return {
            'xz': xz_filtered,
            'yz': yz_filtered,
            'drive_xz': xz_drive,
            'drive_yz': yz_drive,
            't': t_vec_new,
        }
