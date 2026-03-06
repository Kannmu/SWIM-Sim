import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupyx.scipy.signal as cpx_signal
except Exception:
    cp = None
    cpx_ndimage = None
    cpx_signal = None


class StressProcessor:
    def __init__(self, fs, filter_order, f_low, f_high, spatial_sigma, use_gpu=False):
        self.fs = fs
        self.filter_order = filter_order
        self.f_low = f_low
        self.f_high = f_high
        self.spatial_sigma = spatial_sigma
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

        N_receptors = receptor_coords.shape[0]
        x_vec_mm = (x_vec * 1000.0).astype(np.float64, copy=False)
        y_vec_mm = (y_vec * 1000.0).astype(np.float64, copy=False)

        t_indices = np.tile(np.arange(T, dtype=np.float64), N_receptors)
        rx_coords = np.repeat(receptor_coords[:, 0], T).astype(np.float64, copy=False)
        ry_coords = np.repeat(receptor_coords[:, 1], T).astype(np.float64, copy=False)
        query_points = np.column_stack((t_indices, rx_coords, ry_coords))

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
            "query_points": query_points,
            "x_vec_mm": x_vec_mm,
            "y_vec_mm": y_vec_mm,
            "N_receptors": N_receptors,
            "x0_idx": x0_idx.astype(np.int64, copy=False),
            "x1_idx": x1_idx.astype(np.int64, copy=False),
            "y0_idx": y0_idx.astype(np.int64, copy=False),
            "y1_idx": y1_idx.astype(np.int64, copy=False),
            "wx": wx.astype(np.float64, copy=False),
            "wy": wy.astype(np.float64, copy=False),
        }
        self._cache[cache_key] = cached
        return cached

    @staticmethod
    def _build_sos_zi_array(zi, first_sample):
        zi = np.asarray(zi, dtype=np.float64)
        first_sample = np.asarray(first_sample, dtype=np.float64)
        n_sections = zi.shape[0]
        n_receptors = first_sample.shape[0]
        zi_array = np.empty((n_sections, n_receptors, 2), dtype=np.float64)
        for sec in range(n_sections):
            zi_array[sec, :, :] = zi[sec][None, :] * first_sample[:, None]
        return zi_array

    @staticmethod
    def _build_sos_zi_array_gpu(zi, first_sample):
        n_sections = zi.shape[0]
        n_receptors = first_sample.shape[0]
        zi_array = cp.empty((n_sections, n_receptors, 2), dtype=cp.float64)
        for sec in range(n_sections):
            zi_array[sec, :, :] = zi[sec][None, :] * first_sample[:, None]
        return zi_array

    def _interpolate_cpu(self, smoothed_stress, T, cache):
        interp = RegularGridInterpolator(
            (np.arange(T, dtype=np.float64), cache["x_vec_mm"], cache["y_vec_mm"]),
            smoothed_stress,
            bounds_error=False,
            fill_value=0
        )
        all_signals = interp(cache["query_points"])
        return all_signals.reshape(cache["N_receptors"], T)

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

    def process(self, stress_tensor, x_vec, y_vec, receptor_coords, original_dt):
        T, _, _ = stress_tensor.shape

        dx = x_vec[1] - x_vec[0]
        dy = y_vec[1] - y_vec[0]
        sigma_pixels_x = self.spatial_sigma / dx
        sigma_pixels_y = self.spatial_sigma / dy

        cache = self._get_interpolation_cache(T, x_vec, y_vec, receptor_coords)
        input_fs = 1.0 / original_dt

        if self.use_gpu:
            stress_gpu = cp.asarray(stress_tensor, dtype=cp.float64)
            smoothed_stress = cpx_ndimage.gaussian_filter(
                stress_gpu,
                sigma=[0, sigma_pixels_x, sigma_pixels_y],
                mode='nearest'
            )
            raw_signals = self._interpolate_gpu(smoothed_stress, cache)

            if abs(input_fs - self.fs) > 1.0:
                num_samples_new = int(T * self.fs / input_fs)
                final_signals = cpx_signal.resample(raw_signals, num_samples_new, axis=1)
                t_vec_new = cp.arange(num_samples_new, dtype=cp.float64) / self.fs
            else:
                final_signals = raw_signals
                t_vec_new = cp.arange(T, dtype=cp.float64) / self.fs

            detrended_signals = final_signals - cp.mean(final_signals, axis=1, keepdims=True)
            zi = cpx_signal.sosfilt_zi(self.sos_gpu)
            zi_array = self._build_sos_zi_array_gpu(zi, detrended_signals[:, 0])
            filtered_signals, _ = cpx_signal.sosfilt(self.sos_gpu, detrended_signals, axis=1, zi=zi_array)
            return filtered_signals, t_vec_new, final_signals

        smoothed_stress = scipy.ndimage.gaussian_filter(
            stress_tensor,
            sigma=[0, sigma_pixels_x, sigma_pixels_y],
            mode='nearest'
        )
        raw_signals = self._interpolate_cpu(smoothed_stress, T, cache)

        if abs(input_fs - self.fs) > 1.0:
            num_samples_new = int(T * self.fs / input_fs)
            final_signals = scipy.signal.resample(raw_signals, num_samples_new, axis=1)
            t_vec_new = np.arange(num_samples_new, dtype=np.float64) / self.fs
        else:
            final_signals = raw_signals
            t_vec_new = np.arange(T, dtype=np.float64) / self.fs

        detrended_signals = final_signals - np.mean(final_signals, axis=1, keepdims=True)
        zi = scipy.signal.sosfilt_zi(self.sos)
        zi_array = self._build_sos_zi_array(zi, detrended_signals[:, 0])
        filtered_signals, _ = scipy.signal.sosfilt(self.sos, detrended_signals, axis=1, zi=zi_array)
        return filtered_signals, t_vec_new, final_signals
