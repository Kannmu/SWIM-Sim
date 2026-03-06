import numpy as np
import scipy.signal
try:
    import cupy as cp
    import cupyx.scipy.signal as cpx_signal
except Exception:
    cp = None
    cpx_signal = None


class PopulationDecoder:
    def __init__(self, roi_area_mm2, density_sigma_mm, density_grid_mm, use_gpu=False):
        self.roi_area = roi_area_mm2
        self.density_sigma = density_sigma_mm
        self.density_grid = density_grid_mm
        self.use_gpu = bool(use_gpu and cp is not None)

        roi_size = np.sqrt(self.roi_area)
        half_roi = roi_size / 2.0
        self._grid_vec = np.arange(
            -half_roi,
            half_roi + self.density_grid / 100.0,
            self.density_grid,
            dtype=np.float64
        )
        gx, gy = np.meshgrid(self._grid_vec, self._grid_vec, indexing='xy')
        self._grid_points = np.column_stack((gx.ravel(), gy.ravel()))
        self._grid_points_gpu = cp.asarray(self._grid_points) if self.use_gpu else None
        self._cached_receptor_key = None
        self._gaussian_kernel = None

    def _ensure_kernel(self, receptor_coords):
        receptor_coords = np.asarray(receptor_coords, dtype=np.float64)
        key = (receptor_coords.shape, np.ascontiguousarray(receptor_coords).tobytes())
        if self._cached_receptor_key == key and self._gaussian_kernel is not None:
            return

        if self.use_gpu:
            receptor_gpu = cp.asarray(receptor_coords)
            diff = self._grid_points_gpu[:, None, :] - receptor_gpu[None, :, :]
            dist_sq = cp.sum(diff * diff, axis=2)
            two_sigma_sq = 2.0 * self.density_sigma * self.density_sigma
            self._gaussian_kernel = cp.exp(-dist_sq / two_sigma_sq)
        else:
            diff = self._grid_points[:, None, :] - receptor_coords[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            two_sigma_sq = 2.0 * self.density_sigma * self.density_sigma
            self._gaussian_kernel = np.exp(-dist_sq / two_sigma_sq)
        self._cached_receptor_key = key

    def compute_intensity_score(self, spike_trains):
        """
        Total spike count in the window.
        Args:
            spike_trains: [N, T_window] boolean array (sliced to window)
        """
        return np.sum(spike_trains)

    def compute_spatial_clarity(self, spike_trains, receptor_coords, roi_size):
        """
        Spatial Clarity Score = -log(A_0.5 / A_ROI)
        """
        # 1. Count spikes per receptor in window
        spike_counts = np.sum(spike_trains, axis=1)
        return self.compute_spatial_clarity_from_counts(spike_counts, receptor_coords)

    def compute_spatial_clarity_from_counts(self, spike_counts, receptor_coords):
        if self.use_gpu:
            spike_counts = cp.asarray(spike_counts, dtype=cp.float64)
        else:
            spike_counts = np.asarray(spike_counts, dtype=np.float64)
        self._ensure_kernel(receptor_coords)
        density_flat = self._gaussian_kernel @ spike_counts
        n = self._grid_vec.size
        density_map = density_flat.reshape(n, n)

        rho_max = cp.max(density_map) if self.use_gpu else np.max(density_map)
        if rho_max == 0:
            return 0.0

        mask = density_map >= (0.5 * rho_max)
        area_pixels = cp.sum(mask) if self.use_gpu else np.sum(mask)
        pixel_area = self.density_grid ** 2
        area_mm2 = area_pixels * pixel_area

        if area_mm2 == 0:
            return 0.0

        if self.use_gpu:
            score = -cp.log(area_mm2 / self.roi_area)
            return float(score.item())
        score = -np.log(area_mm2 / self.roi_area)
        return float(score)

    def compute_ffi(self, raw_signals, fs, f_signal_band, f_noise_band, epsilon=1e-9):
        """
        Frequency Fidelity Index.
        Based on PSD of raw inputs.
        """
        if self.use_gpu:
            raw_signals = cp.asarray(raw_signals, dtype=cp.float64)
            f, Pxx = cpx_signal.welch(raw_signals, fs, window='hann', nperseg=int(fs * 0.05), axis=1)
        else:
            raw_signals = np.asarray(raw_signals, dtype=np.float64)
            f, Pxx = scipy.signal.welch(raw_signals, fs, window='hann', nperseg=int(fs * 0.05), axis=1)

        idx_sig = (f >= f_signal_band[0]) & (f <= f_signal_band[1])
        power_sig = cp.sum(Pxx[:, idx_sig], axis=1) if self.use_gpu else np.sum(Pxx[:, idx_sig], axis=1)

        idx_noise = (f >= f_noise_band[0]) & (f <= f_noise_band[1])
        power_noise = cp.sum(Pxx[:, idx_noise], axis=1) if self.use_gpu else np.sum(Pxx[:, idx_noise], axis=1)

        numerator = cp.sum(power_sig) if self.use_gpu else np.sum(power_sig)
        denominator = (cp.sum(power_noise) if self.use_gpu else np.sum(power_noise)) + epsilon

        if denominator == 0:
            return 0.0

        if self.use_gpu:
            return float((numerator / denominator).item())
        return float(numerator / denominator)
