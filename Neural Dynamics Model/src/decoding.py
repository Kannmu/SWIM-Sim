import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


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

    @staticmethod
    def _to_float(value):
        if cp is not None and isinstance(value, cp.ndarray):
            return float(value.item())
        return float(value)

    def compute_phase_locked_weight(self, spike_train, t_vec, f0=200.0):
        if self.use_gpu:
            spikes = cp.asarray(spike_train, dtype=cp.float64)
            t = cp.asarray(t_vec, dtype=cp.float64)
            phases = cp.exp(-1j * 2.0 * cp.pi * f0 * t)
            return cp.abs(cp.sum(spikes * phases, axis=1))

        spikes = np.asarray(spike_train, dtype=np.float64)
        t = np.asarray(t_vec, dtype=np.float64)
        phases = np.exp(-1j * 2.0 * np.pi * f0 * t)
        return np.abs(np.sum(spikes * phases, axis=1))

    def build_phase_locked_density_map(self, phase_weights, receptor_coords):
        if self.use_gpu:
            phase_weights = cp.asarray(phase_weights, dtype=cp.float64)
        else:
            phase_weights = np.asarray(phase_weights, dtype=np.float64)

        self._ensure_kernel(receptor_coords)
        density_flat = self._gaussian_kernel @ phase_weights
        n = self._grid_vec.size
        return density_flat.reshape(n, n)

    def compute_core_map_metrics(self, phase_weights, receptor_coords):
        density_map = self.build_phase_locked_density_map(phase_weights, receptor_coords)
        rho_max = cp.max(density_map) if self.use_gpu else np.max(density_map)
        rho_max_scalar = self._to_float(rho_max)

        if rho_max_scalar <= 0.0:
            return 0.0, 0.0, density_map

        mask = density_map >= (0.5 * rho_max)
        area_pixels = cp.sum(mask) if self.use_gpu else np.sum(mask)
        area_mm2 = self._to_float(area_pixels) * (self.density_grid ** 2)
        if area_mm2 <= 0.0:
            return rho_max_scalar, 0.0, density_map

        clarity = -np.log(area_mm2 / self.roi_area)
        return rho_max_scalar, float(clarity), density_map
