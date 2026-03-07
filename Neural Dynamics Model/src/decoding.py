import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


class PopulationDecoder:
    def __init__(
        self,
        roi_area_mm2,
        density_sigma_mm,
        density_grid_mm,
        fidelity_freqs_hz=None,
        use_gpu=False,
        temporal_smoothing_ms=2.0,
    ):
        self.roi_area = roi_area_mm2
        self.density_sigma = density_sigma_mm
        self.density_grid = density_grid_mm
        self.use_gpu = bool(use_gpu and cp is not None)
        self.fidelity_freqs_hz = tuple(fidelity_freqs_hz or (200.0, 400.0, 600.0, 800.0))
        self.temporal_smoothing_ms = float(temporal_smoothing_ms)

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

    def _xp(self):
        return cp if self.use_gpu else np

    def compute_phase_locked_weight(self, signal_train, t_vec, f0=200.0):
        if self.use_gpu:
            signal_arr = cp.asarray(signal_train, dtype=cp.float64)
            t = cp.asarray(t_vec, dtype=cp.float64)
            phases = cp.exp(-1j * 2.0 * cp.pi * f0 * t)
            return cp.sum(signal_arr * phases, axis=1)

        signal_arr = np.asarray(signal_train, dtype=np.float64)
        t = np.asarray(t_vec, dtype=np.float64)
        phases = np.exp(-1j * 2.0 * np.pi * f0 * t)
        return np.sum(signal_arr * phases, axis=1)

    def build_vector_phase_field(self, spikes_x, spikes_y, t_vec, fidelity_weights, f0=200.0):
        qx = self.compute_phase_locked_weight(spikes_x, t_vec, f0=f0)
        qy = self.compute_phase_locked_weight(spikes_y, t_vec, f0=f0)
        xp = self._xp()
        fidelity = xp.asarray(fidelity_weights, dtype=xp.float64)

        mean_fid = float(xp.mean(fidelity))
        max_fid = float(xp.max(fidelity))
        print(f"DEBUG: Fidelity Weights: mean={mean_fid:.4f}, max={max_fid:.4f}")

        return fidelity * qx, fidelity * qy

    def build_phase_locked_density_map(self, qx_complex, qy_complex, receptor_coords):
        if self.use_gpu:
            qx_complex = cp.asarray(qx_complex, dtype=cp.complex128)
            qy_complex = cp.asarray(qy_complex, dtype=cp.complex128)
        else:
            qx_complex = np.asarray(qx_complex, dtype=np.complex128)
            qy_complex = np.asarray(qy_complex, dtype=np.complex128)

        self._ensure_kernel(receptor_coords)
        psi_x_flat = self._gaussian_kernel @ qx_complex
        psi_y_flat = self._gaussian_kernel @ qy_complex
        n = self._grid_vec.size
        psi_x = psi_x_flat.reshape(n, n)
        psi_y = psi_y_flat.reshape(n, n)
        rho_map = (cp.sqrt(cp.abs(psi_x) ** 2 + cp.abs(psi_y) ** 2) if self.use_gpu
                   else np.sqrt(np.abs(psi_x) ** 2 + np.abs(psi_y) ** 2))
        return psi_x, psi_y, rho_map

    def compute_directional_concentration(self, psi_x, psi_y):
        xp = self._xp()
        psi_x_centered = psi_x - xp.mean(psi_x)
        psi_y_centered = psi_y - xp.mean(psi_y)
        fft_x = xp.fft.fft2(psi_x_centered)
        fft_y = xp.fft.fft2(psi_y_centered)
        energy = xp.abs(fft_x) ** 2 + xp.abs(fft_y) ** 2
        energy_no_dc = energy.copy()
        energy_no_dc[0, 0] = 0.0
        total_energy = xp.sum(energy_no_dc)
        total_energy_scalar = self._to_float(total_energy)
        if total_energy_scalar <= 0.0:
            return 0.0, energy_no_dc
        peak_energy = xp.max(energy_no_dc)
        ndwci = self._to_float(peak_energy / (total_energy + 1e-12))
        return ndwci, energy_no_dc

    def compute_core_map_metrics(self, qx_complex, qy_complex, receptor_coords, fidelity_weights=None):
        xp = self._xp()
        psi_x, psi_y, rho_map = self.build_phase_locked_density_map(qx_complex, qy_complex, receptor_coords)
        rho_max = xp.max(rho_map)
        rho_max_scalar = self._to_float(rho_max)
        rho_mean_scalar = self._to_float(xp.mean(rho_map))
        print(f"DEBUG: Rho Map: max={rho_max_scalar:.4f}, mean={rho_mean_scalar:.4f}")

        mean_ffi = 0.0
        if fidelity_weights is not None:
            mean_ffi = self._to_float(xp.mean(xp.asarray(fidelity_weights, dtype=xp.float64)))

        if rho_max_scalar <= 0.0:
            zero_energy = xp.zeros_like(rho_map)
            return {
                'intensity': 0.0,
                'clarity': 0.0,
                'rho_max': 0.0,
                'rho_map': rho_map,
                'psi_x': psi_x,
                'psi_y': psi_y,
                'energy_map': zero_energy,
                'ndwci': 0.0,
                'mean_ffi': mean_ffi,
            }

        ndwci, energy_map = self.compute_directional_concentration(psi_x, psi_y)
        intensity = rho_mean_scalar * mean_ffi * ndwci
        clarity = float(ndwci * mean_ffi)
        return {
            'intensity': intensity,
            'clarity': clarity,
            'rho_max': rho_max_scalar,
            'rho_map': rho_map,
            'psi_x': psi_x,
            'psi_y': psi_y,
            'energy_map': energy_map,
            'ndwci': ndwci,
            'mean_ffi': mean_ffi,
        }
