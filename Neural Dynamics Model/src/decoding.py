import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


class CoherentFieldDecoder:
    def __init__(
        self,
        roi_area_mm2,
        density_sigma_mm,
        density_grid_mm,
        fidelity_freqs_hz=None,
        use_gpu=False,
    ):
        self.roi_area = float(roi_area_mm2)
        self.density_sigma = float(density_sigma_mm)
        self.density_grid = float(density_grid_mm)
        self.use_gpu = bool(use_gpu and cp is not None)
        self.fidelity_freqs_hz = tuple(
            fidelity_freqs_hz or (200.0, 400.0, 600.0, 800.0)
        )

        roi_size = np.sqrt(self.roi_area)
        half_roi = roi_size / 2.0
        self._grid_vec = np.arange(
            -half_roi,
            half_roi + self.density_grid / 100.0,
            self.density_grid,
            dtype=np.float64,
        )
        gx, gy = np.meshgrid(self._grid_vec, self._grid_vec, indexing="xy")
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
    def to_numpy(array_like):
        if cp is not None and isinstance(array_like, cp.ndarray):
            return cp.asnumpy(array_like)
        return np.asarray(array_like)

    @staticmethod
    def _to_float(value):
        if cp is not None and isinstance(value, cp.ndarray):
            return float(value.item())
        return float(value)

    def _xp(self):
        return cp if self.use_gpu else np

    def compute_frequency_fidelity(
        self,
        drive_x,
        drive_y,
        t_vec,
        target_freq_hz=200.0,
    ):
        drive_x = self.to_numpy(drive_x).astype(np.float64, copy=False)
        drive_y = self.to_numpy(drive_y).astype(np.float64, copy=False)
        t = self.to_numpy(t_vec).astype(np.float64, copy=False).ravel()

        if drive_x.shape != drive_y.shape:
            raise ValueError(f"Drive shape mismatch: {drive_x.shape} vs {drive_y.shape}")
        if drive_x.shape[1] != t.size:
            raise ValueError(
                f"Time vector length mismatch: drive has {drive_x.shape[1]} samples, t_vec has {t.size}"
            )

        drive_x = drive_x - np.mean(drive_x, axis=1, keepdims=True)
        drive_y = drive_y - np.mean(drive_y, axis=1, keepdims=True)

        power_sum = None
        target_power = None

        for freq in self.fidelity_freqs_hz:
            phase = np.exp(-1j * 2.0 * np.pi * float(freq) * t)
            coeff_x = drive_x @ phase
            coeff_y = drive_y @ phase
            power = np.abs(coeff_x) ** 2 + np.abs(coeff_y) ** 2

            if abs(float(freq) - float(target_freq_hz)) < 1e-9:
                target_power = power

            power_sum = power if power_sum is None else (power_sum + power)

        if target_power is None:
            raise ValueError("target_freq_hz must be included in fidelity_freqs_hz")

        return target_power / (power_sum + 1e-12)

    def compute_phase_locked_coefficients(
        self,
        signal_x,
        signal_y,
        t_vec,
        target_freq_hz=200.0,
    ):
        xp = self._xp()
        x = xp.asarray(signal_x, dtype=xp.float64)
        y = xp.asarray(signal_y, dtype=xp.float64)
        t = xp.asarray(t_vec, dtype=xp.float64).ravel()

        if x.shape != y.shape:
            raise ValueError(f"Signal shape mismatch: {x.shape} vs {y.shape}")
        if x.ndim != 2:
            raise ValueError("signal_x and signal_y must be 2D arrays")
        if x.shape[1] != t.size:
            raise ValueError(
                f"Time vector length mismatch: signals have {x.shape[1]} samples, t_vec has {t.size}"
            )
        if t.size < 2:
            raise ValueError("t_vec must contain at least two samples")

        x = x - xp.mean(x, axis=1, keepdims=True)
        y = y - xp.mean(y, axis=1, keepdims=True)
        dt = self._to_float(t[1] - t[0])
        phase = xp.exp(-1j * 2.0 * xp.pi * float(target_freq_hz) * t)
        coeff_x = dt * xp.sum(x * phase[None, :], axis=1)
        coeff_y = dt * xp.sum(y * phase[None, :], axis=1)
        return coeff_x, coeff_y

    def build_coherent_field(self, coeff_x, coeff_y, receptor_coords):
        xp = self._xp()
        self._ensure_kernel(receptor_coords)

        coeff_x = xp.asarray(coeff_x, dtype=xp.complex128)
        coeff_y = xp.asarray(coeff_y, dtype=xp.complex128)

        psi_x_flat = self._gaussian_kernel @ coeff_x
        psi_y_flat = self._gaussian_kernel @ coeff_y

        n = self._grid_vec.size
        psi_x = psi_x_flat.reshape(n, n)
        psi_y = psi_y_flat.reshape(n, n)
        rho_map = xp.sqrt(xp.abs(psi_x) ** 2 + xp.abs(psi_y) ** 2)
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

    def compute_total_mass(self, rho_map):
        xp = self._xp()
        rho = xp.asarray(rho_map, dtype=xp.float64)
        delta_a = self.density_grid * self.density_grid
        return self._to_float(xp.sum(rho) * delta_a)

    def compute_effective_area(self, rho_map, eps=1e-12):
        xp = self._xp()
        rho = xp.asarray(rho_map, dtype=xp.float64)
        delta_a = self.density_grid * self.density_grid
        total_mass = xp.sum(rho) * delta_a
        squared_mass = total_mass * total_mass
        energy_mass = xp.sum(rho * rho) * delta_a
        effective_area = squared_mass / (energy_mass + float(eps))
        return self._to_float(effective_area)

    def compute_bridge_metrics(
        self,
        filtered_x,
        filtered_y,
        raw_drive_x,
        raw_drive_y,
        t_vec,
        receptor_coords,
        target_freq_hz=200.0,
    ):
        coeff_x, coeff_y = self.compute_phase_locked_coefficients(
            filtered_x,
            filtered_y,
            t_vec,
            target_freq_hz=target_freq_hz,
        )
        psi_x, psi_y, rho_map = self.build_coherent_field(
            coeff_x,
            coeff_y,
            receptor_coords,
        )
        ndwci, energy_map = self.compute_directional_concentration(psi_x, psi_y)
        mean_ffi = float(
            np.mean(
                self.to_numpy(
                    self.compute_frequency_fidelity(
                        raw_drive_x,
                        raw_drive_y,
                        t_vec,
                        target_freq_hz=target_freq_hz,
                    )
                )
            )
        )
        intensity_score = self.compute_total_mass(rho_map)
        effective_area_mm2 = self.compute_effective_area(rho_map)
        clarity_score = 1.0 / (effective_area_mm2 + 1e-12)
        rho_map_np = self.to_numpy(rho_map)
        energy_map_np = self.to_numpy(energy_map)

        return {
            "intensity_score": float(intensity_score),
            "clarity_score": float(clarity_score),
            "effective_area_mm2": float(effective_area_mm2),
            "mean_ffi": float(mean_ffi),
            "ndwci": float(ndwci),
            "rho_mean": float(np.mean(rho_map_np)),
            "rho_max": float(np.max(rho_map_np)),
            "rho_map": rho_map_np,
            "energy_map": energy_map_np,
        }
