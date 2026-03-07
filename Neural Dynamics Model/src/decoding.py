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
        self._pca_mean = None
        self._pca_components = None
        self._pca_variance = None

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

    @staticmethod
    def to_numpy(array_like):
        if cp is not None and isinstance(array_like, cp.ndarray):
            return cp.asnumpy(array_like)
        return np.asarray(array_like)

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

    def compute_mechanistic_metrics(self, spikes_x, spikes_y, drive_x, drive_y, t_vec, receptor_coords, f0=200.0):
        fidelity_weights = self.compute_frequency_fidelity(drive_x, drive_y, target_freq_hz=f0)
        qx_complex, qy_complex = self.build_vector_phase_field(
            spikes_x,
            spikes_y,
            t_vec,
            fidelity_weights,
            f0=f0,
        )
        psi_x, psi_y, rho_map = self.build_phase_locked_density_map(qx_complex, qy_complex, receptor_coords)
        ndwci, energy_map = self.compute_directional_concentration(psi_x, psi_y)
        mean_ffi = float(np.mean(self.to_numpy(fidelity_weights)))
        return {
            'mean_ffi': mean_ffi,
            'ndwci': float(ndwci),
            'dwci': float(ndwci),
            'rho_mean': float(np.mean(self.to_numpy(rho_map))),
            'rho_max': float(np.max(self.to_numpy(rho_map))),
            'energy_map': self.to_numpy(energy_map),
            'rho_map': self.to_numpy(rho_map),
        }

    def build_trial_response_vector(self, spikes_x, spikes_y, samples_per_bin):
        spikes_x = self.to_numpy(spikes_x).astype(np.float64, copy=False)
        spikes_y = self.to_numpy(spikes_y).astype(np.float64, copy=False)
        if spikes_x.shape != spikes_y.shape:
            raise ValueError(f"Spike shape mismatch: {spikes_x.shape} vs {spikes_y.shape}")
        n_receptors, n_time = spikes_x.shape
        if samples_per_bin <= 0:
            raise ValueError("samples_per_bin must be positive")
        n_bins = n_time // samples_per_bin
        if n_bins <= 0:
            raise ValueError("Not enough samples in steady-state window for binning.")
        trimmed = n_bins * samples_per_bin
        x_bins = spikes_x[:, :trimmed].reshape(n_receptors, n_bins, samples_per_bin).sum(axis=2)
        y_bins = spikes_y[:, :trimmed].reshape(n_receptors, n_bins, samples_per_bin).sum(axis=2)
        stacked = np.stack((x_bins, y_bins), axis=1)
        return stacked.reshape(-1)

    def fit_pca(self, response_matrix, variance_ratio=0.95):
        xp = self._xp()
        X = xp.asarray(response_matrix, dtype=xp.float64)
        if X.ndim != 2:
            raise ValueError("response_matrix must be 2D")
        if X.shape[0] < 2:
            raise ValueError("Need at least two response vectors to fit PCA")
        self._pca_mean = X.mean(axis=0)
        Xc = X - self._pca_mean
        _, svals, vt = xp.linalg.svd(Xc, full_matrices=False)
        explained = (svals ** 2) / max(X.shape[0] - 1, 1)
        total = explained.sum()
        total_scalar = self._to_float(total)
        if total_scalar <= 0.0:
            n_keep = 1
        else:
            ratio = xp.cumsum(explained) / total
            n_keep = int(np.searchsorted(self.to_numpy(ratio), variance_ratio) + 1)
        n_keep = max(1, min(n_keep, vt.shape[0]))
        self._pca_components = vt[:n_keep]
        self._pca_variance = explained[:n_keep]
        kept_ratio = self._to_float(explained[:n_keep].sum() / total) if total_scalar > 0.0 else 1.0
        return {
            'n_components': n_keep,
            'explained_variance_ratio': float(kept_ratio),
        }

    def transform_pca(self, response_matrix):
        if self._pca_mean is None or self._pca_components is None:
            raise RuntimeError("PCA must be fit before transform.")
        xp = cp if (cp is not None and (isinstance(self._pca_mean, cp.ndarray) or isinstance(self._pca_components, cp.ndarray))) else np
        X = xp.asarray(response_matrix, dtype=xp.float64)
        return (X - self._pca_mean) @ self._pca_components.T

    def compute_covariance(self, response_matrix, reg_eps=1e-6):
        xp = self._xp()
        X = xp.asarray(response_matrix, dtype=xp.float64)
        if X.ndim != 2:
            raise ValueError("response_matrix must be 2D")
        if X.shape[0] <= 1:
            cov = xp.eye(X.shape[1], dtype=xp.float64) * reg_eps
        else:
            cov = xp.cov(X, rowvar=False)
            if cov.ndim == 0:
                cov = xp.asarray([[self._to_float(cov)]], dtype=xp.float64)
        cov = xp.asarray(cov, dtype=xp.float64)
        cov += xp.eye(cov.shape[0], dtype=xp.float64) * reg_eps
        return cov

    @staticmethod
    def compute_detectability(mu_stim, mu_base, sigma_base):
        use_gpu = cp is not None and (
            isinstance(mu_stim, cp.ndarray) or isinstance(mu_base, cp.ndarray) or isinstance(sigma_base, cp.ndarray)
        )
        xp = cp if use_gpu else np
        delta = xp.asarray(mu_stim, dtype=xp.float64) - xp.asarray(mu_base, dtype=xp.float64)
        solved = xp.linalg.solve(xp.asarray(sigma_base, dtype=xp.float64), delta)
        return float((delta.T @ solved).item() if use_gpu else (delta.T @ solved))

    @staticmethod
    def compute_fisher_clarity(mu_pos_x, mu_neg_x, mu_pos_y, mu_neg_y, sigma_cond, delta_mm):
        use_gpu = cp is not None and (
            isinstance(mu_pos_x, cp.ndarray)
            or isinstance(mu_neg_x, cp.ndarray)
            or isinstance(mu_pos_y, cp.ndarray)
            or isinstance(mu_neg_y, cp.ndarray)
            or isinstance(sigma_cond, cp.ndarray)
        )
        xp = cp if use_gpu else np
        denom = 2.0 * float(delta_mm)
        g_x = (xp.asarray(mu_pos_x, dtype=xp.float64) - xp.asarray(mu_neg_x, dtype=xp.float64)) / denom
        g_y = (xp.asarray(mu_pos_y, dtype=xp.float64) - xp.asarray(mu_neg_y, dtype=xp.float64)) / denom
        sigma_arr = xp.asarray(sigma_cond, dtype=xp.float64)
        sigma_inv_gx = xp.linalg.solve(sigma_arr, g_x)
        sigma_inv_gy = xp.linalg.solve(sigma_arr, g_y)
        j11 = float((g_x.T @ sigma_inv_gx).item() if use_gpu else (g_x.T @ sigma_inv_gx))
        j12 = float((g_x.T @ sigma_inv_gy).item() if use_gpu else (g_x.T @ sigma_inv_gy))
        j21 = float((g_y.T @ sigma_inv_gx).item() if use_gpu else (g_y.T @ sigma_inv_gx))
        j22 = float((g_y.T @ sigma_inv_gy).item() if use_gpu else (g_y.T @ sigma_inv_gy))
        fisher = xp.asarray([[j11, j12], [j21, j22]], dtype=xp.float64)
        det_scalar = float(xp.linalg.det(fisher).item() if use_gpu else xp.linalg.det(fisher))
        det_val = max(det_scalar, 0.0)
        return float(np.sqrt(det_val)), PopulationDecoder.to_numpy(fisher)

    def compute_frequency_fidelity(self, drive_x, drive_y, target_freq_hz=200.0):
        drive_x = self.to_numpy(drive_x)
        drive_y = self.to_numpy(drive_y)
        n_time = drive_x.shape[1]
        if n_time <= 0:
            raise ValueError("Drive must contain at least one sample.")
        t_vec = np.arange(n_time, dtype=np.float64)
        power_sum = None
        target_power = None
        for freq in self.fidelity_freqs_hz:
            phase = np.exp(-1j * 2.0 * np.pi * float(freq) * t_vec / float(n_time))
            coeff_x = np.sum(drive_x * phase[None, :], axis=1)
            coeff_y = np.sum(drive_y * phase[None, :], axis=1)
            power = np.abs(coeff_x) ** 2 + np.abs(coeff_y) ** 2
            if int(round(freq)) == int(round(target_freq_hz)):
                target_power = power
            power_sum = power if power_sum is None else (power_sum + power)
        if target_power is None:
            raise ValueError("target_freq_hz must be included in fidelity_freqs_hz")
        return target_power / (power_sum + 1e-12)
