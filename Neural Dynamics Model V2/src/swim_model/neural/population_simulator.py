from __future__ import annotations

import numpy as np

from ..mechanics.coherent_integration import CoherentIntegrator
from ..neural.lif import simulate_lif_population
from ..neural.pacinian_filter import PacinianBandpassFilter
from ..preprocessing.detrend_and_window import extract_analysis_window
from ..preprocessing.shear_equivalent import compute_dynamic_shear


class PopulationSimulator:
    def __init__(self, cfg, backend):
        self.cfg = cfg
        self.backend = backend

    @staticmethod
    def compute_vector_strength(spikes, t_vec, target_frequency_hz):
        spikes = np.asarray(spikes, dtype=bool)
        t_vec = np.asarray(t_vec, dtype=np.float64)
        spike_counts = spikes.sum(axis=1)
        phases = np.exp(1j * 2.0 * np.pi * float(target_frequency_hz) * t_vec)
        complex_sum = (spikes * phases[None, :]).sum(axis=1)
        vs = np.zeros(spikes.shape[0], dtype=np.float64)
        valid = spike_counts > 0
        vs[valid] = np.abs(complex_sum[valid]) / spike_counts[valid]
        return vs

    def run_condition(self, method_name, method_data, lattice):
        xp = self.backend.xp
        dtype = self.backend.get_default_dtype(self.cfg.model.float_dtype)
        tau_xy = self.backend.asarray(method_data["tau_xy"], dtype=dtype)
        tau_xz = self.backend.asarray(method_data["tau_xz"], dtype=dtype)
        tau_yz = self.backend.asarray(method_data["tau_yz"], dtype=dtype)
        t_vec = np.array(method_data["t"], dtype=np.float64, copy=False)
        dt = float(np.median(np.diff(t_vec)))

        tau_eq, tau_dyn = compute_dynamic_shear(tau_xy, tau_xz, tau_yz, xp)
        if self.cfg.model.steady_state_only:
            tau_dyn, t_vec = extract_analysis_window(
                tau_dyn,
                t_vec,
                self.cfg.model.carrier_frequency_hz,
                self.cfg.model.fft_window_cycles,
                xp,
            )
            t_vec = np.array(t_vec, dtype=np.float64, copy=False)
            dt = float(np.median(np.diff(t_vec)))

        integrator = CoherentIntegrator(
            roi_x=method_data["roi_x"],
            roi_y=method_data["roi_y"],
            receptor_coords=lattice["coords_m"],
            conduction_velocity_m_s=self.cfg.model.conduction_velocity_m_s,
            spatial_decay_lambda_m=self.cfg.model.spatial_decay_lambda_m,
            dt=dt,
            chunk_size=self.cfg.model.chunk_size,
            receptor_chunk_size=self.cfg.model.receptor_chunk_size,
        )
        m_drive = integrator.integrate(tau_dyn, xp)

        fs = 1.0 / dt
        bp = PacinianBandpassFilter(
            fs_hz=fs,
            low_hz=self.cfg.model.bandpass_low_hz,
            high_hz=self.cfg.model.bandpass_high_hz,
            order=self.cfg.model.bandpass_order,
            backend=self.backend,
        )
        u_drive = bp.apply(m_drive)
        u_drive_cpu = self.backend.to_numpy(u_drive)
        tau_eq_cpu = self.backend.to_numpy(tau_eq)
        spikes, rates = simulate_lif_population(
            u_drive_cpu,
            dt=dt,
            tau_m=self.cfg.model.membrane_tau_s,
            gain=self.cfg.model.gain,
            threshold=self.cfg.model.threshold,
            refractory_s=self.cfg.model.refractory_s,
        )
        vs = self.compute_vector_strength(spikes, t_vec, self.cfg.model.target_frequency_hz)
        weights = rates * vs
        population_map = integrator.collapse_to_map(weights)

        return {
            "method_name": method_name,
            "dt": dt,
            "time_s": t_vec,
            "tau_eq_mean": float(np.mean(tau_eq_cpu)),
            "m_drive": self.backend.to_numpy(m_drive),
            "u_drive": u_drive_cpu,
            "spikes": spikes,
            "rates": rates,
            "vector_strength": vs,
            "weights": weights,
            "population_map": population_map,
            "receptor_coords_m": np.array(lattice["coords_m"], dtype=np.float64, copy=False),
            "receptor_shape": lattice["shape"],
            "roi_x": np.array(method_data["roi_x"], dtype=np.float64, copy=False),
            "roi_y": np.array(method_data["roi_y"], dtype=np.float64, copy=False),
        }
