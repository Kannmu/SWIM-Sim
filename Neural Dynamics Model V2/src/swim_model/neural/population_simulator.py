from __future__ import annotations

import numpy as np

from ..mechanics.coherent_integration import CoherentIntegrator
from ..neural.lif import simulate_lif_population
from ..neural.pacinian_filter import PacinianBandpassFilter
from ..preprocessing.detrend_and_window import extract_analysis_window
from ..preprocessing.shear_equivalent import compute_dynamic_shear_components


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

    def _trim_filter_transient(self, signal, t_vec):
        signal = np.asarray(signal)
        t_vec = np.asarray(t_vec, dtype=np.float64)
        discard_cycles = max(1, int(self.cfg.model.fft_window_cycles // 2))
        samples_per_cycle = max(1, int(round((1.0 / self.cfg.model.carrier_frequency_hz) / max(float(np.median(np.diff(t_vec))), 1e-12))))
        discard_samples = min(signal.shape[-1] - 1, discard_cycles * samples_per_cycle) if signal.shape[-1] > 1 else 0
        if discard_samples <= 0:
            return signal, t_vec
        return signal[..., discard_samples:], t_vec[discard_samples:]

    def _simulate_component(self, tau_component_dyn, t_vec, integrator, bp, xp):
        m_drive = integrator.integrate(tau_component_dyn, xp)
        u_drive = bp.apply(m_drive)
        u_drive_cpu, t_trim = self._trim_filter_transient(self.backend.to_numpy(u_drive), t_vec)
        spikes, rates = simulate_lif_population(
            u_drive_cpu,
            dt=float(np.median(np.diff(t_trim))),
            tau_m=self.cfg.model.membrane_tau_s,
            gain=self.cfg.model.gain,
            threshold=self.cfg.model.threshold,
            refractory_s=self.cfg.model.refractory_s,
        )
        vs = self.compute_vector_strength(spikes, t_trim, self.cfg.model.target_frequency_hz)
        weights = rates * vs
        print("m_drive max abs:", np.max(np.abs(self.backend.to_numpy(m_drive))))
        print("u_drive max abs:", np.max(np.abs(u_drive_cpu)))
        print("u_drive positive max:", np.max(u_drive_cpu))
        return {
            "m_drive": self.backend.to_numpy(m_drive),
            "u_drive": u_drive_cpu,
            "time_s": t_trim,
            "spikes": spikes,
            "rates": rates,
            "vector_strength": vs,
            "weights": weights,
        }

    def run_condition(self, method_name, method_data, lattice):
        xp = self.backend.xp
        dtype = self.backend.get_default_dtype(self.cfg.model.float_dtype)
        tau_xy = self.backend.asarray(method_data["tau_xy"], dtype=dtype)
        tau_xz = self.backend.asarray(method_data["tau_xz"], dtype=dtype)
        tau_yz = self.backend.asarray(method_data["tau_yz"], dtype=dtype)
        t_vec = np.array(method_data["t"], dtype=np.float64, copy=False)

        tau_xy_dyn, tau_xz_dyn, tau_yz_dyn = compute_dynamic_shear_components(tau_xy, tau_xz, tau_yz, xp)
        if self.cfg.model.steady_state_only:
            tau_xy_dyn, t_vec = extract_analysis_window(
                tau_xy_dyn,
                t_vec,
                self.cfg.model.carrier_frequency_hz,
                self.cfg.model.fft_window_cycles,
                xp,
            )
            tau_xz_dyn, _ = extract_analysis_window(
                tau_xz_dyn,
                np.array(method_data["t"], dtype=np.float64, copy=False),
                self.cfg.model.carrier_frequency_hz,
                self.cfg.model.fft_window_cycles,
                xp,
            )
            tau_yz_dyn, _ = extract_analysis_window(
                tau_yz_dyn,
                np.array(method_data["t"], dtype=np.float64, copy=False),
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

        fs = 1.0 / dt
        bp = PacinianBandpassFilter(
            fs_hz=fs,
            low_hz=self.cfg.model.bandpass_low_hz,
            high_hz=self.cfg.model.bandpass_high_hz,
            order=self.cfg.model.bandpass_order,
            backend=self.backend,
        )

        components = {
            "xy": self._simulate_component(tau_xy_dyn, t_vec, integrator, bp, xp),
            "xz": self._simulate_component(tau_xz_dyn, t_vec, integrator, bp, xp),
            "yz": self._simulate_component(tau_yz_dyn, t_vec, integrator, bp, xp),
        }

        weights = components["xy"]["weights"] + components["xz"]["weights"] + components["yz"]["weights"]
        rates = components["xy"]["rates"] + components["xz"]["rates"] + components["yz"]["rates"]
        vector_strength = components["xy"]["vector_strength"] + components["xz"]["vector_strength"] + components["yz"]["vector_strength"]
        population_map = integrator.collapse_to_map(weights)

        return {
            "method_name": method_name,
            "dt": dt,
            "time_s": components["xy"]["time_s"],
            "dynamic_components": {
                "xy": self.backend.to_numpy(tau_xy_dyn),
                "xz": self.backend.to_numpy(tau_xz_dyn),
                "yz": self.backend.to_numpy(tau_yz_dyn),
            },
            "component_outputs": components,
            "m_drive": {
                key: value["m_drive"] for key, value in components.items()
            },
            "u_drive": {
                key: value["u_drive"] for key, value in components.items()
            },
            "spikes": {
                key: value["spikes"] for key, value in components.items()
            },
            "rates": rates,
            "vector_strength": vector_strength,
            "weights": weights,
            "population_map": population_map,
            "receptor_coords_m": np.array(lattice["coords_m"], dtype=np.float64, copy=False),
            "receptor_shape": lattice["shape"],
            "roi_x": np.array(method_data["roi_x"], dtype=np.float64, copy=False),
            "roi_y": np.array(method_data["roi_y"], dtype=np.float64, copy=False),
        }
