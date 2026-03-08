from __future__ import annotations


def extract_analysis_window(signal, t_vec, carrier_frequency_hz: float, n_cycles: int, xp):
    dt = float(t_vec[1] - t_vec[0])
    samples = max(1, int(round(n_cycles / carrier_frequency_hz / dt)))
    if samples >= signal.shape[-1]:
        return signal, t_vec
    return signal[..., -samples:], t_vec[-samples:]
