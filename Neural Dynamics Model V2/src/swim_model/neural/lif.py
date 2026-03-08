from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def run_lif_population(u_drive, dt, tau_m, gain, threshold, refractory_steps):
    n_units, n_time = u_drive.shape
    spikes = np.zeros((n_units, n_time), dtype=np.uint8)
    rates = np.zeros(n_units, dtype=np.float64)
    alpha = dt / tau_m

    for i in prange(n_units):
        v = 0.0
        ref = 0
        count = 0
        for t in range(n_time):
            if ref > 0:
                ref -= 1
                v = 0.0
                continue
            drive = gain * u_drive[i, t]
            if drive < 0.0:
                drive = 0.0
            v += alpha * (-v + drive)
            if v >= threshold:
                spikes[i, t] = 1
                count += 1
                v = 0.0
                ref = refractory_steps
        rates[i] = count / (n_time * dt)
    return spikes, rates


def simulate_lif_population(u_drive, dt, tau_m, gain, threshold, refractory_s):
    u_drive = np.asarray(u_drive, dtype=np.float32)
    refractory_steps = max(1, int(round(refractory_s / dt)))
    spikes, rates = run_lif_population(u_drive, float(dt), float(tau_m), float(gain), float(threshold), refractory_steps)
    return spikes.astype(bool), rates
