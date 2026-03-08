from __future__ import annotations

import numpy as np


def _nearest_indices(source_coords, target_coords):
    diff = source_coords[:, None, :] - target_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return np.argmin(dist2, axis=0)


class CoherentIntegrator:
    def __init__(
        self,
        roi_x,
        roi_y,
        receptor_coords,
        conduction_velocity_m_s,
        spatial_decay_lambda_m,
        dt,
        chunk_size=128,
        receptor_chunk_size=64,
    ):
        self.roi_x = np.asarray(roi_x, dtype=np.float64)
        self.roi_y = np.asarray(roi_y, dtype=np.float64)
        self.receptor_coords = np.asarray(receptor_coords, dtype=np.float64)
        self.conduction_velocity_m_s = float(conduction_velocity_m_s)
        self.spatial_decay_lambda_m = float(spatial_decay_lambda_m)
        self.dt = float(dt)
        self.chunk_size = int(chunk_size)
        self.receptor_chunk_size = int(receptor_chunk_size)

        gx, gy = np.meshgrid(self.roi_x, self.roi_y, indexing="xy")
        self.source_coords = np.column_stack([gx.ravel(), gy.ravel()])
        self.source_to_receptor = _nearest_indices(self.source_coords, self.receptor_coords)
        self.receptor_count = self.receptor_coords.shape[0]
        self.source_count = self.source_coords.shape[0]
        self.distance_m = np.linalg.norm(
            self.receptor_coords[:, None, :] - self.source_coords[None, :, :],
            axis=-1,
        )
        self.weight = np.exp(-self.distance_m / max(self.spatial_decay_lambda_m, 1e-12)).astype(np.float32)
        self.delay_steps = np.rint(self.distance_m / max(self.conduction_velocity_m_s * self.dt, 1e-12)).astype(np.int32)

    def integrate(self, tau_dyn, xp):
        source_signal = xp.reshape(xp.asarray(tau_dyn, dtype=xp.float32), (-1, tau_dyn.shape[-1]))
        weight = xp.asarray(self.weight, dtype=xp.float32)
        delay_steps = xp.asarray(self.delay_steps, dtype=xp.int32)
        n_receptors = self.receptor_count
        n_time = source_signal.shape[1]
        integrated = xp.zeros((n_receptors, n_time), dtype=xp.float32)
        time_index = xp.arange(n_time, dtype=xp.int32)

        receptor_chunk = max(1, self.receptor_chunk_size)
        source_chunk = max(1, self.chunk_size)

        for receptor_start in range(0, n_receptors, receptor_chunk):
            receptor_end = min(receptor_start + receptor_chunk, n_receptors)
            integrated_chunk = integrated[receptor_start:receptor_end]
            time_buffer = xp.zeros((receptor_end - receptor_start, n_time), dtype=xp.float32)

            for source_start in range(0, self.source_count, source_chunk):
                source_end = min(source_start + source_chunk, self.source_count)
                src = source_signal[source_start:source_end]
                w = weight[receptor_start:receptor_end, source_start:source_end]
                d = delay_steps[receptor_start:receptor_end, source_start:source_end]
                block = xp.zeros((receptor_end - receptor_start, n_time), dtype=xp.float32)

                for local_source in range(source_end - source_start):
                    shifted_idx = time_index - d[:, local_source][:, None]
                    valid = shifted_idx >= 0
                    clipped = xp.clip(shifted_idx, 0, n_time - 1)
                    gathered = xp.take(src[local_source], clipped)
                    block += gathered * valid * w[:, local_source][:, None]

                time_buffer += block

            integrated_chunk[...] = time_buffer

        return integrated

    def collapse_to_map(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        out = np.zeros(self.source_count, dtype=np.float64)
        counts = np.zeros(self.source_count, dtype=np.float64)
        for receptor_idx, source_idx in enumerate(self.source_to_receptor):
            out[source_idx] += weights[receptor_idx]
            counts[source_idx] += 1.0
        counts[counts == 0.0] = 1.0
        return (out / counts).reshape(len(self.roi_y), len(self.roi_x))
