from __future__ import annotations

import numpy as np


def build_receptor_lattice(roi_x, roi_y, spacing_m: float, padding_m: float = 0.0) -> dict:
    x_min = float(np.min(roi_x)) + float(padding_m)
    x_max = float(np.max(roi_x)) - float(padding_m)
    y_min = float(np.min(roi_y)) + float(padding_m)
    y_max = float(np.max(roi_y)) - float(padding_m)

    x_coords = np.arange(x_min, x_max + spacing_m * 0.5, spacing_m, dtype=np.float64)
    y_coords = np.arange(y_min, y_max + spacing_m * 0.5, spacing_m, dtype=np.float64)
    gx, gy = np.meshgrid(x_coords, y_coords, indexing="xy")
    coords = np.column_stack([gx.ravel(), gy.ravel()])

    return {
        "coords_m": coords,
        "grid_x_m": x_coords,
        "grid_y_m": y_coords,
        "shape": (y_coords.size, x_coords.size),
    }
