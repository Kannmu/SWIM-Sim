from __future__ import annotations

import numpy as np
import scipy.ndimage


def compute_a90_area(population_map, roi_x, roi_y, mass_fraction=0.90, sigma=1.0):
    pop = np.asarray(population_map, dtype=np.float64)
    pop = scipy.ndimage.gaussian_filter(pop, sigma=sigma, mode="nearest")
    pop = np.maximum(pop, 0.0)
    total = float(pop.sum())
    if total <= 0.0:
        dx = abs(float(roi_x[1] - roi_x[0])) if len(roi_x) > 1 else 1.0
        dy = abs(float(roi_y[1] - roi_y[0])) if len(roi_y) > 1 else 1.0
        return dx * dy * pop.size
    flat = np.sort(pop.ravel())[::-1]
    csum = np.cumsum(flat)
    idx = int(np.searchsorted(csum, mass_fraction * total, side="left"))
    cell_count = idx + 1
    dx = abs(float(roi_x[1] - roi_x[0])) if len(roi_x) > 1 else 1.0
    dy = abs(float(roi_y[1] - roi_y[0])) if len(roi_y) > 1 else 1.0
    return cell_count * dx * dy


def compute_clarity_score(population_map, roi_x, roi_y, mass_fraction=0.90, sigma=1.0):
    area = compute_a90_area(population_map, roi_x, roi_y, mass_fraction=mass_fraction, sigma=sigma)
    return float(-np.log(area + 1e-12)), float(area)
