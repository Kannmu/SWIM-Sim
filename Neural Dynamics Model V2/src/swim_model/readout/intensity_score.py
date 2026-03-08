from __future__ import annotations

import numpy as np


def compute_intensity_score(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.log1p(np.sum(weights)))
