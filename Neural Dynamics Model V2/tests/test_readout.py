from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swim_model.readout.intensity_score import compute_intensity_score
from swim_model.readout.pairwise_prediction import pairwise_preferences


w = np.array([0.0, 1.0, 3.0], dtype=np.float64)
assert compute_intensity_score(w) > 0.0

_, pairwise = pairwise_preferences({"A": 1.0, "B": 0.0, "C": -1.0})
assert len(pairwise) == 3
