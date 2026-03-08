from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swim_model.backend import ArrayBackend
from swim_model.neural.pacinian_filter import PacinianBandpassFilter


backend = ArrayBackend(use_gpu=False)
fs = 2000.0
t = np.arange(0, 0.1, 1 / fs)
sig = np.sin(2 * np.pi * 200.0 * t)[None, :] + 0.1 * np.sin(2 * np.pi * 20.0 * t)[None, :]
flt = PacinianBandpassFilter(fs_hz=fs, low_hz=60.0, high_hz=450.0, order=2, backend=backend)
out = flt.apply(sig)
assert out.shape == sig.shape
assert float(np.max(np.abs(out))) > 0.1
