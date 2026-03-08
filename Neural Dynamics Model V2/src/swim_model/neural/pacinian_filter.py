from __future__ import annotations

import numpy as np
import scipy.signal


class PacinianBandpassFilter:
    def __init__(self, fs_hz: float, low_hz: float, high_hz: float, order: int, backend):
        self.fs_hz = float(fs_hz)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.order = int(order)
        self.backend = backend
        sos = scipy.signal.butter(
            N=self.order,
            Wn=[self.low_hz, self.high_hz],
            btype="bandpass",
            fs=self.fs_hz,
            output="sos",
        )
        self.sos = backend.asarray(sos, dtype=backend.xp.float64)

    def apply(self, signal):
        xp = self.backend.xp
        signal = xp.asarray(signal, dtype=xp.float64)
        filtered = self.backend.sosfilt(self.sos, signal, axis=-1)
        return xp.asarray(filtered, dtype=xp.float32)
