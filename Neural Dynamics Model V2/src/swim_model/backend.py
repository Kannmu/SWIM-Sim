from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupyx.scipy.signal as cpx_signal
except Exception:
    cp = None
    cpx_ndimage = None
    cpx_signal = None


@dataclass(frozen=True)
class ArrayBackend:
    use_gpu: bool = False

    @property
    def xp(self):
        return cp if self.use_gpu and cp is not None else np

    def asarray(self, array: Any, dtype=None):
        xp = self.xp
        return xp.asarray(array, dtype=dtype)

    def to_numpy(self, array: Any):
        if cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        if isinstance(array, dict):
            return {k: self.to_numpy(v) for k, v in array.items()}
        if isinstance(array, (list, tuple)):
            return type(array)(self.to_numpy(v) for v in array)
        return np.array(array, copy=False)

    def gaussian_filter(self, array, sigma, mode="nearest"):
        if self.use_gpu and cpx_ndimage is not None:
            return cpx_ndimage.gaussian_filter(array, sigma=sigma, mode=mode)
        import scipy.ndimage
        return scipy.ndimage.gaussian_filter(self.to_numpy(array), sigma=sigma, mode=mode)

    def sosfilt(self, sos, signal, axis=-1, zi=None):
        if self.use_gpu and cpx_signal is not None:
            return cpx_signal.sosfilt(sos, signal, axis=axis, zi=zi)
        import scipy.signal
        return scipy.signal.sosfilt(np.asarray(sos), self.to_numpy(signal), axis=axis, zi=zi)

    def sosfiltfilt(self, sos, signal, axis=-1):
        if self.use_gpu and cpx_signal is not None:
            return cpx_signal.sosfiltfilt(sos, signal, axis=axis)
        import scipy.signal
        return scipy.signal.sosfiltfilt(np.asarray(sos), self.to_numpy(signal), axis=axis)

    def get_default_dtype(self, name: str):
        if name == "float32":
            return self.xp.float32
        return self.xp.float64


def gpu_available() -> bool:
    return cp is not None
