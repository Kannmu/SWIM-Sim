from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class KWaveMatLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _decode_utf16(dataset) -> str:
        arr = np.asarray(dataset, dtype=np.uint16).ravel()
        return "".join(chr(v) for v in arr if v != 0)

    @staticmethod
    def _read_array(dataset):
        arr = np.asarray(dataset)
        return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))

    def load(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"MAT file not found: {self.path}")

        methods: dict[str, dict] = {}
        with h5py.File(self.path, "r") as f:
            results = f["results"]
            dt = float(np.asarray(f["dt"]).squeeze())
            for idx in range(results.shape[0]):
                group = f[results[idx, 0]]
                name = self._decode_utf16(group["name"])
                methods[name] = {
                    "tau_xy": self._read_array(group["tau_roi_steady_xy"]),
                    "tau_xz": self._read_array(group["tau_roi_steady_xz"]),
                    "tau_yz": self._read_array(group["tau_roi_steady_yz"]),
                    "tau_eq": self._read_array(group["tau_roi_steady"]),
                    "roi_x": np.asarray(group["roi_x_vec"]).reshape(-1),
                    "roi_y": np.asarray(group["roi_y_vec"]).reshape(-1),
                    "t": np.asarray(group["t_vec_steady"]).reshape(-1),
                }

        return {"dt": dt, "methods": methods}
