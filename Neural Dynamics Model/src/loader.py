import h5py
import numpy as np
import scipy.io


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load(self):
        """
        Loads the .mat file. Handles v7.3 (HDF5) and older formats.
        Returns a dictionary containing the structured data.
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                return self._parse_h5(f)
        except OSError:
            try:
                mat = scipy.io.loadmat(self.file_path)
                return self._parse_mat(mat)
            except Exception as e:
                raise IOError(f"Failed to load {self.file_path}: {e}")

    @staticmethod
    def _transpose_roi_tensor(tensor):
        arr = np.asarray(tensor)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D ROI tensor, got shape {arr.shape}")
        return np.transpose(arr, (0, 2, 1)).astype(np.float64, copy=False)

    @staticmethod
    def _flatten_numeric(array_like):
        return np.asarray(array_like).astype(np.float64, copy=False).flatten()

    def _build_method_entry(self, name, tau_xz, tau_yz, tau_mag, roi_x, roi_y, t_vec):
        if tau_xz is None or tau_yz is None:
            raise ValueError(f"Method '{name}' is missing signed tangential stress components.")

        return {
            'stress_xz': tau_xz,
            'stress_yz': tau_yz,
            'stress_magnitude': tau_mag,
            't_vec': t_vec,
            'roi_x': roi_x,
            'roi_y': roi_y,
        }

    def _parse_h5(self, f):
        """
        Parses HDF5 structure from MATLAB v7.3 files.
        Extracts relevant fields for the neural model.
        """
        results_ref = f['results']

        parsed_data = {}
        methods_data = {}

        refs = results_ref[()].flatten()

        for i, ref in enumerate(refs):
            try:
                method_group = f[ref]
                name_data = method_group['name'][()]
                name = self._decode_h5_string(name_data)

                tau_mag = None
                if 'tau_roi_steady' in method_group:
                    tau_mag = self._transpose_roi_tensor(method_group['tau_roi_steady'][()])

                tau_xz = None
                if 'tau_roi_steady_xz' in method_group:
                    tau_xz = self._transpose_roi_tensor(method_group['tau_roi_steady_xz'][()])

                tau_yz = None
                if 'tau_roi_steady_yz' in method_group:
                    tau_yz = self._transpose_roi_tensor(method_group['tau_roi_steady_yz'][()])

                t_vec = self._flatten_numeric(method_group['t_vec_steady'][()])
                roi_x = self._flatten_numeric(method_group['roi_x_vec'][()])
                roi_y = self._flatten_numeric(method_group['roi_y_vec'][()])

                methods_data[name] = self._build_method_entry(
                    name=name,
                    tau_xz=tau_xz,
                    tau_yz=tau_yz,
                    tau_mag=tau_mag,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    t_vec=t_vec,
                )
                print(f"DEBUG: Loaded method '{name}': stress shape {tau_xz.shape if tau_xz is not None else 'None'}, t_vec {t_vec.shape}")
            except Exception as e:
                print(f"Warning: Failed to parse method index {i}: {e}")

        parsed_data['methods'] = methods_data

        if 'dt' in f:
            parsed_data['dt'] = float(f['dt'][()][0, 0])

        return parsed_data

    def _decode_h5_string(self, data):
        """Helper to decode MATLAB HDF5 strings"""
        try:
            return "".join([chr(c) for c in data.flatten() if c != 0])
        except Exception:
            return "Unknown"

    def _parse_mat(self, mat):
        """
        Parses older .mat format.
        """
        raise NotImplementedError("Legacy MATLAB struct parsing is not implemented for this pipeline.")
