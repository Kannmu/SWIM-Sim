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
            # Try loading as v7.3 (HDF5)
            with h5py.File(self.file_path, 'r') as f:
                return self._parse_h5(f)
        except OSError:
            # Fallback to scipy.io.loadmat for older versions
            try:
                mat = scipy.io.loadmat(self.file_path)
                return self._parse_mat(mat)
            except Exception as e:
                raise IOError(f"Failed to load {self.file_path}: {e}")

    def _parse_h5(self, f):
        """
        Parses HDF5 structure from MATLAB v7.3 files.
        Extracts relevant fields for the neural model.
        """
        # MATLAB cell arrays in HDF5 are stored as object references
        # 'results' is a cell array of structs
        
        results_ref = f['results']
        # results_ref might be a dataset of references
        
        parsed_data = {}
        methods_data = {}
        
        # Determine number of methods
        refs = results_ref[()]
        # Flatten refs if needed
        refs = refs.flatten()
        
        for i, ref in enumerate(refs):
            try:
                method_group = f[ref]
                name_data = method_group['name'][()]
                name = self._decode_h5_string(name_data)
                
                # Check if tau_roi_steady exists
                if 'tau_roi_steady' not in method_group:
                    print(f"Warning: Method '{name}' (index {i}) missing 'tau_roi_steady'. Skipping.")
                    continue

                tau_roi = method_group['tau_roi_steady'][()]
                tau_roi = np.transpose(tau_roi, (0, 2, 1))

                tau_xz = None
                tau_xy = None
                tau_yz = None

                if 'tau_roi_steady_xz' in method_group:
                    tau_xz = method_group['tau_roi_steady_xz'][()]
                    tau_xz = np.transpose(tau_xz, (0, 2, 1))
                
                if 'tau_roi_steady_xy' in method_group:
                    tau_xy = method_group['tau_roi_steady_xy'][()]
                    tau_xy = np.transpose(tau_xy, (0, 2, 1))

                if 'tau_roi_steady_yz' in method_group:
                    tau_yz = method_group['tau_roi_steady_yz'][()]
                    tau_yz = np.transpose(tau_yz, (0, 2, 1))
                
                # Also extract time vector
                t_vec = method_group['t_vec_steady'][()].flatten()
                
                # Extract ROI vectors
                roi_x = method_group['roi_x_vec'][()].flatten()
                roi_y = method_group['roi_y_vec'][()].flatten()
                
                methods_data[name] = {
                    'stress_magnitude': tau_roi,
                    'stress_xz': tau_xz,
                    'stress_xy': tau_xy,
                    'stress_yz': tau_yz,
                    't_vec': t_vec,
                    'roi_x': roi_x,
                    'roi_y': roi_y
                }
                
            except Exception as e:
                print(f"Warning: Failed to parse method index {i}: {e}")
                
        parsed_data['methods'] = methods_data
        
        if 'dt' in f:
            parsed_data['dt'] = f['dt'][()][0,0]
        
        return parsed_data

    def _decode_h5_string(self, data):
        """Helper to decode MATLAB HDF5 strings"""
        try:
            # If it's a list of integers (uint16)
            return "".join([chr(c) for c in data.flatten() if c != 0])
        except:
            return "Unknown"

    def _parse_mat(self, mat):
        """
        Parses older .mat format.
        """
        pass
