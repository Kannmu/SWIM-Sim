import h5py
import numpy as np
import os

# Configuration
MAT_FILE_PATH = r'k:\Work\SWIM\Sim\Outputs_Experiment1\experiment1_data.mat'
OUTPUT_REPORT_PATH = r'k:\Work\SWIM\Sim\Outputs_Experiment1\Simulation_Report.md'
PLOT_DIR_REL = 'Python_Plots'

def read_hdf5_string(obj_or_ref, f):
    """Reads a MATLAB string from an HDF5 object or reference."""
    try:
        # If it's a reference, dereference it
        if isinstance(obj_or_ref, h5py.h5r.Reference):
            obj = f[obj_or_ref]
        elif isinstance(obj_or_ref, (h5py.Dataset, h5py.Group)):
            obj = obj_or_ref
        elif isinstance(obj_or_ref, np.ndarray) and obj_or_ref.dtype == np.dtype('O'):
             # It might be a numpy array of references
             if obj_or_ref.size > 0:
                 return read_hdf5_string(obj_or_ref.flat[0], f)
             return ""
        else:
            # It might be the data itself (numpy array)
            obj = obj_or_ref

        # Get the data
        if isinstance(obj, h5py.Dataset):
            data = obj[:]
        else:
            data = obj

        # Handle MATLAB strings (usually uint16)
        if hasattr(data, 'dtype') and data.dtype == np.uint16:
            chars = data.flatten().view('uint16')
            s = ''.join([chr(c) for c in chars if c != 0])
            return s
        # Handle ASCII strings
        elif hasattr(data, 'dtype') and data.dtype.kind == 'S':
            return data.tobytes().decode('utf-8', errors='ignore').strip('\x00')
        # Handle simple char arrays
        elif isinstance(data, str):
            return data
            
        return str(data)
    except Exception as e:
        return f"Error reading string: {str(e)}"

def load_results(mat_path):
    """Loads the results from the .mat file."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")
    
    data = {}
    with h5py.File(mat_path, 'r') as f:
        # Check if 'results' exists
        if 'results' not in f:
            raise KeyError("'results' key not found in the .mat file.")
            
        results_refs = f['results'][:]
        
        methods_data = []
        
        # Flatten references to iterate easily
        refs_list = results_refs.flatten()
            
        for ref in refs_list:
            # Dereference the object (struct)
            try:
                method_group = f[ref]
            except:
                continue
                
            method_info = {}
            
            # Extract Name
            if 'name' in method_group:
                # 'name' dataset usually contains the string data directly or a reference
                # In many cases for structs in cells, it's a dataset of uint16
                try:
                    name_ds = method_group['name']
                    # Check if it's a reference inside a cell-like structure or direct data
                    # For a simple struct field string, it is often a direct dataset of uint16
                    if name_ds.dtype == np.dtype('O'):
                         # It's a reference
                         method_info['name'] = read_hdf5_string(name_ds[0][0], f)
                    else:
                         # It's direct data
                         method_info['name'] = read_hdf5_string(name_ds, f)
                except Exception as e:
                    method_info['name'] = f"Unknown ({e})"
            else:
                method_info['name'] = "Unnamed"
            
            # Helper to get max value from a field
            def get_max(field_name):
                if field_name in method_group:
                    try:
                        val = method_group[field_name][:]
                        return float(np.max(val))
                    except:
                        return 0.0
                return 0.0

            method_info['max_rms'] = get_max('tau_rms')
            method_info['max_peak'] = get_max('tau_peak')
            method_info['max_rate'] = get_max('tau_dt_peak')
            method_info['max_grad'] = get_max('grad_mag')
            
            methods_data.append(method_info)
            
        data['methods'] = methods_data
        
        # Global maximums might be stored separately or we can calculate them
        # The MATLAB script calculates Global_Max_RMS etc., but they might not be in the saved 'results' cell array
        # They might be separate variables in the .mat file if saved.
        # Checking the save command: save(save_path, 'results', 'x_vec', 'y_vec', 'dt', 'target', 'grid_cfg', '-v7.3');
        # It seems Global_Max_* are NOT saved. We should calculate them from the loaded data.
        
    return data

def generate_markdown(data, report_path):
    """Generates the markdown report."""
    methods = data['methods']
    
    # Sort methods by name for consistent ordering
    methods.sort(key=lambda x: x['name'])
    
    # Calculate global max for normalization (optional, but good for context)
    global_max_rms = max([m['max_rms'] for m in methods]) if methods else 1.0
    global_max_peak = max([m['max_peak'] for m in methods]) if methods else 1.0
    global_max_rate = max([m['max_rate'] for m in methods]) if methods else 1.0
    global_max_grad = max([m['max_grad'] for m in methods]) if methods else 1.0
    
    lines = []
    lines.append("# Simulation Report: Experiment 1 Modulations")
    lines.append(f"**Date:** {os.path.basename(report_path)}") # Just a placeholder
    lines.append("")
    
    lines.append("## Summary of Key Metrics")
    lines.append("Comparison of maximum values for different modulation methods.")
    lines.append("")
    
    # Table Header
    lines.append("| Method | Max RMS Stress (Pa) | Max Peak Stress (Pa) | Max Rate (Pa/s) | Max Gradient (Pa/m) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    
    for m in methods:
        name = m['name']
        rms = m['max_rms']
        peak = m['max_peak']
        rate = m['max_rate']
        grad = m['max_grad']
        
        # Formatting numbers
        lines.append(f"| **{name}** | {rms:.4e} | {peak:.4e} | {rate:.4e} | {grad:.4e} |")
        
    lines.append("")
    
    lines.append("## Detailed Visualizations")
    
    for m in methods:
        name = m['name']
        lines.append(f"### {name}")
        
        # Add links/images to the plots
        # Assuming standard naming convention from visualize_experiment1.py
        # DLM_2_1_energy_rms.png, etc.
        
        lines.append(f"**Key Metrics:**")
        lines.append(f"- Max RMS: {m['max_rms']:.4e}")
        lines.append(f"- Max Peak: {m['max_peak']:.4e}")
        
        lines.append("")
        lines.append("#### RMS & Peak Stress")
        lines.append(f"![RMS]({PLOT_DIR_REL}/{name}_1_energy_rms.png) ![Peak]({PLOT_DIR_REL}/{name}_2_peak_stress.png)")
        
        lines.append("#### Dynamics")
        lines.append(f"![Rate]({PLOT_DIR_REL}/{name}_3_rate_of_change.png) ![Gradient]({PLOT_DIR_REL}/{name}_4_gradient_snapshot.png)")
        
        lines.append("#### Waveforms & Snapshots")
        lines.append(f"![Waveform]({PLOT_DIR_REL}/{name}_5_waveform.png) ![Snapshots]({PLOT_DIR_REL}/{name}_snapshots.png)")
        
        lines.append("---")
        
    lines.append("## Comparative Analysis")
    lines.append("### Spatial Focusing")
    lines.append(f"![Spatial Focusing]({PLOT_DIR_REL}/comparison_spatial_focusing.png)")
    
    lines.append("### Cross-Section RMS")
    lines.append(f"![Cross Section]({PLOT_DIR_REL}/comparison_cross_section_rms.png)")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    try:
        data = load_results(MAT_FILE_PATH)
        generate_markdown(data, OUTPUT_REPORT_PATH)
    except Exception as e:
        print(f"Error: {e}")