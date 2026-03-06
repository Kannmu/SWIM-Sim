import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import get_window

# =============================================================================
# Configuration
# =============================================================================

# Paths
MAT_FILE_PATH = r'k:\Work\SWIM\Sim\Outputs_Experiment1\experiment1_data.mat'
OUTPUT_DIR = r'k:\Work\SWIM\Sim\Outputs_Experiment1\Python_Plots'

# Visualization Settings
FONT_NAME = 'Arial'
SNS_CONTEXT = 'talk'  # 'paper', 'notebook', 'talk', 'poster'
SNS_STYLE = 'white'   # 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'

# Figure Sizes (width, height) in inches
FIG_SIZE_MAP = (8, 6)
FIG_SIZE_WAVEFORM = (8, 6)
FIG_SIZE_SPECTRUM = (8, 6)
FIG_SIZE_PROFILE = (8, 6)
FIG_SIZE_SNAPSHOTS = (8, 6)

# Snapshot Settings
SNAPSHOT_TIME_LABELS = ['t = T/4', 't = T/2', 't = 3T/4', 't = T']
SNAPSHOT_LABEL_POS = (0.5, 0.92)  # (x, y) in axes coordinates
SNAPSHOT_LABEL_BG_ALPHA = 0.6

# Font Sizes
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 18

# Colormaps
CMAP_ENERGY = 'inferno'
CMAP_STRESS = 'inferno'
CMAP_RATE = 'turbo'
CMAP_GRAD = 'turbo'
CMAP_XT = 'viridis'
CMAP_WAVEFRONT = 'RdBu_r'

# Plot Styling
LINE_WIDTH = 4
TITLE_PAD = 5
GRID_ALPHA = 0.7
ZOOM_DURATION = 0.015

# Layout Settings
LAYOUT_PAD = 0.0
WSPACE = 0.0
HSPACE = 0.0

# Waveform Plot Settings
WAVEFORM_RIGHT_MARGIN = 0.95  # Defines the right boundary (0 to 1) for the plot layout

# Comparison Plot Colors
# Specify the colormap name (Seaborn or Matplotlib). 
# Examples: 'tab10', 'deep', 'viridis', 'magma', 'coolwarm'
COMPARISON_COLORMAP = 'tab10'

# =============================================================================
# Setup
# =============================================================================

def get_comparison_colors(n, cmap_name=None):
    """Generates a list of n distinct colors from the specified colormap."""
    if cmap_name is None:
        cmap_name = COMPARISON_COLORMAP
    try:
        return sns.color_palette(cmap_name, n_colors=n).as_hex()
    except Exception as e:
        print(f"Warning: Could not generate colors from '{cmap_name}'. Using default fallback. Error: {e}")
        # Fallback to a safe default list
        defaults = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        return [defaults[i % len(defaults)] for i in range(n)]

def setup_style():
    """Configures the plotting style."""
    sns.set_theme(context=SNS_CONTEXT, style=SNS_STYLE)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_NAME, 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    plt.rcParams['legend.fontsize'] = LEGEND_SIZE
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# =============================================================================
# Data Loading
# =============================================================================

class Mat73Struct:
    """Helper class to mimic scipy.io.loadmat struct behavior."""
    def __repr__(self):
        return f"Mat73Struct({list(self.__dict__.keys())})"

def load_mat_h5py(path):
    """Loads v7.3 MAT-files using h5py with structure similar to scipy.io.loadmat."""
    f = h5py.File(path, 'r')
    
    def _convert(item):
        if isinstance(item, h5py.Group):
            d = Mat73Struct()
            for k in item.keys():
                setattr(d, k, _convert(item[k]))
            return d
        elif isinstance(item, h5py.Dataset):
            val = item[()]
            
            # Handle Object References (e.g., results array)
            if item.dtype == np.dtype('O'): 
                flat_val = val.flatten()
                converted_list = []
                for ref in flat_val:
                    if ref: # valid reference
                        obj = f[ref]
                        converted_list.append(_convert(obj))
                return np.array(converted_list)
            
            # Handle Strings (uint16 arrays)
            if val.dtype == np.uint16:
                # Check if it's likely a string (all printable or common chars)
                # Or just assume uint16 arrays are strings in this context
                try:
                    # Flatten and convert
                    chars = val.flatten()
                    s = "".join([chr(c) for c in chars if c != 0])
                    return s
                except:
                    pass

            # Handle Transpose for matrices (MATLAB compatibility)
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                val = val.T
                
            # Squeeze (MATLAB squeeze_me=True behavior)
            val = np.squeeze(val)
            
            # Handle 0-d array (scalar) conversion
            if val.ndim == 0 and isinstance(val, np.ndarray):
                val = val.item()
                
            return val
        return item
    
    mat = {}
    for k in f.keys():
        if k == '#refs#': continue
        mat[k] = _convert(f[k])
        
    return mat

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    print(f"Loading data from {path}...")
    try:
        # Try standard loadmat first
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        print("v7.3 MAT-file detected. Using h5py...")
        mat = load_mat_h5py(path)
    except Exception as e:
        # Fallback if specific error message varies
        if 'v7.3' in str(e) or 'HDF reader' in str(e):
             print("v7.3 MAT-file detected (via generic error). Using h5py...")
             mat = load_mat_h5py(path)
        else:
             raise e
    return mat

def extract_results(mat_data):
    """Extracts results into a list of dictionaries for easier handling."""
    results_raw = mat_data['results']
    
    # Robustly handle MATLAB arrays/cells/structs
    # np.atleast_1d converts 0-d arrays (scalars) to 1-d arrays
    results_array = np.atleast_1d(results_raw)
        
    results_list = []
    for res in results_array:
        # Convert struct to dict-like access
        item = {}
        # Handle potential string wrapping
        item['name'] = str(res.name) if not isinstance(res.name, np.ndarray) else str(res.name.item())
        
        item['tau_rms'] = res.tau_rms
        item['tau_peak'] = res.tau_peak
        item['tau_dt_peak'] = res.tau_dt_peak
        item['center_waveform'] = res.center_waveform
        item['xt_slice'] = res.xt_slice
        # Check for tau_xy_snapshots or fallback to tau_xz for backward compatibility
        if hasattr(res, 'tau_xy_snapshots'):
            item['tau_xy_snapshots'] = res.tau_xy_snapshots
        elif hasattr(res, 'tau_xz_snapshots'):
            item['tau_xy_snapshots'] = res.tau_xz_snapshots # Fallback map to xy key
            
        if hasattr(res, 'uz_snapshots'):
            item['uz_snapshots'] = res.uz_snapshots
            
        item['xt_slice_signed'] = res.xt_slice_signed
        item['grad_mag'] = res.grad_mag
        item['t_end_val'] = float(res.t_end_val)
        item['t_vec'] = res.t_vec
        results_list.append(item)
        
    return results_list

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_heatmap(data, x, y, title, cbar_label, cmap, output_path, vmin=None, vmax=None):
    """Generic heatmap plotter."""
    plt.figure(figsize=FIG_SIZE_MAP)
    
    # Let's align with MATLAB logic:
    # We will plot data.T (transpose) to match MATLAB's imagesc(..., data')
    
    extent = [x.min()*1000, x.max()*1000, y.min()*1000, y.max()*1000]
    
    img = plt.imshow(data.T, extent=extent, origin='lower', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    plt.colorbar(img, label=cbar_label)
    plt.xlabel('x [mm]', fontweight='bold')
    plt.ylabel('y [mm]', fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.grid(False)
    
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path)
    plt.close()

def plot_waveform(t, waveform, title, output_path, zoom_dur=ZOOM_DURATION):
    """Plots the center waveform."""
    plt.figure(figsize=FIG_SIZE_WAVEFORM)
    
    # Zoom logic
    t_end = t[-1]
    t_start = max(0, t_end - zoom_dur)
    mask = t >= t_start
    
    plt.plot(t[mask]*1000, waveform[mask], linewidth=LINE_WIDTH, color='b')
    
    plt.xlabel('Time [ms]', fontweight='bold')
    plt.ylabel('Stress [Pa]', fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.xlim(t_start*1000, t_end*1000)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    
    plt.tight_layout(pad=LAYOUT_PAD, rect=[0, 0, WAVEFORM_RIGHT_MARGIN, 1])
    plt.savefig(output_path)
    plt.close()

def plot_xt_diagram(data, t, x, title, output_path, vmax, zoom_dur=ZOOM_DURATION):
    """Plots Spatiotemporal X-T diagram."""
    plt.figure(figsize=FIG_SIZE_MAP)
    
    # Zoom logic
    t_end = t[-1]
    t_start = max(0, t_end - zoom_dur)
    mask = t >= t_start
    
    data_zoom = data[:, mask]
    t_zoom = t[mask]
    
    # MATLAB: imagesc(t_vec_zoom * 1000, kgrid.x_vec * 1000, xt_slice_zoom);
    # x-axis is time, y-axis is space (x)
    extent = [t_zoom.min()*1000, t_zoom.max()*1000, x.min()*1000, x.max()*1000]
    
    img = plt.imshow(data_zoom, extent=extent, origin='lower', cmap=CMAP_XT, aspect='auto', vmin=0, vmax=vmax)
    
    plt.colorbar(img, label='Shear Stress [Pa]')
    plt.xlabel('Time [ms]', fontweight='bold')
    plt.ylabel('x [mm]', fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.grid(False)
    
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path)
    plt.close()

def plot_wavefront_snapshots(snapshots, x, y, title, output_path, unit_label='[Pa]'):
    """Plots 4 snapshots of the wavefront."""
    # Ensure 4 snapshots
    if snapshots.ndim == 3 and snapshots.shape[2] >= 4:
        snapshots = snapshots[:, :, :4]
    else:
        print(f"Warning: {title} does not have 4 snapshots.")
        return

    # 1. Share axes to remove internal tick labels and align plots
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_SNAPSHOTS, sharex=True, sharey=True)
    axes = axes.flatten()
    
    extent = [x.min()*1000, x.max()*1000, y.min()*1000, y.max()*1000]
    
    # Calculate global vmin/vmax for this set
    # Demean first as requested
    snapshots_demeaned = []
    for i in range(4):
        snap = snapshots[:, :, i]
        snap = snap - np.mean(snap)
        snapshots_demeaned.append(snap)
    
    snapshots_demeaned = np.array(snapshots_demeaned)
    
    # Check for identical frames (debug)
    if np.allclose(snapshots_demeaned[0], snapshots_demeaned[1], atol=1e-6):
        print(f"  WARNING: Snapshots 0 and 1 for {title} are identical!")

    vmax = np.max(np.abs(snapshots_demeaned))
    if vmax < 1e-6: 
        print(f"  WARNING: Signal too weak (max < 1e-6). Vmax set to 1.")
        vmax = 1
        
    print(f"  Snapshots Vmax: {vmax:.2e} {unit_label}")
    vmin = -vmax
    
    for i, ax in enumerate(axes):
        # Transpose for plotting to match MATLAB imagesc
        img = ax.imshow(snapshots_demeaned[i].T, extent=extent, origin='lower', 
                        cmap=CMAP_WAVEFRONT, aspect='auto', vmin=vmin, vmax=vmax)
        
        # 2. Time markers inside the subplot
        # Place text at top-center inside the axes
        ax.text(SNAPSHOT_LABEL_POS[0], SNAPSHOT_LABEL_POS[1], SNAPSHOT_TIME_LABELS[i], transform=ax.transAxes, 
                ha='center', va='top', color='black', fontweight='bold', fontsize=LABEL_SIZE-2,
                bbox=dict(facecolor='white', alpha=SNAPSHOT_LABEL_BG_ALPHA, edgecolor='none', boxstyle='round,pad=0.2'))

        # 3. Bold axis labels (only on edges)
        if i >= 2: # Bottom row
            ax.set_xlabel('x [mm]', fontweight='bold')
        if i % 2 == 0: # Left column
            ax.set_ylabel('y [mm]', fontweight='bold')
            
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        
    # Add a single colorbar
    # 4. Control padding and layout
    # Adjust right to make space for colorbar, and respect WSPACE/HSPACE
    plt.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.93, wspace=WSPACE, hspace=HSPACE)
    
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(f'Amplitude {unit_label}', fontweight='bold', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    
    fig.suptitle(title, fontweight='bold', fontsize=TITLE_SIZE, y=0.98)
    
    # Use bbox_inches='tight' to remove extra white border
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_xt_diagram_signed(data, t, x, title, output_path, vmax=None, zoom_dur=ZOOM_DURATION):
    """Plots Spatiotemporal X-T diagram using signed data."""
    plt.figure(figsize=FIG_SIZE_MAP)
    
    # Zoom logic (last 15ms or whatever)
    t_end = t[-1]
    t_start = max(0, t_end - zoom_dur)
    mask = t >= t_start
    
    data_zoom = data[:, mask]
    t_zoom = t[mask]
    
    # Remove mean over time for each spatial point (row)
    # data is [Nx, Nt]
    # mean over axis 1 (time)
    means = np.mean(data_zoom, axis=1, keepdims=True)
    data_zoom = data_zoom - means
    
    # Determine limits
    if vmax is None:
        vmax = np.max(np.abs(data_zoom))
    if vmax == 0: vmax = 1
    vmin = -vmax
    
    # MATLAB: imagesc(t_vec_zoom * 1000, kgrid.x_vec * 1000, xt_slice_zoom);
    # x-axis is time, y-axis is space (x)
    extent = [t_zoom.min()*1000, t_zoom.max()*1000, x.min()*1000, x.max()*1000]
    
    img = plt.imshow(data_zoom, extent=extent, origin='lower', cmap=CMAP_WAVEFRONT, aspect='auto', vmin=vmin, vmax=vmax)
    
    plt.colorbar(img, label='Shear Stress (Signed) [Pa]')
    plt.xlabel('Time [ms]', fontweight='bold')
    plt.ylabel('x [mm]', fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.grid(False)
    
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path)
    plt.close()

def plot_spectrum_comparison(results_list, dt, output_path):
    plt.figure(figsize=FIG_SIZE_SPECTRUM)
    
    fs = 1.0 / dt
    
    colors = get_comparison_colors(len(results_list))
    for i, item in enumerate(results_list):
        color = colors[i]
        waveform = item['center_waveform']
        # Demean
        waveform = waveform - np.mean(waveform)
        # Window
        L = len(waveform)
        win = get_window('hann', L)
        waveform_win = waveform * win
        # FFT
        n_fft = 2 ** int(np.ceil(np.log2(L * 8)))
        Y = np.fft.fft(waveform_win, n_fft)
        P2 = np.abs(Y / L)
        P1 = P2[:n_fft // 2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        
        with np.errstate(divide='ignore'):
            P1_dB = 20 * np.log10(P1 + 1e-12)
            
        f = fs * np.arange(n_fft // 2 + 1) / n_fft
        
        plt.plot(f, P1_dB, linewidth=LINE_WIDTH, label=item['name'], color=color)
        
    plt.xlim(0, 1000)
    plt.xlabel('Frequency [Hz]', fontweight='bold')
    plt.ylabel('Magnitude [dB]', fontweight='bold')
    plt.title('Frequency Spectrum Comparison (Log Scale)', fontweight='bold', pad=TITLE_PAD)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_spatial_focusing(results_list, x_vec, dx, output_path):
    plt.figure(figsize=FIG_SIZE_PROFILE)
    
    ny = results_list[0]['tau_peak'].shape[1]
    y_idx = ny // 2
    
    colors = get_comparison_colors(len(results_list))
    FWHM_Y_OFFSET = 0.0
    for i, item in enumerate(results_list):
        color = colors[i]
        profile = item['tau_peak'][:, y_idx]
        profile_norm = profile / np.max(profile)
        
        # Calculate FWHM
        half_max = 0.5
        above_half = np.where(profile_norm >= half_max)[0]
        
        label = item['name']
        if len(above_half) > 0:
            fwhm_idx = above_half[-1] - above_half[0]
            fwhm_mm = fwhm_idx * dx * 1000
            plt.text(-38, 1 + FWHM_Y_OFFSET, f'FWHM={fwhm_mm:.0f} mm', fontsize=LEGEND_SIZE, color=color, ha='left',va='top')
            # label += f' (FWHM={fwhm_mm:.0f} mm)'
            
        plt.plot(x_vec * 1000, profile_norm, linewidth=LINE_WIDTH, label=label, color=color)
        FWHM_Y_OFFSET -= 0.08
        
    plt.xlabel('x [mm]', fontweight='bold')
    plt.ylabel('Normalized Peak Stress', fontweight='bold')
    plt.title('Spatial Focusing Profile (Normalized)', fontweight='bold', pad=TITLE_PAD)
    plt.legend(loc='best', frameon=True)
    # plt.legend(bbox_to_anchor=(0.5, 0.02), loc='lower center', frameon=True, ncol=1)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    plt.xlim(x_vec.min()*1000, x_vec.max()*1000)
    
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_cross_section_comparison(results_list, x_vec, output_path):
    """Plots the cross-section comparison at y=0 for RMS stress."""
    plt.figure(figsize=FIG_SIZE_PROFILE)
    
    # Assume all grids are the same size
    ny = results_list[0]['tau_rms'].shape[1]
    y_idx = ny // 2
    
    colors = get_comparison_colors(len(results_list))
    for i, item in enumerate(results_list):
        color = colors[i]
        # Extract the profile at the center y-index
        profile = item['tau_rms'][:, y_idx]
        
        label = item['name']
        plt.plot(x_vec * 1000, profile, linewidth=LINE_WIDTH, label=label, color=color)
        
    plt.xlabel('x [mm]', fontweight='bold')
    plt.ylabel('Shear Stress RMS [Pa]', fontweight='bold')
    plt.title('Cross-section Comparison (y=0)', fontweight='bold', pad=TITLE_PAD)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    plt.xlim(x_vec.min()*1000, x_vec.max()*1000)
    
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# =============================================================================
# Main Execution
# =============================================================================

def main():
    setup_style()
    ensure_dir(OUTPUT_DIR)
    
    try:
        mat_data = load_data(MAT_FILE_PATH)
    except FileNotFoundError:
        print("Wait! Please run the MATLAB simulation first to generate the data file.")
        return

    # Extract global variables
    x_vec = mat_data['x_vec']
    y_vec = mat_data['y_vec']
    dt = mat_data['dt']
    grid_cfg = mat_data['grid_cfg']
    dx = grid_cfg.dx
    
    results = extract_results(mat_data)
    
    # Calculate Global Maxima for consistent colorbars
    global_max_rms = max([np.max(r['tau_rms']) for r in results])
    global_max_peak = max([np.max(r['tau_peak']) for r in results])
    global_max_rate = max([np.max(r['tau_dt_peak']) for r in results])
    global_max_grad = max([np.max(r['grad_mag']) for r in results])
    
    # Calculate global max for Signed XT Diagram (after zooming and demeaning)
    global_max_xt_signed = 0
    for res in results:

        t_vec = res['t_vec']
        xt_data = res['xt_slice_signed']
        
        # Replicate zoom and demean logic
        t_end = t_vec[-1]
        t_start = max(0, t_end - ZOOM_DURATION)
        mask = t_vec >= t_start
        
        data_zoom = xt_data[:, mask]
        means = np.mean(data_zoom, axis=1, keepdims=True)
        data_zoom = data_zoom - means
        
        current_max = np.max(np.abs(data_zoom))
        if current_max > global_max_xt_signed:
            global_max_xt_signed = current_max
    
    print("Generating plots...")
    
    for res in results:

        name = res['name']
        print(f"Processing {name}...")
        
        # 1. RMS Shear Stress
        plot_heatmap(res['tau_rms'], x_vec, y_vec, 
                     f'{name} | RMS Shear Stress', 'RMS [Pa]', CMAP_ENERGY,
                     os.path.join(OUTPUT_DIR, f'{name}_1_energy_rms.png'),
                     vmax=global_max_rms)
        
        # 2. Peak Stress
        plot_heatmap(res['tau_peak'], x_vec, y_vec,
                     f'{name} | Peak Shear Stress', 'Peak Stress [Pa]', CMAP_STRESS,
                     os.path.join(OUTPUT_DIR, f'{name}_2_peak_stress.png'),
                     vmax=global_max_peak)
                     
        # 3. Rate of Change
        plot_heatmap(res['tau_dt_peak'], x_vec, y_vec,
                     f'{name} | Max Rate of Change', '|dτ/dt| [Pa/s]', CMAP_RATE,
                     os.path.join(OUTPUT_DIR, f'{name}_3_rate_of_change.png'),
                     vmax=global_max_rate)
                     
        # 4. Gradient
        plot_heatmap(res['grad_mag'], x_vec, y_vec,
                     f'{name} | Stress Gradient Snapshot', '|∇τ| [Pa/m]', CMAP_GRAD,
                     os.path.join(OUTPUT_DIR, f'{name}_4_gradient_snapshot.png'),
                     vmax=global_max_grad)
                     
        # 5. Waveform
        plot_waveform(res['t_vec'], res['center_waveform'],
                      f'{name} | Center Waveform',
                      os.path.join(OUTPUT_DIR, f'{name}_5_waveform.png'))
                      
        # 6. XT Diagram (Signed & Demeaned)
        plot_xt_diagram_signed(res['xt_slice_signed'], res['t_vec'], x_vec,
                        f'{name} | Signed XT Diagram (tau_xz)',
                        os.path.join(OUTPUT_DIR, f'comparison_XT_diagram_{name}_tau_xz.png'),
                        vmax=global_max_xt_signed)

        # 7. Wavefront Snapshots (Shear Stress tau_xy)
        if 'tau_xy_snapshots' in res:
            plot_wavefront_snapshots(res['tau_xy_snapshots'], x_vec, y_vec,
                                     f'{name} | Wavefront Snapshots (tau_xy)',
                                     os.path.join(OUTPUT_DIR, f'{name}_snapshots_tau_xy.png'),
                                     unit_label='[Pa]')
                                     
        # 8. Wavefront Snapshots (Particle Velocity uz)
        if 'uz_snapshots' in res:
            plot_wavefront_snapshots(res['uz_snapshots'], x_vec, y_vec,
                                     f'{name} | Wavefront Snapshots (uz)',
                                     os.path.join(OUTPUT_DIR, f'{name}_snapshots_uz.png'),
                                     unit_label='[m/s]')

    print("Generating comparison plots...")
    
    # Spectrum
    plot_spectrum_comparison(results, dt, os.path.join(OUTPUT_DIR, 'comparison_spectrum.png'))
    
    # Spatial Focusing
    plot_spatial_focusing(results, x_vec, dx, os.path.join(OUTPUT_DIR, 'comparison_spatial_focusing.png'))

    # Cross-section RMS
    plot_cross_section_comparison(results, x_vec, os.path.join(OUTPUT_DIR, 'comparison_cross_section_rms.png'))
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
