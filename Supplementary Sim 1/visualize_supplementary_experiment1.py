import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import get_window

MAT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'Outputs_Supplementary_Experiment1', 'supplementary_experiment1_data.mat')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'Outputs_Supplementary_Experiment1', 'Python_Plots')

FONT_NAME = 'Arial'
SNS_CONTEXT = 'talk'
SNS_STYLE = 'white'

FIG_SIZE_MAP = (8, 6)
FIG_SIZE_WAVEFORM = (8, 6)
FIG_SIZE_SPECTRUM = (8, 6)
FIG_SIZE_PROFILE = (8, 6)
FIG_SIZE_SNAPSHOTS = (8, 6)

SNAPSHOT_TIME_LABELS = ['t = T/4', 't = T/2', 't = 3T/4', 't = T']
SNAPSHOT_LABEL_POS = (0.5, 0.92)
SNAPSHOT_LABEL_BG_ALPHA = 0.6

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 18

CMAP_ENERGY = 'inferno'
CMAP_STRESS = 'inferno'
CMAP_RATE = 'turbo'
CMAP_GRAD = 'turbo'
CMAP_WAVEFRONT = 'RdBu_r'

LINE_WIDTH = 4
TITLE_PAD = 5
GRID_ALPHA = 0.7
ZOOM_DURATION = 0.015
LAYOUT_PAD = 0.0
WSPACE = 0.0
HSPACE = 0.0
WAVEFORM_RIGHT_MARGIN = 0.95
COMPARISON_COLORMAP = 'viridis'

METHOD_ORDER = ['ULM_L_M0p2', 'ULM_L_M0p5', 'ULM_L_M1p0', 'ULM_L_M1p2', 'ULM_L_M2p0']


def get_comparison_colors(n, cmap_name=None):
    if cmap_name is None:
        cmap_name = COMPARISON_COLORMAP
    try:
        return sns.color_palette(cmap_name, n_colors=n).as_hex()
    except Exception:
        defaults = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        return [defaults[i % len(defaults)] for i in range(n)]


_METHOD_COLOR_MAP = {}


def get_method_color(method_name):
    global _METHOD_COLOR_MAP
    if not _METHOD_COLOR_MAP:
        colors = get_comparison_colors(len(METHOD_ORDER))
        _METHOD_COLOR_MAP = dict(zip(METHOD_ORDER, colors))
    return _METHOD_COLOR_MAP.get(method_name, '#333333')


LABEL_MAP = {
    'ULM_L_M0p2': 'ULM_L (M=0.2)',
    'ULM_L_M0p5': 'ULM_L (M=0.5)',
    'ULM_L_M1p0': 'ULM_L (M=1.0)',
    'ULM_L_M1p2': 'ULM_L (M=1.2)',
    'ULM_L_M2p0': 'ULM_L (M=2.0)',
}


def setup_style():
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


class Mat73Struct:
    def __repr__(self):
        return f"Mat73Struct({list(self.__dict__.keys())})"



def load_mat_h5py(path):
    f = h5py.File(path, 'r')

    def _convert(item):
        if isinstance(item, h5py.Group):
            d = Mat73Struct()
            for k in item.keys():
                setattr(d, k, _convert(item[k]))
            return d
        if isinstance(item, h5py.Dataset):
            val = item[()]
            if item.dtype == np.dtype('O'):
                flat_val = val.flatten()
                converted_list = []
                for ref in flat_val:
                    if ref:
                        converted_list.append(_convert(f[ref]))
                return np.array(converted_list)
            if val.dtype == np.uint16:
                try:
                    chars = val.flatten()
                    return ''.join([chr(c) for c in chars if c != 0])
                except Exception:
                    pass
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                val = val.T
            val = np.squeeze(val)
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            return val
        return item

    mat = {}
    for k in f.keys():
        if k == '#refs#':
            continue
        mat[k] = _convert(f[k])
    return mat



def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Data file not found: {path}')
    print(f'Loading data from {path}...')
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return load_mat_h5py(path)
    except Exception as e:
        if 'v7.3' in str(e) or 'HDF reader' in str(e):
            return load_mat_h5py(path)
        raise



def extract_results(mat_data):
    results_raw = mat_data['results']
    results_array = np.atleast_1d(results_raw)
    results_list = []
    for res in results_array:
        item = {}
        item['name'] = str(res.name) if not isinstance(res.name, np.ndarray) else str(res.name.item())
        item['tau_rms'] = res.tau_rms
        item['tau_peak'] = res.tau_peak
        item['tau_dt_peak'] = res.tau_dt_peak
        item['center_waveform'] = res.center_waveform
        item['xt_slice'] = res.xt_slice
        item['tau_xy_snapshots'] = res.tau_xy_snapshots if hasattr(res, 'tau_xy_snapshots') else None
        item['uz_snapshots'] = res.uz_snapshots if hasattr(res, 'uz_snapshots') else None
        item['xt_slice_signed'] = res.xt_slice_signed
        item['grad_mag'] = res.grad_mag
        item['t_end_val'] = float(res.t_end_val)
        item['t_vec'] = res.t_vec
        item['tau_roi_steady'] = res.tau_roi_steady if hasattr(res, 'tau_roi_steady') else None
        item['tau_roi_steady_xy'] = res.tau_roi_steady_xy if hasattr(res, 'tau_roi_steady_xy') else None
        item['tau_roi_steady_xz'] = res.tau_roi_steady_xz if hasattr(res, 'tau_roi_steady_xz') else None
        item['tau_roi_steady_yz'] = res.tau_roi_steady_yz if hasattr(res, 'tau_roi_steady_yz') else None
        item['mach'] = float(res.mach) if hasattr(res, 'mach') else np.nan
        item['length'] = float(res.length) if hasattr(res, 'length') else np.nan
        item['scan_speed'] = float(res.scan_speed) if hasattr(res, 'scan_speed') else np.nan
        results_list.append(item)
    return results_list



def plot_heatmap(data, x, y, title, cbar_label, cmap, output_path, vmin=None, vmax=None):
    plt.figure(figsize=FIG_SIZE_MAP)
    extent = [x.min() * 1000, x.max() * 1000, y.min() * 1000, y.max() * 1000]
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
    plt.figure(figsize=FIG_SIZE_WAVEFORM)
    t_end = t[-1]
    t_start = max(0, t_end - zoom_dur)
    mask = t >= t_start
    plt.plot(t[mask] * 1000, waveform[mask], linewidth=LINE_WIDTH, color='b')
    plt.xlabel('Time [ms]', fontweight='bold')
    plt.ylabel('Stress [Pa]', fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.xlim(t_start * 1000, t_end * 1000)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    plt.tight_layout(pad=LAYOUT_PAD, rect=[0, 0, WAVEFORM_RIGHT_MARGIN, 1])
    plt.savefig(output_path)
    plt.close()



def plot_wavefront_snapshots(snapshots, x, y, title, output_path, unit_label='[Pa]', vmax=None):
    if snapshots is None or snapshots.ndim != 3 or snapshots.shape[2] < 4:
        return
    snapshots = snapshots[:, :, :4]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_SNAPSHOTS, sharex=True, sharey=True)
    axes = axes.flatten()
    extent = [x.min() * 1000, x.max() * 1000, y.min() * 1000, y.max() * 1000]

    snapshots_demeaned = []
    for i in range(4):
        snap = snapshots[:, :, i]
        snapshots_demeaned.append(snap - np.mean(snap))
    snapshots_demeaned = np.array(snapshots_demeaned)

    if vmax is None:
        vmax = np.max(np.abs(snapshots_demeaned))
        if vmax < 1e-6:
            vmax = 1
    vmin = -vmax

    for i, ax in enumerate(axes):
        img = ax.imshow(snapshots_demeaned[i].T, extent=extent, origin='lower', cmap=CMAP_WAVEFRONT, aspect='auto', vmin=vmin, vmax=vmax)
        ax.text(SNAPSHOT_LABEL_POS[0], SNAPSHOT_LABEL_POS[1], SNAPSHOT_TIME_LABELS[i], transform=ax.transAxes,
                ha='center', va='top', color='black', fontweight='bold', fontsize=LABEL_SIZE - 2,
                bbox=dict(facecolor='white', alpha=SNAPSHOT_LABEL_BG_ALPHA, edgecolor='none', boxstyle='round,pad=0.2'))
        if i >= 2:
            ax.set_xlabel('x [mm]', fontweight='bold')
        if i % 2 == 0:
            ax.set_ylabel('y [mm]', fontweight='bold')
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    plt.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.93, wspace=WSPACE, hspace=HSPACE)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(f'Amplitude {unit_label}', fontweight='bold', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    fig.suptitle(title, fontweight='bold', fontsize=TITLE_SIZE, y=0.98)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()



def plot_xt_diagram_signed(data, t, x, title, output_path, vmax=None, zoom_dur=ZOOM_DURATION):
    plt.figure(figsize=FIG_SIZE_MAP)
    t_end = t[-1]
    t_start = max(0, t_end - zoom_dur)
    mask = t >= t_start
    data_zoom = data[:, mask]
    t_zoom = t[mask]
    data_zoom = data_zoom - np.mean(data_zoom, axis=1, keepdims=True)
    if vmax is None:
        vmax = np.max(np.abs(data_zoom))
    if vmax == 0:
        vmax = 1
    extent = [t_zoom.min() * 1000, t_zoom.max() * 1000, x.min() * 1000, x.max() * 1000]
    img = plt.imshow(data_zoom, extent=extent, origin='lower', cmap=CMAP_WAVEFRONT, aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(img, label='Shear stress (signed) [Pa]')
    plt.xlabel('Time [ms]', fontweight='bold')
    plt.ylabel('x [mm]', fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.grid(False)
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path)
    plt.close()



def get_ordered_results(results_list, method_order=METHOD_ORDER):
    order_map = {name: idx for idx, name in enumerate(method_order)}
    return sorted(results_list, key=lambda item: order_map.get(item['name'], len(order_map)))



def ensure_time_last(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        return arr
    time_axis = int(np.argmax(arr.shape))
    if time_axis != 2:
        arr = np.moveaxis(arr, time_axis, 2)
    return arr



def compute_single_sided_spectrum(waveform, dt):
    waveform = np.asarray(waveform, dtype=float)
    waveform = waveform - np.mean(waveform)
    L = len(waveform)
    win = get_window('hann', L)
    waveform_win = waveform * win
    n_fft = 2 ** int(np.ceil(np.log2(L * 8)))
    Y = np.fft.fft(waveform_win, n_fft)
    P2 = np.abs(Y / L)
    P1 = P2[:n_fft // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    fs = 1.0 / dt
    f = fs * np.arange(n_fft // 2 + 1) / n_fft
    return f, P1



def compute_frequency_fidelity_index(res, dt):
    s = ensure_time_last(res['tau_roi_steady'])
    s = s - np.mean(s, axis=2, keepdims=True)
    nt = s.shape[2]
    freqs = np.fft.rfftfreq(nt, d=dt)
    S = np.fft.rfft(s, axis=2)
    P = np.abs(S) ** 2
    P_sum = np.sum(P, axis=(0, 1))

    def bin_power(fc):
        idx = int(np.argmin(np.abs(freqs - fc)))
        return float(P_sum[idx])

    p200 = bin_power(200)
    p400 = bin_power(400)
    p600 = bin_power(600)
    p800 = bin_power(800)
    total = p200 + p400 + p600 + p800
    return float(p200 / total) if total > 0 else 0.0



def compute_directional_wavefront_concentration_index(res, dt, f0=200):
    component_keys = ['tau_roi_steady_xy', 'tau_roi_steady_xz', 'tau_roi_steady_yz']
    e_total = None
    for key in component_keys:
        s = ensure_time_last(res[key])
        s = s - np.mean(s, axis=2, keepdims=True)
        nt = s.shape[2]
        freqs = np.fft.rfftfreq(nt, d=dt)
        idx = int(np.argmin(np.abs(freqs - f0)))
        S = np.fft.rfft(s, axis=2)
        A = S[:, :, idx]
        K = np.fft.fftshift(np.fft.fft2(A))
        E = np.abs(K) ** 2
        if e_total is None:
            e_total = E
        else:
            e_total += E
    total_energy = float(np.sum(e_total))
    if total_energy <= 0:
        return 0.0
    return float(np.max(e_total) / total_energy)



def plot_metric_bar(results_list, metric_values, title, ylabel, output_path):
    ordered_results = get_ordered_results(results_list)
    labels = [LABEL_MAP.get(item['name'], item['name']) for item in ordered_results]
    values = [metric_values.get(item['name'], np.nan) for item in ordered_results]
    colors = [get_method_color(item['name']) for item in ordered_results]
    plt.figure(figsize=FIG_SIZE_PROFILE)
    bars = plt.bar(labels, values, color=colors)
    finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    ymax = float(np.nanmax(finite_values)) if finite_values.size > 0 else 0.0
    text_offset = 0.01 if ymax <= 0 else max(0.001, ymax * 0.02)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            plt.text(bar.get_x() + bar.get_width() / 2, value + text_offset, f'{value:.4f}', ha='center', va='bottom', fontsize=TICK_SIZE - 2)
    plt.xlabel('Condition', fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.title(title, fontweight='bold', pad=TITLE_PAD)
    plt.ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
    plt.xticks(rotation=15)
    plt.grid(True, axis='y', linestyle='--', alpha=GRID_ALPHA)
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def plot_spectrum_comparison(results_list, dt, output_path):
    plt.figure(figsize=FIG_SIZE_SPECTRUM)
    ordered_results = get_ordered_results(results_list)
    for item in ordered_results:
        color = get_method_color(item['name'])
        freqs, spectrum = compute_single_sided_spectrum(item['center_waveform'], dt)
        with np.errstate(divide='ignore'):
            spectrum_db = 20 * np.log10(spectrum + 1e-12)
        plt.plot(freqs, spectrum_db, linewidth=LINE_WIDTH, label=LABEL_MAP.get(item['name'], item['name']), color=color)
    plt.xlim(0, 1000)
    plt.xlabel('Frequency [Hz]', fontweight='bold')
    plt.ylabel('Magnitude [dB]', fontweight='bold')
    plt.title('Frequency spectrum comparison (log scale)', fontweight='bold', pad=TITLE_PAD)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def plot_spatial_focusing(results_list, x_vec, dx, output_path):
    plt.figure(figsize=FIG_SIZE_PROFILE)
    ordered_results = get_ordered_results(results_list)
    if not ordered_results:
        return
    ny = ordered_results[0]['tau_peak'].shape[1]
    y_idx = ny // 2
    fwhm_y_offset = 0.0
    for item in ordered_results:
        color = get_method_color(item['name'])
        profile = item['tau_peak'][:, y_idx]
        profile_norm = profile / np.max(profile)
        above_half = np.where(profile_norm >= 0.5)[0]
        if len(above_half) > 0:
            fwhm_idx = above_half[-1] - above_half[0]
            fwhm_mm = fwhm_idx * dx * 1000
            plt.text(x_vec.min() * 1000 + 2, 1 + fwhm_y_offset, f"{LABEL_MAP.get(item['name'], item['name'])}: FWHM={fwhm_mm:.0f} mm", fontsize=LEGEND_SIZE - 2, color=color, ha='left', va='top')
        plt.plot(x_vec * 1000, profile_norm, linewidth=LINE_WIDTH, label=LABEL_MAP.get(item['name'], item['name']), color=color)
        fwhm_y_offset -= 0.08
    plt.xlabel('x [mm]', fontweight='bold')
    plt.ylabel('Normalized peak stress', fontweight='bold')
    plt.title('Spatial focusing profile (normalized)', fontweight='bold', pad=TITLE_PAD)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    plt.xlim(x_vec.min() * 1000, x_vec.max() * 1000)
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def plot_cross_section_comparison(results_list, x_vec, output_path):
    plt.figure(figsize=FIG_SIZE_PROFILE)
    ordered_results = get_ordered_results(results_list)
    if not ordered_results:
        return
    ny = ordered_results[0]['tau_rms'].shape[1]
    y_idx = ny // 2
    for item in ordered_results:
        color = get_method_color(item['name'])
        profile = item['tau_rms'][:, y_idx]
        plt.plot(x_vec * 1000, profile, linewidth=LINE_WIDTH, label=LABEL_MAP.get(item['name'], item['name']), color=color)
    plt.xlabel('x [mm]', fontweight='bold')
    plt.ylabel('Shear stress RMS [Pa]', fontweight='bold')
    plt.title('Cross-section comparison (y=0)', fontweight='bold', pad=TITLE_PAD)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
    plt.xlim(x_vec.min() * 1000, x_vec.max() * 1000)
    plt.tight_layout(pad=LAYOUT_PAD)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def main():
    setup_style()
    ensure_dir(OUTPUT_DIR)
    try:
        mat_data = load_data(MAT_FILE_PATH)
    except FileNotFoundError:
        print('Please run the MATLAB simulation first to generate the data file.')
        return

    x_vec = mat_data['x_vec']
    y_vec = mat_data['y_vec']
    dt = mat_data['dt']
    grid_cfg = mat_data['grid_cfg']
    dx = grid_cfg.dx
    results = extract_results(mat_data)

    global_max_rms = max([np.max(r['tau_rms']) for r in results])
    global_max_peak = max([np.max(r['tau_peak']) for r in results])
    global_max_rate = max([np.max(r['tau_dt_peak']) for r in results])
    global_max_grad = max([np.max(r['grad_mag']) for r in results])

    global_max_tau_xy = 0
    for res in results:
        snapshots = res['tau_xy_snapshots']
        if snapshots is not None and snapshots.ndim == 3 and snapshots.shape[2] >= 4:
            for i in range(4):
                snap = snapshots[:, :, i] - np.mean(snapshots[:, :, i])
                global_max_tau_xy = max(global_max_tau_xy, np.max(np.abs(snap)))

    global_max_uz = 0
    for res in results:
        snapshots = res['uz_snapshots']
        if snapshots is not None and snapshots.ndim == 3 and snapshots.shape[2] >= 4:
            for i in range(4):
                snap = snapshots[:, :, i] - np.mean(snapshots[:, :, i])
                global_max_uz = max(global_max_uz, np.max(np.abs(snap)))

    global_max_xt_signed = 0
    for res in results:
        t_vec = res['t_vec']
        xt_data = res['xt_slice_signed']
        t_end = t_vec[-1]
        t_start = max(0, t_end - ZOOM_DURATION)
        mask = t_vec >= t_start
        data_zoom = xt_data[:, mask]
        data_zoom = data_zoom - np.mean(data_zoom, axis=1, keepdims=True)
        global_max_xt_signed = max(global_max_xt_signed, np.max(np.abs(data_zoom)))

    print('Generating plots...')
    for res in results:
        name = res['name']
        label = LABEL_MAP.get(name, name)
        suffix = name.lower()
        plot_heatmap(res['tau_rms'], x_vec, y_vec, f'{label} | RMS shear stress', 'RMS [Pa]', CMAP_ENERGY, os.path.join(OUTPUT_DIR, f'{suffix}_1_energy_rms.png'), vmax=global_max_rms)
        plot_heatmap(res['tau_peak'], x_vec, y_vec, f'{label} | Peak shear stress', 'Peak stress [Pa]', CMAP_STRESS, os.path.join(OUTPUT_DIR, f'{suffix}_2_peak_stress.png'), vmax=global_max_peak)
        plot_heatmap(res['tau_dt_peak'], x_vec, y_vec, f'{label} | Max rate of change', '|dτ/dt| [Pa/s]', CMAP_RATE, os.path.join(OUTPUT_DIR, f'{suffix}_3_rate_of_change.png'), vmax=global_max_rate)
        plot_heatmap(res['grad_mag'], x_vec, y_vec, f'{label} | Stress gradient snapshot', '|∇τ| [Pa/m]', CMAP_GRAD, os.path.join(OUTPUT_DIR, f'{suffix}_4_gradient_snapshot.png'), vmax=global_max_grad)
        plot_waveform(res['t_vec'], res['center_waveform'], f'{label} | Center waveform', os.path.join(OUTPUT_DIR, f'{suffix}_5_waveform.png'))
        plot_xt_diagram_signed(res['xt_slice_signed'], res['t_vec'], x_vec, f'{label} | Signed XT diagram (tau_xz)', os.path.join(OUTPUT_DIR, f'{suffix}_xt_diagram_tau_xz.png'), vmax=global_max_xt_signed)
        plot_wavefront_snapshots(res['tau_xy_snapshots'], x_vec, y_vec, f'{label} | Wavefront snapshots (tau_xy)', os.path.join(OUTPUT_DIR, f'{suffix}_snapshots_tau_xy.png'), unit_label='[Pa]', vmax=global_max_tau_xy)
        plot_wavefront_snapshots(res['uz_snapshots'], x_vec, y_vec, f'{label} | Wavefront snapshots (uz)', os.path.join(OUTPUT_DIR, f'{suffix}_snapshots_uz.png'), unit_label='[m/s]', vmax=global_max_uz)

    ordered_results = get_ordered_results(results)
    ffi_values = {item['name']: compute_frequency_fidelity_index(item, dt) for item in ordered_results}
    dwci_values = {item['name']: compute_directional_wavefront_concentration_index(item, dt) for item in ordered_results}

    plot_metric_bar(ordered_results, ffi_values, 'Frequency fidelity index', 'FFI', os.path.join(OUTPUT_DIR, 'comparison_frequency_fidelity_index.png'))
    plot_metric_bar(ordered_results, dwci_values, 'Directional wavefront concentration index', 'DWCI', os.path.join(OUTPUT_DIR, 'comparison_directional_wavefront_concentration_index.png'))
    plot_spectrum_comparison(ordered_results, dt, os.path.join(OUTPUT_DIR, 'comparison_spectrum.png'))
    plot_spatial_focusing(ordered_results, x_vec, dx, os.path.join(OUTPUT_DIR, 'comparison_spatial_focusing.png'))
    plot_cross_section_comparison(ordered_results, x_vec, os.path.join(OUTPUT_DIR, 'comparison_cross_section_rms.png'))
    print('All plots generated successfully.')


if __name__ == '__main__':
    main()
