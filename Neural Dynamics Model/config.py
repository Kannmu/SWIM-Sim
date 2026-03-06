import os
import sys

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'Sim', 'Outputs_Experiment1')
DATA_FILE = os.path.join(DATA_DIR, 'experiment1_data.mat')

# Topography Parameters
ROI_SIZE_MM = 40.0
N_RECEPTORS = 320
MIN_DISTANCE_MM = 1.8
RANDOM_SEED = 2025

# Spatial Pooling Parameters
SPATIAL_SIGMA_MM = 2.0  # Gaussian kernel sigma for spatial pooling

# Biomechanical Filter Parameters
FS_MODEL = 10000.0  # 10 kHz
DT_MS = 0.1         # 0.1 ms (derived from FS_MODEL)
FILTER_ORDER = 2
F_LOW_HZ = 100.0
F_HIGH_HZ = 400.0

# LIF Model Parameters
TAU_M_MS = 2.0      # Membrane time constant
V_REST = 0.0
V_RESET = 0.0
V_THRESH = 1.0
R_M = 1.0
T_REF_MS = 1.0      # Refractory period
GLOBAL_GAIN = 1.0   # To be calibrated

# Decoding Parameters
DECODING_WINDOW_MS = 50.0
DENSITY_GRID_MM = 1.0
DENSITY_SIGMA_MM = 2.0
FFI_SIGNAL_WINDOW_START_HZ = 180
FFI_SIGNAL_WINDOW_END_HZ = 220
FFI_NOISE_WINDOW_START_HZ = 380
FFI_NOISE_WINDOW_END_HZ = 420
FFI_EPSILON = 1e-9

# Simulation Parameters
STIMULUS_METHODS = ['DLM_2', 'DLM_3', 'ULM_L', 'LM_L', 'LM_C']
CALIBRATION_METHOD = 'LM_C'
CALIBRATION_TARGET_RATE = 0.2  # spike per 5ms cycle (median active receptor)
CALIBRATION_CYCLE_MS = 5.0

USE_GPU = True
GPU_DEVICE_ID = 0

if sys.platform.startswith("win"):
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "0")
