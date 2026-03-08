import os
import sys

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'Sim', 'Outputs_Experiment1')
DATA_FILE = os.path.join(DATA_DIR, 'experiment1_data.mat')

# Topography
ROI_SIZE_MM = 40.0
N_RECEPTORS = 320
MIN_DISTANCE_MM = 1.8
RANDOM_SEED = 2025
SEED_RUN_COUNT = 1
SEED_STRIDE = 1

# Time / preprocessing
STEADY_STATE_WINDOW_MS = 50.0
FS_MODEL = 10000.0
DT_MS = 0.1
FILTER_ORDER = 4
F_LOW_HZ = 80.0
F_HIGH_HZ = 900.0
SPATIAL_SIGMA_MM = 2.0

# Fixed receptor temporal tuning
ENABLE_TEMPORAL_TUNING = True
TEMPORAL_TUNING_CENTER_HZ = 200.0
TEMPORAL_TUNING_SIGMA_OCT = 0.50

# Stimuli
STIMULUS_METHODS = ['DLM_2', 'DLM_3', 'ULM_L', 'LM_L', 'LM_C']
PAIRWISE_METHODS = ['DLM_2', 'DLM_3', 'ULM_L', 'LM_L', 'LM_C']

# Bridge / mechanistic diagnostics
BRIDGE_TARGET_FREQ_HZ = 200.0
BRIDGE_SCORE_EPS = 1e-12
FIDELITY_FREQS_HZ = (200.0, 400.0, 600.0, 800.0)
DENSITY_GRID_MM = 1.0
DENSITY_SIGMA_MM = 2.0

USE_GPU = True
GPU_DEVICE_ID = 0

if sys.platform.startswith("win"):
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "0")
