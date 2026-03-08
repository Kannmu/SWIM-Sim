from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swim_model.mechanics.coherent_integration import CoherentIntegrator


roi = np.array([-0.001, 0.001], dtype=np.float64)
receptors = np.array([[0.0, 0.0]], dtype=np.float64)
integrator = CoherentIntegrator(roi, roi, receptors, conduction_velocity_m_s=5.0, spatial_decay_lambda_m=0.004, dt=1e-4, chunk_size=2)
tau_dyn = np.zeros((2, 2, 20), dtype=np.float32)
tau_dyn[0, 0, 5] = 1.0
out = integrator.integrate(tau_dyn, np)
assert out.shape == (1, 20)
assert float(out.max()) > 0.0
