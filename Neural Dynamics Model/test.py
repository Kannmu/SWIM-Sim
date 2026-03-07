import numpy as np
from src.decoding import PopulationDecoder

decoder = PopulationDecoder(
    roi_area_mm2=40.0**2,
    density_sigma_mm=2.0,
    density_grid_mm=1.0,
    fidelity_freqs_hz=(200.0, 400.0, 600.0, 800.0),
    use_gpu=False,
)

fs = 10000.0
t = np.arange(500) / fs

# Test 1: pure 200 Hz
x200 = np.sin(2*np.pi*200*t)[None, :]
y0 = np.zeros_like(x200)
ffi_200 = decoder.compute_frequency_fidelity(x200, y0, t, target_freq_hz=200.0)[0]

# Test 2: pure 400 Hz
x400 = np.sin(2*np.pi*400*t)[None, :]
ffi_400_as_200 = decoder.compute_frequency_fidelity(x400, y0, t, target_freq_hz=200.0)[0]

print("FFI pure 200Hz:", ffi_200)
print("FFI pure 400Hz measured at 200Hz:", ffi_400_as_200)
