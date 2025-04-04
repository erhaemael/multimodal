from src.data.WESAD.preprocess import load_patient_data
from src.utils.resampling_pipeline import bvp_chebyshev_resample
import matplotlib.pyplot as plt
import numpy as np

signals, _ = load_patient_data('data/raw/WESAD/S10/S10.pkl')
bvp_signal = signals['BVP'].flatten()
bvp_filtered = bvp_chebyshev_resample(bvp_signal, original_sr=64, target_sr=32)

plt.figure(figsize=(12, 5))
plt.plot(bvp_signal[:1000], label='Original BVP (64 Hz)')
plt.plot(np.linspace(0, 1000, len(bvp_filtered[:500])), bvp_filtered[:500], label='Filtered BVP (Chebyshev)', linestyle='--')
plt.legend()
plt.title("BVP Original vs After Filtering (S10)")
plt.show()
