from typing import Dict
import numpy as np
from scipy.signal import cheby1, filtfilt, gaussian, convolve
from scipy.interpolate import interp1d

def gaussian_smoothing(signal, window_size=40, sigma=400):
    """Gaussian smoothing untuk meredam noise tanpa menghilangkan terlalu banyak informasi."""
    gauss_win = gaussian(window_size, std=sigma)
    gauss_win /= np.sum(gauss_win)
    return convolve(signal, gauss_win, mode='same')

def upsample_interp(signal, original_sr, target_sr):
    """Interpolasi linier untuk upsampling sinyal."""
    duration = len(signal) / original_sr
    original_time = np.linspace(0, duration, len(signal))
    target_time = np.linspace(0, duration, int(duration * target_sr))
    interpolator = interp1d(original_time, signal, kind='linear', fill_value="extrapolate")
    return interpolator(target_time)

def bvp_chebyshev_resample(bvp_signal, original_sr, target_sr):
    """Resampling BVP dengan low-pass Chebyshev filter untuk menghindari aliasing."""
    nyq = 0.5 * original_sr
    cutoff = 15  # Cutoff < Nyquist target (16 Hz untuk 32Hz target)
    b, a = cheby1(N=8, rp=0.5, Wn=cutoff/nyq, btype='low')
    bvp_filtered = filtfilt(b, a, bvp_signal)

    duration = len(bvp_filtered) / original_sr
    num_target_samples = int(duration * target_sr)
    bvp_resampled = np.interp(
        np.linspace(0, duration, num_target_samples),
        np.linspace(0, duration, len(bvp_filtered)),
        bvp_filtered
    )
    return bvp_resampled

def resample_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Resampling semua sinyal ke 32 Hz dengan filtering yang sesuai."""

    # BVP Processing
    bvp_signal = signals['BVP'].flatten()
    bvp_resampled = bvp_chebyshev_resample(bvp_signal, original_sr=64, target_sr=32)

    # EDA Processing
    eda_signal = signals['EDA'].flatten()
    eda_upsampled = upsample_interp(eda_signal, original_sr=4, target_sr=32)
    eda_smoothed = gaussian_smoothing(eda_upsampled, window_size=40, sigma=400)

    # 🌡 TEMP Processing with Spike Filtering
    temp_signal = signals['TEMP'].flatten()
    # Buang nilai aneh (misal lebih dari 100 derajat) sebelum upsampling
    temp_cleaned = temp_signal[(temp_signal >= 20) & (temp_signal <= 50)]
    temp_upsampled = upsample_interp(temp_cleaned, original_sr=4, target_sr=32)

    # ACC Processing - RAW 3 axis, no resample
    acc_data = signals['ACC']  # Pastikan ACC shape (N, 3)
    assert acc_data.ndim == 2 and acc_data.shape[1] == 3, "ACC harus (N, 3)"

    # Sinkronisasi panjang sinyal
    min_len = min(len(bvp_resampled), len(eda_smoothed), len(temp_upsampled), acc_data.shape[0])

    return {
        "BVP": bvp_resampled[:min_len],
        "EDA": eda_smoothed[:min_len],
        "TEMP": temp_upsampled[:min_len],
        "ACC": acc_data[:min_len, :]  # ACC tetap 3 axis
    }
