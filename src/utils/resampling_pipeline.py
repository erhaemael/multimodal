from typing import Dict
import numpy as np
from scipy.signal import cheby1, filtfilt, gaussian, convolve
from scipy.interpolate import interp1d
from src.data.WESAD.constants import TARGET_SR  # Misalnya 64 Hz

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

def resample_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Resampling semua sinyal ke TARGET_SR Hz dan menyamakan panjang."""

    # BVP: diasumsikan sudah 64 Hz
    bvp_signal = signals['BVP'].flatten()

    # EDA: Upsample dari 4 Hz -> 64 Hz, lalu smoothing
    eda_signal = signals['EDA'].flatten()
    eda_upsampled = upsample_interp(eda_signal, original_sr=4, target_sr=TARGET_SR)
    eda_smoothed = gaussian_smoothing(eda_upsampled, window_size=40, sigma=400)

    # TEMP: Upsample dari 4 Hz -> 64 Hz, kemudian bersihkan (opsional)
    temp_signal = signals['TEMP'].flatten()
    temp_upsampled = upsample_interp(temp_signal, original_sr=4, target_sr=TARGET_SR)
    temp_cleaned = np.clip(temp_upsampled, 20, 50)  # bisa juga pakai z-score outlier cleaning

    # ACC: Upsample dari 32 Hz -> 64 Hz
    # acc_data = signals['ACC']  # shape (N, 3)
    # acc_upsampled = np.stack([
    #     upsample_interp(acc_data[:, i], original_sr=32, target_sr=TARGET_SR)
    #     for i in range(acc_data.shape[1])
    # ], axis=-1)

    # Sinkronisasi panjang
    # min_len = min(len(bvp_signal), len(eda_smoothed), len(temp_cleaned), acc_upsampled.shape[0])
    min_len = min(len(bvp_signal), len(eda_smoothed), len(temp_cleaned))

    return {
        "BVP": bvp_signal[:min_len],
        "EDA": eda_smoothed[:min_len],
        "TEMP": temp_cleaned[:min_len],
        # "ACC": acc_upsampled[:min_len, :]
    }
