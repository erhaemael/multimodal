# src/utils/eda.py
import numpy as np
from scipy.signal import find_peaks

def extract_scr_features(eda_signal, sample_rate=32):
    peaks, _ = find_peaks(eda_signal, height=0.01, distance=sample_rate * 1)  # Minimal 1 detik antar peak
    num_scr_peaks = len(peaks)
    avg_scr_amplitude = np.mean(eda_signal[peaks]) if num_scr_peaks > 0 else 0
    return num_scr_peaks, avg_scr_amplitude

def extract_scl_feature(eda_signal):
    scl = np.mean(eda_signal)
    return scl

def extract_eda_features(eda_signal, sample_rate=32):
    num_scr_peaks, avg_scr_amp = extract_scr_features(eda_signal, sample_rate)
    scl = extract_scl_feature(eda_signal)
    return {
        "SCR_count": num_scr_peaks,
        "SCR_avg_amplitude": avg_scr_amp,
        "SCL_mean": scl
    }
