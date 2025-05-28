import numpy as np
from scipy.signal import find_peaks

# Extracts features related to Skin Conductance Response (SCR) from an EDA signal
def extract_scr_features(eda_signal, sample_rate=64):
    # Detect peaks in the EDA signal with a minimum height and spacing (at least 1 second apart)
    peaks, _ = find_peaks(eda_signal, height=0.01, distance=sample_rate * 1)
    
    # Count the number of SCR peaks
    num_scr_peaks = len(peaks)
    
    # Calculate the average amplitude of detected SCR peaks, or 0 if no peaks are found
    avg_scr_amplitude = np.mean(eda_signal[peaks]) if num_scr_peaks > 0 else 0
    
    return num_scr_peaks, avg_scr_amplitude

# Extracts the Skin Conductance Level (SCL) feature as the mean of the EDA signal
def extract_scl_feature(eda_signal):
    scl = np.mean(eda_signal)
    return scl

# Aggregates all EDA-related features (SCR and SCL) into a single dictionary
def extract_eda_features(eda_signal, sample_rate=64):
    num_scr_peaks, avg_scr_amp = extract_scr_features(eda_signal, sample_rate)
    scl = extract_scl_feature(eda_signal)
    
    return {
        "SCR_count": num_scr_peaks,              # Number of SCR peaks detected
        "SCR_avg_amplitude": avg_scr_amp,        # Average amplitude of SCR peaks
        "SCL_mean": scl                           # Mean skin conductance level
    }