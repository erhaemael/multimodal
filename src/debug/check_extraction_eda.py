# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# import pickle

# def load_eda_from_pickle(pickle_path):
#     with open(pickle_path, "rb") as f:
#         data = pickle.load(f, encoding="latin1")
#     eda = data["signal"]["wrist"]["EDA"]
#     return eda

# # Extracts features related to Skin Conductance Response (SCR) from an EDA signal
# def extract_scr_features(eda_signal, sample_rate=32):
#     # Detect peaks in the EDA signal with a minimum height and spacing (at least 1 second apart)
#     peaks, _ = find_peaks(eda_signal, height=0.01, distance=sample_rate * 1)
    
#     # Count the number of SCR peaks
#     num_scr_peaks = len(peaks)
    
#     # Calculate the average amplitude of detected SCR peaks, or 0 if no peaks are found
#     avg_scr_amplitude = np.mean(eda_signal[peaks]) if num_scr_peaks > 0 else 0
    
#     return num_scr_peaks, avg_scr_amplitude

# # Extracts the Skin Conductance Level (SCL) feature as the mean of the EDA signal
# def extract_scl_feature(eda_signal):
#     scl = np.mean(eda_signal)
#     return scl

# # Aggregates all EDA-related features (SCR and SCL) into a single dictionary
# def extract_eda_features(eda_signal, sample_rate=32):
#     num_scr_peaks, avg_scr_amp = extract_scr_features(eda_signal, sample_rate)
#     scl = extract_scl_feature(eda_signal)
    
#     return {
#         "SCR_count": num_scr_peaks,              # Number of SCR peaks detected
#         "SCR_avg_amplitude": avg_scr_amp,        # Average amplitude of SCR peaks
#         "SCL_mean": scl                           # Mean skin conductance level
#     }

# def plot_eda_with_peaks(eda_signal, sample_rate=32):
#     peaks, _ = find_peaks(eda_signal, height=0.01, distance=sample_rate)
#     plt.plot(eda_signal, label="EDA signal")
#     plt.plot(peaks, eda_signal[peaks], "x", color="red", label="SCR Peaks")
#     plt.legend()
#     plt.title("EDA Signal with Detected SCR Peaks")
#     plt.show()

# def main():
#     # Ganti path ke file pickle pasien, misal:
#     pickle_path = "data/raw/WESAD/S2/S2.pkl"  # sesuaikan lokasi file
#     eda_full = load_eda_from_pickle(pickle_path)

#     # Ambil 60 detik pertama (32Hz × 60 = 1920 sample)
#     eda_signal = eda_full[0:32*60]  # Window pertama

#     # Ekstraksi fitur
#     features = extract_eda_features(eda_signal, sample_rate=32)
#     print("Extracted EDA Features:", features)

#     # Visualisasi
#     plot_eda_with_peaks(eda_signal, sample_rate=32)

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Extracts features related to Skin Conductance Response (SCR) from an EDA signal
def extract_scr_features(eda_signal, sample_rate=32):
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
def extract_eda_features(eda_signal, sample_rate):
    num_scr_peaks, avg_scr_amp = extract_scr_features(eda_signal, sample_rate)
    scl = extract_scl_feature(eda_signal)
    
    return {
        "SCR_count": num_scr_peaks,              # Number of SCR peaks detected
        "SCR_avg_amplitude": avg_scr_amp,        # Average amplitude of SCR peaks
        "SCL_mean": scl                           # Mean skin conductance level
    }

# Contoh sinyal EDA sederhana
eda_test = np.array([0, 0.01, 0.05, 0.1, 0.08, 0.03, 0, 0.02, 0.06, 0.09, 0.04, 0.01])
features = extract_eda_features(eda_test, 4)  # sample_rate lebih kecil untuk contoh ini
print(features)

# Visualisasi puncak
peaks, _ = find_peaks(eda_test, height=0.01, distance=4)
plt.plot(eda_test)
plt.plot(peaks, eda_test[peaks], "x")
plt.show()
