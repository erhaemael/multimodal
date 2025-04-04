# src/data/WESAD/preprocess.py
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd

from src.data.WESAD.constants import DATASET_PATH, PREPROCESSED_CSV
from src.utils.resampling_pipeline import resample_signals

def downsample_labels_majority(labels: np.ndarray, original_sr=700, target_sr=32) -> np.ndarray:
    """Downsample label dari 700 Hz ke 32 Hz menggunakan mayoritas voting dalam window."""
    factor = original_sr // target_sr  # Misalnya 700/32 ≈ 22
    num_samples = len(labels) // factor  # Jumlah sampel setelah downsample
    
    downsampled_labels = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        window = labels[i * factor:(i + 1) * factor]  # Ambil window data
        downsampled_labels[i] = np.argmax(np.bincount(window))  # Mayoritas voting
    
    return downsampled_labels

def load_patient_data(pickle_path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load patient data from the WESAD dataset.
    """
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file, encoding="latin1")

    # Ambil label 700 Hz dan ubah ke biner
    labels = data["label"]
    labels = np.array([1 if l == 2 else 0 for l in labels])  # Ubah label ke biner (1=stress, 0=non-stress)

    # Downsample label dari 700 Hz ke 32 Hz
    labels_resampled = downsample_labels_majority(labels, original_sr=700, target_sr=32)

    # Ambil sinyal wrist
    wrist_signals = data["signal"]["wrist"]
    signals = {
        "ACC": wrist_signals["ACC"],
        "TEMP": wrist_signals["TEMP"],
        "BVP": wrist_signals["BVP"],
        "EDA": wrist_signals["EDA"]
    }
    return signals, labels_resampled

def main():
    if os.path.exists(PREPROCESSED_CSV):
        print("Preprocessed data already exists. Skipping.")
        return

    resampled_data = []
    patients = os.listdir(DATASET_PATH)

    for patient in tqdm(patients, desc="WESAD resampling", position=0):
        patient_path = os.path.join(DATASET_PATH, patient)
        pickle_path = os.path.join(patient_path, f"{patient}.pkl")

        if not os.path.exists(pickle_path):
            print(f"Skipping {patient}, no data found.")
            continue

        signals, labels = load_patient_data(pickle_path)

        # Resampling semua sinyal ke 32 Hz
        resampled_signals = resample_signals(signals)

        # Flatten ACC jika perlu
        acc_data = resampled_signals["ACC"]
        if acc_data.ndim > 2:
            acc_data = acc_data.reshape(acc_data.shape[0], -1)

        # Pastikan panjang label dan sinyal sama
        min_len = min(len(labels), *[len(resampled_signals[s]) for s in resampled_signals])
        labels = labels[:min_len]

        # Pastikan semua sinyal memiliki panjang yang sama
        df = pd.DataFrame({
            "BVP": resampled_signals["BVP"][:min_len].flatten(),
            "EDA": resampled_signals["EDA"][:min_len].flatten(),
            "TEMP": resampled_signals["TEMP"][:min_len].flatten(),
            "ACC_x": acc_data[:min_len, 0],
            "ACC_y": acc_data[:min_len, 1],
            "ACC_z": acc_data[:min_len, 2],
            "Label": labels,
            "Patient": patient
        })

        resampled_data.append(df)

    final_df = pd.concat(resampled_data, ignore_index=True)
    final_df.to_csv(PREPROCESSED_CSV, index=False)
    print(f"Resampled WESAD data saved to {PREPROCESSED_CSV}")

if __name__ == "__main__":
    main()
