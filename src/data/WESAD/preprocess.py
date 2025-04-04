import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd

from src.data.WESAD.constants import DATASET_PATH, PREPROCESSED_CSV
from src.utils.resampling_pipeline import resample_signals
from src.utils.bvp import extract_features as extract_bvp_features
from src.utils.eda import extract_eda_features

WINDOW_SIZE_SEC = 60
OFFSET_SIZE_SEC = 10
SAMPLE_RATE = 32
WINDOW_SIZE = WINDOW_SIZE_SEC * SAMPLE_RATE
OFFSET_SIZE = OFFSET_SIZE_SEC * SAMPLE_RATE

def downsample_labels_majority(labels: np.ndarray, original_sr=700, target_sr=32) -> np.ndarray:
    factor = original_sr // target_sr
    num_samples = len(labels) // factor
    downsampled_labels = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        window = labels[i * factor:(i + 1) * factor]
        downsampled_labels[i] = np.argmax(np.bincount(window))
    return downsampled_labels

def load_patient_data(pickle_path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file, encoding="latin1")
    labels = data["label"]
    labels = np.array([1 if l == 2 else 0 for l in labels])
    labels_resampled = downsample_labels_majority(labels)
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

    features_data = []
    patients = os.listdir(DATASET_PATH)

    for patient in tqdm(patients, desc="WESAD preprocessing", position=0):
        patient_path = os.path.join(DATASET_PATH, patient)
        pickle_path = os.path.join(patient_path, f"{patient}.pkl")

        if not os.path.exists(pickle_path):
            print(f"Skipping {patient}, no data found.")
            continue

        signals, labels = load_patient_data(pickle_path)
        resampled_signals = resample_signals(signals)

        acc_data = resampled_signals["ACC"]
        if acc_data.ndim > 2:
            acc_data = acc_data.reshape(acc_data.shape[0], -1)

        min_len = min(len(labels), *[len(resampled_signals[s]) for s in resampled_signals])
        labels = labels[:min_len]

        # Mulai sliding window + fitur extraction
        all_signals = []
        all_labels = []

        for end in range(WINDOW_SIZE, min_len, OFFSET_SIZE):
            start = end - WINDOW_SIZE

            w_signals = {"Timestamp": start // SAMPLE_RATE}

            bvp_window = resampled_signals["BVP"][start:end]
            eda_window = resampled_signals["EDA"][start:end]
            temp_window = resampled_signals["TEMP"][start:end]
            acc_window = acc_data[start:end]
            label_window = labels[start:end]

            # Ekstraksi fitur BVP & EDA
            bvp_features = extract_bvp_features(bvp_window, sample_rate=SAMPLE_RATE)
            eda_features = extract_eda_features(eda_window, sample_rate=SAMPLE_RATE)

            # Mean TEMP dan ACC
            temp_mean = np.mean(temp_window)
            acc_mean = np.mean(acc_window, axis=0)

            # Majority label dalam window
            window_label = np.argmax(np.bincount(label_window))

            w_signals.update({
                **bvp_features,
                **eda_features,
                "TEMP_mean": temp_mean,
                "ACC_x_mean": acc_mean[0],
                "ACC_y_mean": acc_mean[1],
                "ACC_z_mean": acc_mean[2]
            })

            all_signals.append(w_signals)
            all_labels.append(window_label)

        final_df = pd.DataFrame(all_signals)
        final_df["Label"] = all_labels
        final_df["Patient"] = patient

        features_data.append(final_df)

    final_df = pd.concat(features_data, ignore_index=True)
    final_df.to_csv(PREPROCESSED_CSV, index=False)
    print(f"Preprocessed WESAD data with features saved to {PREPROCESSED_CSV}")

if __name__ == "__main__":
    main()
