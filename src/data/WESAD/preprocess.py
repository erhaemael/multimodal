import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd

from src.data.WESAD.constants import DATASET_PATH, PREPROCESSED_CSV, TARGET_SR
from src.utils.resampling_pipeline import resample_signals
from src.utils.bvp import extract_features as extract_bvp_features
from src.utils.eda import extract_eda_features

# Define sliding window and offset (in seconds and samples)
WINDOW_SIZE_SEC = 60         # Window duration in seconds
OFFSET_SIZE_SEC = 10         # Sliding offset in seconds
SAMPLE_RATE = TARGET_SR      # Target sample rate after resampling
WINDOW_SIZE = WINDOW_SIZE_SEC * SAMPLE_RATE  # Window size in samples
OFFSET_SIZE = OFFSET_SIZE_SEC * SAMPLE_RATE  # Offset size in samples

def downsample_labels_majority(labels: np.ndarray, original_sr=700, target_sr=TARGET_SR) -> np.ndarray:
    """
    Downsample label array using majority voting within each resampling window.

    Args:
        labels (np.ndarray): Original label array (1D).
        original_sr (int): Original sampling rate of the labels.
        target_sr (int): Target sampling rate to downsample to.

    Returns:
        np.ndarray: Downsampled label array.
    """
    factor = original_sr // target_sr
    num_samples = len(labels) // factor
    downsampled_labels = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        window = labels[i * factor:(i + 1) * factor]
        downsampled_labels[i] = np.argmax(np.bincount(window))  # Majority label in the window

    return downsampled_labels

def load_patient_data(pickle_path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load and preprocess signal data and labels from a WESAD pickle file.

    Args:
        pickle_path (str): Path to the pickle file.

    Returns:
        tuple:
            - dict of signals (TEMP, BVP, EDA)
            - downsampled binary labels (1 = stress, 0 = non-stress)
    """
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file, encoding="latin1")

    # Labels (0 = not defined/transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation)
    labels = data["label"]
    # Set the stress label to 1 and the others to 0
    labels = np.array([1 if l == 2 else 0 for l in labels])
    # Downsample labels
    labels_resampled = downsample_labels_majority(labels)   

    wrist_signals = data["signal"]["wrist"]
    signals = {
        "TEMP": wrist_signals["TEMP"],
        "BVP": wrist_signals["BVP"],
        "EDA": wrist_signals["EDA"]
    }

    return signals, labels_resampled

def main():
    """
    Main preprocessing routine:
    - Iterates over each patient in the WESAD dataset
    - Loads and resamples physiological signals
    - Applies sliding window to extract features
    - Saves the final processed dataset to a CSV file
    """
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

        # Load and resample signal data and labels
        signals, labels = load_patient_data(pickle_path)
        resampled_signals = resample_signals(signals)

        # Determine the minimum length across all signals and labels to avoid mismatch
        min_len = min(len(labels), *[len(resampled_signals[s]) for s in resampled_signals])
        labels = labels[:min_len]

        all_signals = []
        all_labels = []

        # Sliding window feature extraction
        for end in range(WINDOW_SIZE, min_len, OFFSET_SIZE):
            start = end - WINDOW_SIZE
            w_signals = {"Timestamp": start // SAMPLE_RATE}

            # Extract window slices for each signal
            bvp_window = resampled_signals["BVP"][start:end]
            eda_window = resampled_signals["EDA"][start:end]
            temp_window = resampled_signals["TEMP"][start:end]
            label_window = labels[start:end]

            # Extract features from BVP and EDA windows
            bvp_features = extract_bvp_features(bvp_window, sample_rate=SAMPLE_RATE)
            eda_features = extract_eda_features(eda_window, sampling_rate=SAMPLE_RATE)

            # Compute mean temperature in the window
            temp_mean = np.mean(temp_window)

            # Determine majority label for the current window
            window_label = np.argmax(np.bincount(label_window))

            # Merge all features into a dictionary
            w_signals.update({
                **bvp_features,
                **eda_features,
                "TEMP_mean": temp_mean,
            })

            all_signals.append(w_signals)
            all_labels.append(window_label)

        # Convert per-patient results to DataFrame
        final_df = pd.DataFrame(all_signals)
        final_df["Label"] = all_labels
        final_df["Patient"] = patient

        features_data.append(final_df)

    # Concatenate all patient data and export to CSV
    final_df = pd.concat(features_data, ignore_index=True)
    final_df.to_csv(PREPROCESSED_CSV, index=False)
    print(f"Preprocessed WESAD data with features saved to {PREPROCESSED_CSV}")

if __name__ == "__main__":
    main()
