import os
import pickle
import numpy as np
import pandas as pd

DATASET_PATH = "data/raw/WESAD"
SIGNALS = ["BVP", "EDA", "TEMP"]

patients = sorted(os.listdir(DATASET_PATH))
missing_matrix = pd.DataFrame(columns=SIGNALS)

for patient in patients:
    pickle_path = os.path.join(DATASET_PATH, patient, f"{patient}.pkl")
    if not os.path.exists(pickle_path):
        print(f"Skipping {patient} (missing pickle)")
        continue

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
        wrist = data["signal"]["wrist"]

        row_result = {}
        for signal in SIGNALS:
            signal_data = wrist[signal]
            if signal_data.ndim > 1:
                signal_data = signal_data.flatten()
            has_nan = np.isnan(signal_data).any()
            row_result[signal] = int(has_nan)

        missing_matrix.loc[patient] = row_result

# 0 = tidak ada NaN, 1 = ada NaN
missing_matrix = missing_matrix.astype(int)

print(missing_matrix)

# Simpan ke CSV jika diinginkan
# missing_matrix.to_csv("missing_check_result.csv")
