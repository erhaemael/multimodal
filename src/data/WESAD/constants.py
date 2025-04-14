# Constants for the WESAD dataset
DATASET_PATH = "data/raw/WESAD"
PREPROCESSED_CSV = "data/preprocessed/WESAD.csv"
SIGNALS = ["ACC", "BVP", "EDA", "TEMP"]
ORIGINAL_SR = {
    "ACC": 32,  # Hz
    "BVP": 64,  # Hz
    "EDA": 4,  # Hz
    "TEMP": 4,  # Hz
    "label": 700  # Hz
}

# Target sampling rate for all signals (after resampling)
TARGET_SR = 64  # Hz
