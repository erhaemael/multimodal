import neurokit2 as nk
import numpy as np
import pandas as pd

def extract_eda_features(eda_signal: np.ndarray, sampling_rate: int):
    """
    Extract EDA features from a given EDA signal.

    Parameters:
    - eda_signal: np.ndarray, preprocessed EDA signal.
    - sampling_rate: int, the sampling rate of the signal.

    Returns:
    - features: dict, extracted EDA features:
        - 'SCR_count': Number of Skin Conductance Responses (SCRs) detected.
        - 'SCR_avg_amplitude': Average amplitude of valid SCRs.
        - 'SCL_mean': Mean tonic Skin Conductance Level (SCL).
    """
    try:
        # Clean the raw EDA signal
        eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate)

        # Process the cleaned EDA signal to extract relevant features
        # Returns a DataFrame of signals (phasic, tonic, SCR) and info dictionary
        signals, info = nk.eda_process(eda_cleaned, sampling_rate=sampling_rate)

        # Extract the SCR amplitude series
        scr_amplitudes = signals["SCR_Amplitude"]

        # Clean NaN values in SCR amplitude (if any)
        # Fill with 0 to avoid skewing the mean calculation
        scr_amplitudes_clean = scr_amplitudes.fillna(0) if isinstance(scr_amplitudes, pd.Series) else np.nan_to_num(scr_amplitudes)

        # Define physiological threshold for valid SCRs
        threshold = 0.01
        scr_valid = scr_amplitudes_clean[scr_amplitudes_clean > threshold]

        # Count the number of valid SCRs
        scr_count = len(scr_valid)

        # Compute the average amplitude of valid SCRs
        scr_avg_amplitude = np.mean(scr_valid) if scr_count > 0 else 0

        # Compute the mean tonic component (SCL)
        scl_mean = np.mean(signals["EDA_Tonic"])

        # Return the extracted features as a dictionary
        return {
            "SCR_count": scr_count,
            "SCR_avg_amplitude": scr_avg_amplitude,
            "SCL_mean": scl_mean,
        }

    except Exception as e:
        # In case of failure, print a warning and return zeroed features
        print(f"[WARNING] Failed to extract EDA features: {e}")
        return {
            "SCR_count": 0,
            "SCR_avg_amplitude": 0,
            "SCL_mean": 0,
        }
