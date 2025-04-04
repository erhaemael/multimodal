import numpy as np
import pandas as pd

def get_samples(
    signals: dict[str, np.ndarray],
    labels: list,
    sample_rates: dict[str, int],
    n: int,
    offset: float
) -> pd.DataFrame:
    """
    Extract the samples with desired signals and labels from the pickle file.

    Args:
        signals (dict[str, np.ndarray]): The signals.
        labels (list): The labels.
        sample_rates (dict[str, int]): The sample rates for each signal.
        n (int): The number of seconds for the window.
        offset (float): The number of seconds to skip.

    Returns:
        pd.DataFrame: All the smartwatch signals and label.
    """

    # Compute the number of labels in seconds
    labels_seconds = int(len(labels) / sample_rates["label"])

    # Tentukan durasi maksimum aman dari semua sinyal agar tidak out of bounds
    min_length_seconds = min([len(signals[s]) / sample_rates[s] for s in signals])

    # Initialize the lists
    all_signals = []
    all_labels = []

    # Looping window per OFFSET detik
    for end in np.arange(n, min(min_length_seconds, labels_seconds), offset):
        # Start and end index
        start = end - n

        # Window signals
        w_signals = {"timestamp": start}

        # Hitung label di window ini
        start_i_label = int(start * sample_rates["label"])
        end_i_label = int(end * sample_rates["label"])
        window_labels = labels[start_i_label:end_i_label]
        window_labels = [int(l) for l in window_labels]
        bin_count = np.bincount(window_labels)
        label = int(bin_count.argmax())

        # Proses masing-masing sinyal
        for s in signals:
            start_i = int(start * sample_rates[s])
            end_i = int(end * sample_rates[s])

            # Tambahan biar aman tidak out of bounds
            end_i = min(end_i, signals[s].shape[0])

            if s == "BVP":
                # Ambil raw BVP window dan hitung statistik dasar
                bvp_window = signals[s][start_i:end_i]
                w_signals["BVP_mean"] = np.mean(bvp_window)
                w_signals["BVP_std"] = np.std(bvp_window)

            elif s == "ACC":
                # Pastikan tidak out of bounds
                if start_i < signals[s].shape[0]:
                    w_signals[f"{s}_x"] = signals[s][start_i, 0]
                    w_signals[f"{s}_y"] = signals[s][start_i, 1]
                    w_signals[f"{s}_z"] = signals[s][start_i, 2]
                else:
                    w_signals[f"{s}_x"] = np.nan
                    w_signals[f"{s}_y"] = np.nan
                    w_signals[f"{s}_z"] = np.nan

            elif s in ["TEMP", "EDA"]:
                if start_i < signals[s].shape[0]:
                    if signals[s].ndim == 2:
                        w_signals[s] = signals[s][start_i, 0]
                    else:
                        w_signals[s] = signals[s][start_i]
                else:
                    w_signals[s] = np.nan

        # Append hasil window dan label
        all_signals.append(w_signals)
        all_labels.append(label)

    # Gabungkan ke DataFrame
    extr_df = pd.DataFrame(all_signals)
    extr_df["Label"] = all_labels

    return extr_df
