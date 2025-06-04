import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wandb

from WESAD.constants import PREPROCESSED_CSV as WESAD_CSV

BVP_FEATURES = ['HR_BVP', 'HRV_BVP', 'SCR_count', 'SCR_avg_amplitude','SCL_mean', 'TEMP_mean']
CONF_FEATURES = ['ACC_x', 'ACC_y', 'ACC_z']

class Features(Dataset):
    def __init__(self, dataset: str, flag="train", k_split=5, k=0, scaler=None, confounding=False, step=1):
        """
        Load the Features dataset for anomaly detection.

        Args:
            dataset (str): Dataset to use.
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            confounding (bool): Whether to include confounding features (default: False)
            step (int): Step for downsampling the data (default: 1 - no downsampling)
        """
        self.flag = flag
        if scaler is None:
            raise ValueError("Scaler must be provided")
        self.scaler = scaler
        print(f"Features dataset with flag {flag}")

        # Get the data folder
        base_path = os.getenv("BASE_PATH")
        if base_path is None:
            raise ValueError("BASE_PATH environment variable not set")

        # Read the data from the correct dataset
        if dataset == "WESAD":
            path = os.path.join(base_path, WESAD_CSV)
        else:
            raise ValueError(f"Dataset must be one of 'WESAD'")
        data = pd.read_csv(path)

        # Remove confounding columns if specified
        if not confounding:
            # Remove confounding features if present
            for feature in CONF_FEATURES:
                if feature in data.columns:
                    data = data.drop(columns=[feature])

        # Skip "Patient" and "timestamp" columns
        data = data.drop(columns=['Patient', 'Timestamp'])
        print("Remaining columns:", list(data.columns))
        data = np.nan_to_num(data.values)
        size = len(data)

        # Indices of the test split are the k-th split of data with k_split
        start_test = int(size * k / k_split)
        end_test = start_test + int(size / k_split)

        # Split the data
        if flag == 'train':
            # If k is the first split, take the rest of the data
            if k == 0:
                self.split = data[end_test:]
            # If k is the last split, take the beginning of the data
            elif k == k_split - 1:
                self.split = data[:start_test]
            # Otherwise, take the two parts of the data
            else:
                self.split = np.concatenate(
                    (data[:start_test], data[end_test:]))
            # Remove all elements with anomaly
            self.split = self.split[self.split[:, -1] == 0]
            # Split into labels and data
            self.labels = self.split[:, -1]
            self.split = self.split[:, :-1]
        elif flag == 'test':
            # Take the split
            self.split = data[start_test:end_test, :]
            # Split into labels and data
            self.labels = self.split[:, -1]
            self.split = self.split[:, :-1]
        else:
            raise ValueError(f"Flag must be one of ['train', 'test']")
        
        # Downsample the data
        self.step = step
        if self.step > 1:
            self.labels = self.labels[::self.step]
            self.split = self.split[::self.step]

        # Normalize the data
        if type(self.scaler) == MinMaxScaler or type(self.scaler) == StandardScaler:
            self.split = self.scaler.fit_transform(self.split)
        else:
            self.split = self.scaler(self.split)

        # Get the labels
        n_anomalies = len(np.where(self.labels == 1)[0])
        tot = len(self.split)
        # print(f"Features {flag} set - #anomalies: {n_anomalies}/{tot}")

        # Wandb logging
        if wandb.run is not None:
            wandb.log({
                f'n_anomalies_{flag}': n_anomalies,
                f'tot_{flag}': tot
            })

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class FeaturesUniTS(Features):
    def __init__(self, dataset: str, win_size: int, step=1, flag="train", k_split=5, k=0, confounding=False):
        """
        Load the Features dataset for anomaly detection.

        Args:
            dataset (str): Dataset to use.
            win_size (int): Size of the sliding window.
            step (int): Step of the sliding window (default: 1)
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
        """
        scaler = StandardScaler()
        super().__init__(
            dataset=dataset,
            flag=flag,
            k_split=k_split,
            k=k,
            scaler=scaler,
            confounding=confounding,
            step=step
        )
        self.win_size = win_size

    def __len__(self):
        return len(self.split) - self.win_size + 1

    def __getitem__(self, index):
        x = np.float32(self.split[index:index + self.win_size])
        if self.flag == 'test':
            y = np.float32(self.labels[index:index + self.win_size])
        else:
            y = np.zeros(self.win_size)
        
        return x, y
