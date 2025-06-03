import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import torch
from torch.utils.data import DataLoader
import wandb
import random

FIXED_SEED = 444
k = 1

# Path
base_path = os.getenv("BASE_PATH") or os.path.dirname(os.path.realpath(__file__))

# Save as environment variable for next scripts
os.environ["BASE_PATH"] = base_path
os.environ["WANDB_USER"] = "erhaemael-politeknik-negeri-bandung"
WANDB_USER = os.getenv("WANDB_USER")
sys.path.append(os.path.join(base_path, "src/data"))
sys.path.append(os.path.join(base_path, "src/models/UniTS"))

from src.data.FeaturesDataset import Features, FeaturesUniTS
from exp.exp_sup import Exp_All_Task


def get_model_predictions(dataset_name: str, majority_threshold: float = 0):
    wandb.init(project="units_analysis", entity=WANDB_USER)
    # Units dataset
    train_dataset = FeaturesUniTS(dataset_name, 5, 1, "train", k_split=5, k=k)
    test_dataset = FeaturesUniTS(dataset_name, 5, 1, "test", k_split=5, k=k)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get model
    args = {
        "is_training": 1,
        "model_id": "UniTS_sup_",
        "model": "UniTS",
        "task_name": "anomaly_detection",
        "prompt_num": 10,
        "patch_len": 16,
        "stride": 16,
        "e_layers": 3,
        "d_model": 128,
        "n_heads": 8,
        "des": "Exp",
        "itr": 1,
        "lradj": "finetune_anl",
        "weight_decay": 5e-6,
        "train_epochs": 20,
        "batch_size": 64,
        "acc_it": 32,
        "dropout": 0,
        "debug": "online",
        "dataset_name": dataset_name,
        "project_name": f"{dataset_name}_units_pretrained_d128_kfold",
        "pretrained_weight": "src/models/UniTS/pretrain_checkpoint.pth",
        "clip_grad": 100,
        "task_data_config_path": f"{dataset_name}.yaml",
        "anomaly_ratio": 7,
        "confounding": False,
        "data": "All",
        "features": "M",
        "checkpoints": "src/models/UniTS/checkpoints/",
        "freq": "h",
        "win_size": 5,
        "k": k,
        "step": 1,
        "subsample_pct": None,
        "num_workers": 0,
        "learning_rate": 5e-5,
        "layer_decay": None,
        "memory_check": False,
        "prompt_tune_epoch": 0,
    }

    # Dict to object
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args)

    #mean
    # task_data_config = {
    #     'WESAD': {
    #         'dataset': 'WESAD', 'data': 'WESAD', 'embed': 'timeF', 'label_len': 0,
    #         'pred_len': 0, 'features': 'M', 'enc_in': 6, 'context_size': 3,
    #         'task_name': 'anomaly_detection', 'max_batch': 64
    #     }
    # }

    #weighted
    task_data_config = {
        'WESAD': {
            'dataset': 'WESAD', 'data': 'WESAD', 'embed': 'timeF', 'label_len': 0,
            'pred_len': 0, 'features': 'M', 'enc_in': 6, 'context_size': 3, 'feature_weights': [0.25, 0.25, 0.15, 0.15, 0.15, 0.05], 
            'task_name': 'anomaly_detection', 'max_batch': 64
        }
    }

    # Load the model
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    exp = Exp_All_Task(args, infer=True, task_data_config=task_data_config)

    # Train the model
    setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des, 0)
    exp.train(setting, train_ddp=False)

    #weighted
    config = {
        "feature_weights": [0.25, 0.25, 0.15, 0.15, 0.15, 0.05]
    }

    # Test the model
    f_score, precision, recall, accuracy, gt, pred, reconstructed = exp.test_anomaly_detection(
        {}, {}, (train_loader, test_loader), "", 0, config=config, ar=3
    )
    print(f"Precision: {precision}, Recall: {recall}, F1: {f_score}, Accuracy: {accuracy}")
 
    # Get test dataset as a numpy array
    data = []
    for x, y in test_loader:
        new_x = x.numpy()
        for i in range(new_x.shape[0]): # For each element in the batch
            new_x[i] = train_dataset.scaler.inverse_transform(new_x[i])
        data.append(x) # (batch_size, w_size, n_features)


    # Get the reconstructed data
    rec = []
    for i in range(len(reconstructed)):
        new_x = train_dataset.scaler.inverse_transform(reconstructed[i])
        rec.append(new_x)

    # Aggregate the data
    data = np.concatenate(data, axis=0) # (n_samples, w_size, n_features)
    rec = np.concatenate(rec, axis=0) # (n_samples, w_size, n_features)

    # Get the prediction by averaging the prediction of the sliding windows
    window_size = 5
    avg_pred = []

    # For each sample in the prediction
    for i in range(0, len(pred), window_size):
        # Get the first predictions for that sample from the sliding windows
        other_pred = []
        min_idx = max(0, i-window_size*window_size) - 1
        for j in range(i, min_idx, -window_size-1):
            other_pred.append(pred[j])
        p = np.mean(other_pred, axis=0)
        p = np.where(p > majority_threshold, 1, 0)
        avg_pred.append(p)

    # Flatten the data
    data = data.reshape(-1, data.shape[-1]) # (n_samples * w_size, n_features)
    rec = rec.reshape(-1, rec.shape[-1]) # (n_samples * w_size, n_features)

    # Get the ground truth and prediction
    avg_pred = np.array(avg_pred) # (n_samples, n_features)
    gt = gt[:-window_size+1:window_size] # (n_samples, n_features)
    data = data[:-window_size+1:window_size] # (n_samples, n_features)
    rec = rec[:-window_size+1:window_size] # (n_samples, n_features) 

    return gt, avg_pred, data, rec

# Function to plot anomaly reconstruction errors and feature contributions
def plot_anomaly(dataset_name: str, all_samples: bool = False, majority_threshold: float = 0):

    # Get the first anomaly
    if all_samples:
        n_samples = 2000
        anomaly_idx = 0
    else:
        n_samples = 500
        anomaly_idx = 600

    print(f"Plotting anomaly multimodal from sample {anomaly_idx} to {anomaly_idx + n_samples}")

    # Create the directory to save the plots
    prefix = "all_" if all_samples else "part_"
    path = os.path.join(base_path, f"plots/{k}")
    path = os.path.join(path)
    if majority_threshold > 0:
        path = os.path.join(path, f"majority_{majority_threshold}")
    os.makedirs(path, exist_ok=True)

    # Get the model prediciton
    gt, pred, data, outputs = get_model_predictions(dataset_name, majority_threshold)
    min_idx = max(anomaly_idx, 0)
    max_idx = min(min_idx + n_samples, len(gt))
    gt = gt[min_idx:max_idx]
    pred = pred[min_idx:max_idx]
    data = data[min_idx:max_idx]
    outputs = outputs[min_idx:max_idx]

    # Konfigurasi
    feature_names = ['HR_BVP', 'HRV_BVP', 'SCR_count', 'SCR_avg_amplitude', 'SCL_mean', 'TEMP_mean']
    feature_colors = ["#5dade2", "#d3d920", "#f89939", "#2ecc71", "#f06e57", "#a569bd"]  # Satu warna per fitur

    # Hitung error rekonstruksi per fitur
    error = np.abs(data - outputs)
    tot_error = np.sum(error, axis=1)
    tot_error_norm = tot_error / np.max(tot_error)  # untuk alpha transparansi
    contrib_error = error / tot_error[:, np.newaxis]  # kontribusi relatif

    x_ticks = np.arange(min_idx, max_idx)

    # Plot sinyal masing-masing fitur
    for i in range(data.shape[1]):
        plt.figure(figsize=(10, 3))
        plt.plot(x_ticks, data[:, i], color='#3499cd', label=feature_names[i])
        plt.ylabel(feature_names[i])
        plt.xlabel("Time (s)")

        # Plot the anomaly prediction as red area
        min_h = data[:, i].min()
        max_h = data[:, i].max()
        h_margin = 0.1 * (max_h - min_h)
        plt.ylim(min_h - h_margin, max_h + h_margin)
        plt.fill_between(x_ticks, min_h, max_h+h_margin, where=pred == 1, color='red', alpha=0.3, linewidth=0.0, label="Predicted Anomaly")
        plt.fill_between(x_ticks, min_h-h_margin, min_h, where=gt == 1, color='green', alpha=0.3, linewidth=0.0, label="Ground Truth Anomaly")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(path + f"/{prefix}anomaly_{feature_names[i]}.svg")
        plt.close()

    # Plot kontribusi error (stacked bar per fitur) dengan alpha = tot_error_norm
    plt.figure(figsize=(10, 3))
    for i in range(len(x_ticks)):
        alpha = min(1, tot_error_norm[i])
        bottom = 0
        for j in range(len(feature_names)):
            plt.bar(x_ticks[i], contrib_error[i, j], bottom=bottom, color=feature_colors[j], alpha=alpha)
            bottom += contrib_error[i, j]
    plt.ylabel("Contribution Probability")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(path + f"/{prefix}anomaly_error_contrib.svg")
    plt.close()

    # Plot error rekonstruksi per fitur
    for i in range(data.shape[1]):
        plt.figure(figsize=(10, 3))
        plt.bar(x_ticks, error[:, i], color=feature_colors[i], alpha=1)
        plt.ylabel(f"Rec Error {feature_names[i]}")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(path + f"/{prefix}anomaly_{feature_names[i]}_error.svg")
        plt.close()


def plot_density(dataset_name: str):
    # Get the dataset
    dataset = Features(dataset_name, "test", k_split=5, k=k, scaler=lambda x: x)
    path = os.path.join(base_path, "plots")

    # Get anomaly samples
    anomaly_samples = dataset.split[dataset.labels == 1]
    non_anomaly_samples = dataset.split[dataset.labels == 0]

    # Window size 5
    w_size = 5
    lanomaly_samples = [
        np.mean(anomaly_samples[i:i+w_size], axis=0)
        for i in range(anomaly_samples.shape[0] - w_size)
    ]
    lnon_anomaly_samples = [
        np.mean(non_anomaly_samples[i:i+w_size], axis=0)
        for i in range(non_anomaly_samples.shape[0] - w_size)
    ]
    anomaly_samples = np.array(lanomaly_samples)
    non_anomaly_samples = np.array(lnon_anomaly_samples)

    feature_names = ['HR_BVP', 'HRV_BVP', 'SCR_count', 'SCR_avg_amplitude', 'SCL_mean', 'TEMP_mean']
    plt.rcParams.update({'font.size': 22})
    for i, name in enumerate(feature_names):
        plt.figure(figsize=(10, 6.5))
        sns.kdeplot(anomaly_samples[:, i], label="Anomalous", fill=True, color="red")
        sns.kdeplot(non_anomaly_samples[:, i], label="Non-anomalous", fill=True, color="blue")
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(path, f"{name.lower()}_density.pdf"))
        plt.close()

def main():
    dataset_name = "WESAD"
    plot_anomaly(dataset_name, all_samples=True)
    plot_anomaly(dataset_name, all_samples=False)
    plot_density(dataset_name)

if __name__ == "__main__":
    main()
