import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import torch
from torch.utils.data import DataLoader
import wandb
import random

# Constanta
feature_names = ['HR_BVP', 'HRV_BVP', 'SCR_count', 'SCR_avg_amplitude', 'SCL_mean', 'TEMP_mean']
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

def get_model_predictions(dataset_name: str, source: str, majority_threshold: float = 0):
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
        "learning_rate": 5e-4,
        "layer_decay": None,
        "memory_check": False,
        "prompt_tune_epoch": 0,
    }

    # Dict to object
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args)

    task_data_config = {
        'WESAD': {
            'dataset': 'WESAD', 'data': 'WESAD', 'embed': 'timeF', 'label_len': 0,
            'pred_len': 0, 'features': 'M', 'enc_in': 6, 'context_size': 6, 'feature_weights': [0.25, 0.25, 0.15, 0.15, 0.15, 0.05], 
            'task_name': 'anomaly_detection', 'max_batch': 64
        }
    }

    # Load the model
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    exp = Exp_All_Task(args, infer=True, task_data_config=task_data_config)

    setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.d_model, args.e_layers, args.des, 0)
    exp.train(setting, train_ddp=False)

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
        if hasattr(train_dataset, 'scaler') and train_dataset.scaler:
            for i in range(new_x.shape[0]): # For each element in the batch
                new_x[i] = train_dataset.scaler.inverse_transform(new_x[i])
        data.append(torch.tensor(new_x))  # (batch_size, w_size, n_features)

    # Get the reconstructed data
    rec = []
    for i in range(len(reconstructed)):
        if hasattr(train_dataset, 'scaler') and train_dataset.scaler:
            new_x = train_dataset.scaler.inverse_transform(reconstructed[i])
        rec.append(torch.tensor(new_x))

    # Aggregate the data
    data = torch.cat(data, dim=0).numpy().reshape(-1, len(feature_names))
    rec = torch.cat(rec, dim=0).numpy().reshape(-1, len(feature_names))

    # For each sample in the prediction
    window_size = 5
    avg_pred = []
    for i in range(0, len(pred), window_size):
        # Get the first predictions for that sample from the sliding windows
        other_pred = []
        min_idx = max(0, i - window_size*window_size) - 1
        for j in range(i, min_idx, -window_size-1):
            other_pred.append(pred[j])
        p = np.mean(other_pred, axis=0)
        p = np.where(p > majority_threshold, 1, 0)
        avg_pred.append(p)

    # Get the ground truth and prediction
    avg_pred = np.array(avg_pred)
    gt = gt[:-window_size+1:window_size]
    data = data[:-window_size+1:window_size]
    rec = rec[:-window_size+1:window_size]

    return gt, avg_pred, data, rec

# Function to plot anomaly reconstruction errors and feature contributions
def plot_anomaly(dataset_name: str, source: str, all_samples: bool = False, majority_threshold: float = 0):

    # Get the first anomaly
    if all_samples:
        n_samples = 2000
        anomaly_idx = 0
    else:
        n_samples = 500
        anomaly_idx = 600

    print(f"Plotting anomaly for {source} from sample {anomaly_idx} to {anomaly_idx + n_samples}")

    # Create the directory to save the plots
    prefix = ("all_" if all_samples else "part_") + f"{source}_"
    path = os.path.join(base_path, f"plots/{k}/{source}")
    if majority_threshold > 0:
        path = os.path.join(path, f"majority_{majority_threshold}")
    os.makedirs(path, exist_ok=True)

    # Get the model prediciton
    gt, pred, data, outputs = get_model_predictions(dataset_name, source, majority_threshold)
    gt = gt[anomaly_idx:anomaly_idx + n_samples]
    pred = pred[anomaly_idx:anomaly_idx + n_samples]
    data = data[anomaly_idx:anomaly_idx + n_samples]
    outputs = outputs[anomaly_idx:anomaly_idx + n_samples]

    # Compute reconstruction error
    errors = np.abs(data - outputs)
    tot_error = np.sum(errors, axis=1) + 1e-8
    perc_error = errors / tot_error[:, None]

    x_ticks = np.arange(len(gt))
    colors = ["#3499cd", "#f89939", "#2ecc71", "#f06e57", "#a569bd", "#5dade2"]

    # Helper to save barplot
    def save_barplot(values, title, filename, color):
        plt.figure(figsize=(10, 3))
        plt.bar(x_ticks, values, color=color, alpha=1)
        plt.ylabel(title)
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(path, filename))
        plt.close()

    # save_barplot(tot_error / np.max(tot_error), "Total Rec Error (norm)", f"{prefix}anomaly_total_error.svg", "#000000")

    # Stacked barplot of relative contribution per feature
    plt.figure(figsize=(10, 4))
    bottom = np.zeros(len(perc_error))
    for i in range(len(feature_names)):
        plt.bar(x_ticks, perc_error[:, i], bottom=bottom, label=feature_names[i], color=colors[i % len(colors)], alpha=0.85)
        bottom += perc_error[:, i]
    plt.ylabel("Relative Contribution")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{prefix}anomaly_feature_contribution.svg"))
    plt.close()

    # Save individual feature reconstruction error plots
    for i, name in enumerate(feature_names):
        save_barplot(errors[:, i] / np.max(errors[:, i]), f"Rec Error {name}", f"{prefix}anomaly_{name.lower()}_error.svg", colors[i % len(colors)])

    # Line plots per feature
    plot_feature_lines(data, pred, x_ticks, path, prefix=prefix, feature_names=feature_names)

# Function to plot feature lines with anomaly prediction areas
def plot_feature_lines(data, pred, x_ticks, path, prefix="", feature_names=None):
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(data.shape[1])]

    os.makedirs(path, exist_ok=True)

    for i in range(data.shape[1]):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(x_ticks, data[:, i], color='#3499cd')
        ax.set_ylabel(feature_names[i])
        ax.set_xlabel("Time (s)")
        min_val, max_val = np.min(data[:, i]), np.max(data[:, i])
        margin = 0.1 * (max_val - min_val)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.fill_between(x_ticks, min_val - margin, max_val + margin, where=pred == 1, color='red', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(path, f"{prefix}anomaly_line_{feature_names[i]}.svg")
        plt.savefig(save_path)
        plt.close()

def plot_density(dataset_name: str, source: str):
    # Get the dataset
    dataset = Features(dataset_name, "test", k_split=5, k=k, scaler=lambda x: x)
    path = os.path.join(base_path, "plots")


    # Get anomaly samples
    anomaly_samples = dataset.split[dataset.labels == 1]
    non_anomaly_samples = dataset.split[dataset.labels == 0]

    # Window size 5
    w_size = 5

    anomaly_samples = np.array([np.mean(anomaly_samples[i:i+w_size], axis=0) for i in range(len(anomaly_samples) - w_size)])
    non_anomaly_samples = np.array([np.mean(non_anomaly_samples[i:i+w_size], axis=0) for i in range(len(non_anomaly_samples) - w_size)])

    plt.rcParams.update({'font.size': 22})
    for i, name in enumerate(feature_names):
        plt.figure(figsize=(10, 6.5))
        sns.kdeplot(anomaly_samples[:, i], label="Anomalous", fill=True, color="red")
        sns.kdeplot(non_anomaly_samples[:, i], label="Non-anomalous", fill=True, color="blue")
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(path, f"{name.lower()}_density_{source}.pdf"))
        plt.close()

def main():
    dataset_name = "WESAD"
    source = "BVP"
    plot_anomaly(dataset_name, source, all_samples=True)
    plot_anomaly(dataset_name, source, all_samples=False)
    plot_density(dataset_name, source)

if __name__ == "__main__":
    main()
