import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats

VERSION = "v1"
WANDB_USER = "erhaemael-politeknik-negeri-bandung"
DATASETS = ["wesad"]
METRICs = {
    "units": ["f1_score_ar3"]
}
AUCs = {
    "units": ["AUC_ROC", "AUC_PR"]
}
RUNS = {
    "units": 25
}

api = wandb.Api()

def process_project(project_name: str, model: str = "units", csvs: list = [], auc: bool = False, ignore_version: bool = False):
    if not any([dataset in project_name for dataset in DATASETS]):
        return
    if VERSION not in project_name and not ignore_version:
        return

    for k in RUNS.keys():
        if k.lower() in project_name:
            model = k
            if model == "contextual":
                break

    runs = api.runs(f"{WANDB_USER}/{project_name}")
    metrics = METRICs[model]
    aucs = AUCs[model] if auc else []

    print(f"Processing {project_name} with model {model}, metrics: {metrics}, aucs: {aucs}")

    if len(runs) != RUNS[model]:
        print(f"Skipping {project_name} because of missing runs ({len(runs)}/{RUNS[model]})")
        return

    groups: dict[tuple[float, str], list[pd.DataFrame]] = {}

    for run in tqdm(runs, desc=f"Processing {project_name}"):
        tags = run.tags

        datas = [run.scan_history(keys=[metric]) for metric in metrics]
        datas += [run.scan_history(keys=aucs)]
        datas = [pd.DataFrame(data) for data in datas]
        data = pd.concat(datas, axis=1)

        lr_tag = [tag for tag in tags if tag.startswith("lr")]
        if not lr_tag:
            continue
        lr_str = lr_tag[0]
        try:
            lr_value = float(lr_str.replace("lr", ""))
        except ValueError:
            print(f"Skipping run with invalid lr tag: {lr_str}")
            continue

        source = "BVP"

        if (lr_value, source) not in groups:
            groups[(lr_value, source)] = []
        groups[(lr_value, source)].append(data)

    avg_f1s: dict[tuple[float, str], float] = {}
    std_f1s: dict[tuple[float, str], float] = {}
    l_kf1s: dict[tuple[float, str], list[float]] = {}
    avg_aucrocs: dict[tuple[float, str], float] = {}
    avg_aucprs: dict[tuple[float, str], float] = {}

    for group, data_runs in groups.items():
        best_f1s = []
        best_aucrocs = []
        best_aucprs = []
        for data in data_runs:
            best_f1 = 0
            best_idx = 0
            for metric in metrics:
                f1s = data.loc[:, metric]
                f1s = f1s.astype(float).fillna(0)
                idx = f1s.idxmax()
                f1 = float(data.loc[idx, metric])
                if f1 > best_f1:
                    best_f1 = f1
                    best_idx = idx
            best_f1s.append(best_f1)
            if len(aucs) > 0:
                best_aucroc = float(data.loc[best_idx, aucs[0]])
                best_aucpr = float(data.loc[best_idx, aucs[1]])
                best_aucrocs.append(best_aucroc)
                best_aucprs.append(best_aucpr)

        avg_f1s[group] = np.mean(best_f1s)
        std_f1s[group] = np.std(best_f1s)
        l_kf1s[group] = best_f1s
        avg_aucrocs[group] = np.mean(best_aucrocs)
        avg_aucprs[group] = np.mean(best_aucprs)

    print("\n=== F1 Scores per Learning Rate ===")
    for (lr, src), f1 in avg_f1s.items():
        print(f"lr={lr:.0e}, mean_f1={f1:.4f}, std={std_f1s[(lr, src)]:.4f}")
        # Tampilkan f1 per fold juga
        f1_folds = l_kf1s[(lr, "BVP")]
        for i, f1 in enumerate(f1_folds):
            print(f"  Fold {i}: {f1:.4f}")

    best_metrics: dict[str, list[float]] = {}
    best_l_kf1s: dict[str, list[float]] = {}
    best_f1 = 0
    best_lr = 0
    best_f1_std = 0
    best_aucroc = 0
    best_aucpr = 0

    for (lr, src), f1 in avg_f1s.items():
        if f1 > best_f1:
            best_f1 = f1
            best_lr = lr
            best_f1_std = std_f1s[(lr, src)]
            best_l_kf1s["BVP"] = l_kf1s[(lr, src)]
            if len(aucs) > 0:
                best_aucroc = avg_aucrocs[(lr, src)]
                best_aucpr = avg_aucprs[(lr, src)]

    best_metrics["BVP"] = [best_f1, best_aucroc, best_aucpr, best_f1_std]

    print(f"\nProject: {project_name} - Best Scores: {best_metrics} with lr={best_lr:.0e}")
    print(f"Project: {project_name} - Best l_kf1s: {best_l_kf1s}")

    if len(csvs) != len(DATASETS):
        print("ERROR: csvs should have the same length as DATASETS")
    else:
        for i in range(len(DATASETS)):
            if DATASETS[i] in project_name:
                best_f1, best_aucroc, best_aucpr, std_f1 = best_metrics["BVP"]
                csvs[i].write(f"{project_name},{best_f1},{best_aucroc},{best_aucpr},{std_f1},{len(runs)},{metrics[0]},lr={best_lr:.0e}\n")
                break

    return best_metrics, best_l_kf1s

def main():
    csv_paths = [f"{dataset}_results.csv" for dataset in DATASETS]
    csv_files = []

    # Buka file CSV dan tulis header jika belum ada
    for path in csv_paths:
        write_header = not os.path.exists(path)
        f = open(path, "w")  # selalu overwrite
        f.write("project_name,best_f1,best_aucroc,best_aucpr,std_f1,num_runs,metric,lr_value\n")
        csv_files.append(f)

    projects = api.projects(f"{WANDB_USER}")
    for project in tqdm(projects, desc="Processing projects"):
        try:
            project_name = project.name.lower()
            process_project(project_name, csvs=csv_files, auc=True)
        except Exception as e:
            print(f"Error in processing project {project.name}: {e}")

    for f in csv_files:
        f.close()

if __name__ == "__main__":
    main()



#   def test_anomaly_detection(self, setting, test_data, test_loader_set, data_task_name, task_id, ar=None):
#         train_loader, test_loader = test_loader_set
#         attens_energy = []
#         anomaly_criterion = nn.MSELoss(reduce=False)

#         self.model.eval()

#         # (1) Calculate reconstruction error on train set
#         with torch.no_grad():
#             for i, (batch_x, batch_y) in enumerate(train_loader):
#                 batch_x = batch_x.float().to(self.device_id)
#                 outputs = self.model(batch_x, None, None, None, task_id=task_id, task_name='anomaly_detection')
#                 score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
#                 score = score.detach().cpu()
#                 attens_energy.append(score)

#             if get_world_size() > 1:
#                 attens_energy = gather_tensors_from_all_gpus(attens_energy, self.device_id, to_numpy=True)
#         train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

#         # (2) Calculate reconstruction error on test set
#         attens_energy = []
#         test_labels = []
#         reconstructed = []
#         for i, (batch_x, batch_y) in enumerate(test_loader):
#             batch_x = batch_x.float().to(self.device_id)
#             outputs = self.model(batch_x, None, None, None, task_id=task_id, task_name='anomaly_detection')
#             reconstructed.append(outputs.detach().cpu().numpy())

#             score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
#             score = score.detach().cpu()
#             attens_energy.append(score)
#             test_labels.append(batch_y)

#         if get_world_size() > 1:
#             attens_energy = gather_tensors_from_all_gpus(attens_energy, self.device_id, to_numpy=True)
#         test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#         test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
#         gt = test_labels.astype(int)

#         combined_energy = np.concatenate([train_energy, test_energy], axis=0)

#         # (3) Compute 3rd and 97th percentile as thresholds
#         lower_threshold = np.percentile(combined_energy, 3)
#         upper_threshold = np.percentile(combined_energy, 97)

#         # (4) Predict anomalies: 1 if outlier, 0 if normal
#         pred = ((test_energy < lower_threshold) | (test_energy > upper_threshold)).astype(int)

#         # (5) Apply adjustment if needed
#         gt, pred = adjustment(gt, pred)

#         tp = np.sum((gt == 1) & (pred == 1))
#         tn = np.sum((gt == 0) & (pred == 0))
#         fp = np.sum((gt == 0) & (pred == 1))
#         fn = np.sum((gt == 1) & (pred == 0))

#         precision = tp / (tp + fp + 1e-8)
#         recall = tp / (tp + fn + 1e-8)
#         f_score = 2 * precision * recall / (precision + recall + 1e-8)
#         accuracy = (tp + tn) / (tp + tn + fp + fn)
#         MIoU = 0.5 * (tp / (tp + fp + fn + 1e-8) + tn / (tn + fn + fp + 1e-8))

#         auc_roc = roc_auc_score(gt, test_energy)
#         auc_pr = average_precision_score(gt, test_energy)

#         wandb.log({
#             "AUC_ROC": auc_roc,
#             "AUC_PR": auc_pr,
#             "f1_score": f_score,
#             "precision": precision,
#             "recall": recall,
#             "accuracy": accuracy,
#             "MIoU": MIoU,
#             "lower_threshold": lower_threshold,
#             "upper_threshold": upper_threshold,
#             "TP": tp,
#             "TN": tn,
#             "FP": fp,
#             "FN": fn
#         })

#         reconstructed = np.concatenate(reconstructed, axis=0)
#         return f_score, precision, recall, accuracy


# METRICs = {
#     "units": ["f1_score"]
# }