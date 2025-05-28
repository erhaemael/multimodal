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
    # Check if project belongs to allowed datasets and version
    if not any([dataset in project_name for dataset in DATASETS]):
        return
    if VERSION not in project_name and not ignore_version:
        return

    # Identify the model type used in the project name
    for k in RUNS.keys():
        if k.lower() in project_name:
            model = k
            if model == "contextual":
                break

    # Retrieve all runs from the project
    runs = api.runs(f"{WANDB_USER}/{project_name}")
    metrics = METRICs[model]
    aucs = AUCs[model] if auc else []

    print(f"Processing {project_name} with model {model}, metrics: {metrics}, aucs: {aucs}")

    # Skip project if number of runs doesn't match the expected count
    if len(runs) != RUNS[model]:
        print(f"Skipping {project_name} because of missing runs ({len(runs)}/{RUNS[model]})")
        return

    # Group results by learning rate and signal source (hardcoded as "BVP")
    groups: dict[tuple[float, str], list[pd.DataFrame]] = {}

    for run in tqdm(runs, desc=f"Processing {project_name}"):
        tags = run.tags

        # Extract metric history for each run
        datas = [run.scan_history(keys=[metric]) for metric in metrics]
        datas += [run.scan_history(keys=aucs)]
        datas = [pd.DataFrame(data) for data in datas]
        data = pd.concat(datas, axis=1)

        # Extract learning rate tag
        lr_tag = [tag for tag in tags if tag.startswith("lr")]
        if not lr_tag:
            continue
        lr_str = lr_tag[0]
        try:
            lr_value = float(lr_str.replace("lr", ""))
        except ValueError:
            print(f"Skipping run with invalid lr tag: {lr_str}")
            continue

        source = "BVP"  # Currently hardcoded source

        if (lr_value, source) not in groups:
            groups[(lr_value, source)] = []
        groups[(lr_value, source)].append(data)

    # Dictionaries to store summary statistics
    avg_f1s = {}
    std_f1s = {}
    l_kf1s = {}
    avg_aucrocs = {}
    avg_aucprs = {}

    # Compute best F1-scores and AUCs for each group
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

        # Store mean and std of best F1s, and best AUCs if available
        avg_f1s[group] = np.mean(best_f1s)
        std_f1s[group] = np.std(best_f1s)
        l_kf1s[group] = best_f1s
        avg_aucrocs[group] = np.mean(best_aucrocs)
        avg_aucprs[group] = np.mean(best_aucprs)

    # Print average and per-fold F1 scores
    print("\n=== F1 Scores per Learning Rate ===")
    for (lr, src), f1 in avg_f1s.items():
        print(f"lr={lr:.0e}, mean_f1={f1:.4f}, std={std_f1s[(lr, src)]:.4f}")
        f1_folds = l_kf1s[(lr, "BVP")]
        for i, f1 in enumerate(f1_folds):
            print(f"  Fold {i}: {f1:.4f}")

    # Determine the best performing configuration
    best_metrics = {}
    best_l_kf1s = {}
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

    # Print the best metric summary
    print(f"\nProject: {project_name} - Best Scores: {best_metrics} with lr={best_lr:.0e}")
    print(f"Project: {project_name} - Best l_kf1s: {best_l_kf1s}")

    # Save results to CSV
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
    # Define CSV paths and open files
    csv_paths = [f"{dataset}_results.csv" for dataset in DATASETS]
    csv_files = []

    # Open CSV files and write header
    for path in csv_paths:
        write_header = not os.path.exists(path)
        f = open(path, "w") 
        f.write("project_name,best_f1,best_aucroc,best_aucpr,std_f1,num_runs,metric,lr_value\n")
        csv_files.append(f)

    # Process all projects under the W&B user
    projects = api.projects(f"{WANDB_USER}")
    for project in tqdm(projects, desc="Processing projects"):
        try:
            project_name = project.name.lower()
            process_project(project_name, csvs=csv_files, auc=True)
        except Exception as e:
            print(f"Error in processing project {project.name}: {e}")

    # Close CSV files
    for f in csv_files:
        f.close()

if __name__ == "__main__":
    main()