import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scikit_posthocs as sp
import scipy.stats as stats
import seaborn as sns


VERSION = "v1"
WANDB_USER = "erhaemael-politeknik-negeri-bandung"
DATASETS = ["wesad"]
# AVERAGED = False
VALID_LRS = { 
    "units": "lr0.0005"
}
METRICs = {
    "units": ["f1_score_pctl"]
}
AUCs = {
    "units": ["AUC_ROC", "AUC_PR"]
}
RUNS = {
    "units": 25
}

api = wandb.Api()

def process_project(project_name: str, model: str = "units", csvs: list = [], auc: bool = True, ignore_version: bool = False):
    if not any([dataset in project_name for dataset in DATASETS]):
        return
    if not ignore_version and not project_name.endswith(VERSION):
        return

    for k in RUNS.keys():
        if k.lower() in project_name:
            model = k

    runs = api.runs(f"{WANDB_USER}/{project_name}")
    metrics = METRICs[model]
    aucs = AUCs[model] if auc else []

    print(f"Processing {project_name} with model {model}, metrics: {metrics}, aucs: {aucs}")

    if len(runs) != RUNS[model]:
        print(f"Skipping {project_name} because of missing runs ({len(runs)}/{RUNS[model]})")
        return

    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for run in runs:
        tags = run.tags

        if "BVP" not in tags:
            continue

        lr = [tag for tag in tags if tag.startswith("lr")][0]
        source = "BVP"

        model_name = project_name.replace("_units_", "-units-").split("-")[1]
        if lr not in VALID_LRS[model_name]:
            continue

        datas = [run.scan_history(keys=[metric]) for metric in metrics]
        if len(aucs) > 0:
            datas += [run.scan_history(keys=aucs)]
        datas = [pd.DataFrame(data) for data in datas]
        data = pd.concat(datas, axis=1)

        if (lr, source) not in groups:
            groups[(lr, source)] = []
        groups[(lr, source)].append(data)
    
    print(f"Project: {project_name} - Groups: {groups.keys()}")

    # For each group, get the average f1 based on best f1 in each fold
    avg_f1s: dict[str, np.ndarray] = {}
    std_f1s: dict[str, np.ndarray] = {}
    l_kf1s: dict[str, list[float]] = {} # list of f1s for each fold
    avg_aucrocs: dict[str, np.ndarray] = {}
    avg_aucprs: dict[str, np.ndarray] = {}
    for group, data_runs in groups.items():
        best_f1s = []
        best_aucrocs = []
        best_aucprs = []
        # For each fold
        for data in data_runs:
            # Get the last non-zero f1
            best_f1 = 0
            best_idx = 0
            for metric in metrics:
                f1s = data.loc[:, metric]
                f1s = f1s.astype(float)
                # Get the last non-zero index
                idx = f1s.last_valid_index()
                if idx is None:
                    f1 = 0
                else:
                    f1 = float(data.loc[idx, metric])
                best_f1 = f1
                best_idx = idx
            # Add the metrics to the lists
            best_f1s.append(best_f1)

            if len(aucs) > 0:
                # Get the corresponding aucroc and aucpr
                best_aucroc = float(data.loc[best_idx, aucs[0]])
                best_aucpr = float(data.loc[best_idx, aucs[1]])
                best_aucrocs.append(best_aucroc)
                best_aucprs.append(best_aucpr)
        # Compute the average f1, aucroc, and aucpr over the folds
        avg_f1s[group] = np.mean(best_f1s)
        # Compute the standard error
        std_f1s[group] = np.std(best_f1s)
        l_kf1s[group] = best_f1s
        avg_aucrocs[group] = np.mean(best_aucrocs)
        avg_aucprs[group] = np.mean(best_aucprs)

    print(f"Project: {project_name} - Average F1:\n {avg_f1s}")
    with open("results.txt", "a") as f:
        f.write(str({project_name: avg_f1s})+"\n")

    # For each source, get the best score among the lrs
    best_metrics: dict[str, list[float]] = {}
    best_l_kf1s: dict[str, list[float]] = {}
    best_lr: dict[str, str] = {}

    source = "BVP"
    best_f1 = 0
    best_f1_std = 0
    best_aucroc = 0
    best_aucpr = 0

    for (lr, src), f1 in avg_f1s.items():
        if src == source and f1 > best_f1:
            best_f1 = f1
            best_f1_std = std_f1s[(lr, src)]
            best_l_kf1s[source] = l_kf1s[(lr, src)]
            best_aucroc = avg_aucrocs[(lr, src)]
            best_aucpr = avg_aucprs[(lr, src)]
            best_lr[source] = lr

    best_metrics[source] = [best_f1, best_aucroc, best_aucpr, best_f1_std]

    print(f"Project: {project_name} - Best Scores: {best_metrics}")
    print(f"Project: {project_name} - Best l_kf1s: {best_l_kf1s}")

    if len(csvs) != len(DATASETS):
        print("ERROR: csvs should have the same length as DATASETS")
    else:
        for i in range(len(DATASETS)):
            if DATASETS[i] in project_name:
                bvp_f1 = f"{best_metrics['BVP'][0]:.3f}"
                bvp_f1_std = f"{best_metrics['BVP'][3]:.3f}"
                bvp_aucroc = f"{best_metrics['BVP'][1]:.3f}"
                bvp_aucpr = f"{best_metrics['BVP'][2]:.3f}"
                bvp_metrics = f"{bvp_f1},{bvp_f1_std},{bvp_aucroc},{bvp_aucpr}"
                print(f"{project_name},{bvp_metrics},{len(runs)},{metrics[0]}\n")
                csvs[i].write(f"{project_name},{bvp_metrics},{len(runs)},{metrics[0]}\n")
                break
    return best_metrics, best_l_kf1s


def main():
    projects = api.projects(f"{WANDB_USER}")
    filenames = [f"results_{dataset}.csv" for dataset in DATASETS]
    csvs = [open(fname, "a") for fname in filenames]

    for csv in csvs:
        if csv.tell() == 0:
            csv.write("Project,BVP_F1,BVP_F1_STD,BVP_AUCROC,BVP_AUCPR,Runs,Metric\n")

    pds = []
    for fname in filenames:
        if os.path.exists(fname) and os.path.getsize(fname) > 0:
            try:
                df = pd.read_csv(fname)
                pds.append(df)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty file: {fname}")
        else:
            print(f"File {fname} belum ada atau kosong.")

    for project in tqdm(projects, desc="Processing projects"):
        try:
            project_name = project.name.lower()
            if any([project_name in pd["Project"].tolist() for pd in pds]):
                print(f"Skipping {project_name}. Already in the csv.")
                continue
            process_project(project_name, csvs=csvs)
            for csv in csvs:
                csv.flush()
        except Exception as e:
            print(f"Error in processing project {project.name}: {e}")

    for csv in csvs:
        csv.close()

if __name__ == "__main__":
    main()