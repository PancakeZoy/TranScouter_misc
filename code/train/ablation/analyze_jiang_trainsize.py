import pickle
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import mean_squared_error as mse

# fmt: off

# Set up argument parser
parser = argparse.ArgumentParser(description="Run TranScouter")
parser.add_argument("i", type=int, help="Train-size split index: 1, 2, 3, 4, or 5")  # fmt: skip
args = parser.parse_args()


ROOT = Path(__file__).resolve().parents[3]
ADATA_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
TOP_DEGS_NAMES_PATH = ROOT / "data" / "jiang" / "processed" / "top_degs_names.pkl"
TOP_DEGS_IDX_PATH = ROOT / "data" / "jiang" / "processed" / "top_degs_idx.pkl"
SPLIT_DIR = ROOT / "data" / "jiang" / "splits" / "trainsize"
OUT_DIR = ROOT / "results" / "ablation" / "trainsize"
TRAIN_SIZE = 30 - 5 * args.i


def cond_unseen(cond, test_cov, masked_perts):
    cov_unseen = cond.split("_")[0] in test_cov
    pert_unseen = cond.split("_")[1] in masked_perts
    return cov_unseen and pert_unseen


def dc(prediction, control, ground_truth):
    return sum(np.sign((prediction - control) * (ground_truth - control)) > 0) / len(prediction)


###################### Load Data ##########################
adata = sc.read_h5ad(ADATA_PATH)

with open(TOP_DEGS_NAMES_PATH, "rb") as f:
    top_degs_names = pickle.load(f)
with open(TOP_DEGS_IDX_PATH, "rb") as f:
    top_degs_idx = pickle.load(f)
with open(SPLIT_DIR / f"TrainSize_SplitDict_{args.i}_10Fold.pkl", "rb") as f:
    split_dict = pickle.load(f)
with open(OUT_DIR / f"Jiang_{TRAIN_SIZE}.pkl", "rb") as f:
    result = pickle.load(f)
    
###################### Evaluate ##########################
metric_cond_ls = []
for split in range(len(split_dict)):
    metric = {"MSE_TranScouter": {}, "DC_TranScouter": {}, "MSE_Ctrl": {}, 
            "MSE_PertAvg": {}, "DC_PertAvg": {}, "n_up": {}, "n_down": {}}
    transcouter_split = result[f"split_{split}"]["Prediction"]
    ctrl_split = result[f"split_{split}"]["Control"]
    true_split = result[f"split_{split}"]["True"]

    test_conds, masked_perts, test_cov = (
        split_dict[split]["test"],
        split_dict[split]["masked_perts"],
        split_dict[split]["test_cov"],
    )
    unseen_test_conds = [cond for cond in test_conds if cond_unseen(cond, test_cov, masked_perts)]

    for cond in sorted(unseen_test_conds):
        degs = top_degs_names[cond]
        cov, pert = cond.split("_")
        
        if len(degs) > 0: 
            transcouter = transcouter_split[transcouter_split.cov_pert == cond].drop(columns="cov_pert")[degs].mean(axis=0)
            ctrl = ctrl_split[ctrl_split.cov_pert == cond].drop(columns="cov_pert")[degs].mean(axis=0)
            true = true_split[true_split.cov_pert == cond].drop(columns="cov_pert")[degs].mean(axis=0)
            pert_avg = adata[(adata.obs.perturbation == pert) & (adata.obs.covariate != cov), degs].X.mean(axis=0)

            metric["MSE_TranScouter"][cond] = mse(true, transcouter)
            metric["DC_TranScouter"][cond] = dc(transcouter, ctrl, true)
            metric["MSE_Ctrl"][cond] = mse(true, ctrl)
            metric["MSE_PertAvg"][cond] = mse(true, pert_avg)
            metric["DC_PertAvg"][cond] = dc(pert_avg, ctrl, true)
            metric["n_up"][cond] = sum(true-ctrl > 0)
            metric["n_down"][cond] = sum(true-ctrl < 0)
            
    metric_cond = pd.DataFrame(metric).assign(split=split)
    metric_cond_ls.append(metric_cond)

metric_all = pd.concat(metric_cond_ls)
metric_all['condition'] = metric_all.index.str.split('_').str[0]
OUT_DIR.mkdir(parents=True, exist_ok=True)
metric_all.to_csv(OUT_DIR / f"Analysis_{TRAIN_SIZE}.csv")
