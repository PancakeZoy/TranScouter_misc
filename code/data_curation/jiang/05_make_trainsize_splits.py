import scanpy as sc
import pandas as pd
import pickle
import numpy as np
import random
import torch
import argparse
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser(description="Run TranScouter")
parser.add_argument("i", type=int)
args = parser.parse_args()


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds()
ROOT = Path(__file__).resolve().parents[3]
adata_path = ROOT / "data/jiang/processed/aggregate_sum_hvg.h5ad"
adata = sc.read_h5ad(adata_path)
adata.obs["cov_pert"] = adata.obs[["covariate", "perturbation"]].agg("_".join, axis=1)

all_conds = sorted(adata.obs.cov_pert.unique())
split_df = pd.DataFrame(all_conds, columns=["cov_pert"])
split_df[["covariate", "perturbation"]] = split_df["cov_pert"].str.split("_", expand=True)  # fmt: skip
all_covs = sorted(split_df["covariate"].unique())

n_test = 3 * args.i
n_val = 2 * args.i
n_split = 10
split_dict = {}
for split in range(n_split):
    random.seed(split)
    test_val_cov = random.sample(all_covs, n_test + n_val)
    test_cov, val_cov = test_val_cov[:n_test], test_val_cov[n_test: (n_test + n_val)]  # fmt: skip
    train_cov = list(np.setdiff1d(all_covs, test_val_cov))

    test_df = split_df[split_df.covariate.isin(test_cov)]
    val_df = split_df[split_df.covariate.isin(val_cov)]
    train_df = split_df[split_df.covariate.isin(train_cov)]

    test_perts = test_df.perturbation.unique()
    val_perts = val_df.perturbation.unique()
    test_val_perts = sorted(set(test_perts).union(set(val_perts)))
    test_val_perts.remove("control")
    n_masked_perts = len(test_val_perts) // n_split

    random.seed(split)
    masked_perts = random.sample(test_val_perts, n_masked_perts)
    masked_train_df = train_df[train_df.perturbation.isin(masked_perts)]
    masked_train_val = masked_train_df.sample(frac=0.4, random_state=split)
    masked_train_test = masked_train_df.drop(masked_train_val.index)
    val_df = pd.concat([val_df, masked_train_val])
    test_df = pd.concat([test_df, masked_train_test])
    train_df = train_df[~train_df.perturbation.isin(masked_perts)]

    test_conds = test_df[test_df.perturbation != "control"]["cov_pert"].tolist()
    val_conds = val_df[val_df.perturbation != "control"]["cov_pert"].tolist()
    train_conds = train_df[train_df.perturbation != "control"]["cov_pert"].tolist()

    split_dict[split] = {
        "test": test_conds,
        "val": val_conds,
        "train": train_conds,
        "masked_perts": masked_perts,
        "test_cov": test_cov,
    }

output_path = ROOT / f"data/jiang/splits/trainsize/TrainSize_SplitDict_{args.i}_10Fold.pkl"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(split_dict, f)


with open(output_path, "rb") as f:
    val = pickle.load(f)
    print([len(val[i]["test"]) for i in range(n_split)])
