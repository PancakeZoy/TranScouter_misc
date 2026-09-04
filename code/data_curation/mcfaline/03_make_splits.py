import scanpy as sc
import pandas as pd
import pickle
import numpy as np
import random
import torch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADATA_PATH = ROOT / "data" / "mcfaline" / "processed" / "mcfaline.h5ad"
SPLIT_PATH = ROOT / "data" / "mcfaline" / "splits" / "split_dict_5Fold.pkl"


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds()
adata = sc.read_h5ad(ADATA_PATH)
adata.obs["cov_pert"] = adata.obs[["covariate", "perturbation"]].agg("_".join, axis=1)

all_conds = sorted(adata.obs.cov_pert.unique())
split_df = pd.DataFrame(all_conds, columns=["cov_pert"])
split_df[["covariate", "perturbation"]] = split_df["cov_pert"].str.split("_", expand=True)  # fmt: skip
all_covs = sorted(split_df["covariate"].unique())
subsets = np.array_split(all_covs, 3)
subsets = [np.random.permutation(sub) for sub in subsets]
subsets = [x for t in zip(*subsets) for x in t]

n_test = 3
n_val = 2
n_train = len(all_covs) - (n_test + n_val)
n_split = len(all_covs) // n_test
test_sets = np.array_split(subsets, n_split)

split_dict = {}
for split in range(n_split):
    random.seed(split)
    test_cov = sorted(test_sets[split])
    train_val_cov = sorted(np.setdiff1d(all_covs, test_cov))
    train_cov = random.sample(train_val_cov, n_train)
    val_cov = sorted(np.setdiff1d(train_val_cov, train_cov))

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
    masked_train_val = masked_train_df.sample(
        frac=n_val / (n_test + n_val), random_state=split
    )
    masked_train_test = masked_train_df.drop(masked_train_val.index)
    val_df = pd.concat([val_df, masked_train_val])
    test_df = pd.concat([test_df, masked_train_test])
    train_df = train_df[~train_df.perturbation.isin(masked_perts)]
    # make validation set only contains seen0 situation
    val_df = val_df[(~val_df.covariate.isin(train_cov)) & (val_df.perturbation.isin(masked_perts))]  # fmt: skip

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

SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SPLIT_PATH, "wb") as f:
    pickle.dump(split_dict, f)
