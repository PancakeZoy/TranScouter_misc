import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import pickle
import gc
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "mcfaline" / "splits" / "split_dict_5Fold.pkl"
EXTRA_PATH = ROOT / "data" / "mcfaline" / "processed" / "mcfaline_extra.h5ad"
HVG_PATH = ROOT / "data" / "mcfaline" / "processed" / "mcfaline.h5ad"
OUT_PATH = ROOT / "results" / "baselines" / "correlation" / "mcfaline" / "corr_nonctrl.pkl"

# fmt: off
# See the code at https://github.com/Chen-Li-17/CellPB for more details
with open(SPLIT_PATH, "rb") as f:
    split_dict = pickle.load(f)

adata_extra = sc.read(EXTRA_PATH)
adata_hvg = sc.read(HVG_PATH)
hvgs = list(adata_hvg.var_names)
adata = ad.concat([adata_hvg, adata_extra], axis=1)
adata.X = adata.X.toarray()

adata.obs = adata_hvg.obs.copy()
adata.obs['cov_pert'] = adata.obs['covariate'].str.cat(adata.obs['perturbation'], sep='_')
var_names_all = list(adata.var_names)
del adata_extra, adata_hvg
gc.collect()

result = {}
for i in range(5):
    print(f"Processing split {i}")
    split = split_dict[i]
    test_conds, train_conds = split["test"], split["train"]
    adata_train = adata[adata.obs.cov_pert.isin(train_conds)]
    corr_mtx = np.corrcoef(adata_train.X.T)

    corr_pred = []
    for cond in test_conds:
        cov, pert = cond.split('_')
        if pert not in var_names_all:
            print(f"{cond}: Gene {pert} is not included in measured genes")
            continue
        pert_idx = var_names_all.index(pert)
        
        ctrl_ad = adata[adata.obs.cov_pert == f'{cov}_control']
        pert_value = ctrl_ad.X[:, pert_idx].copy()
        pert_corr = corr_mtx[pert_idx, :]
        exp_change = np.dot(pert_value.reshape(-1, 1), pert_corr.reshape(1, -1))
        post_expr = (ctrl_ad.X - exp_change).mean(axis=0).reshape(1, -1)
        pred_ad = pd.DataFrame(post_expr, index=[cond], columns=ctrl_ad.var_names)
        pred_ad['cov_pert'] = cond
        corr_pred.append(pred_ad)

    corr_pred = pd.concat(corr_pred)[hvgs+['cov_pert']]
    result[f'split_{i}'] = corr_pred

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(result, f)
