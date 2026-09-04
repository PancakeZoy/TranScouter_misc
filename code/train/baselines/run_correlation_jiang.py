import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import pickle
import gc
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl"
ALL_GENE_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum.h5ad"
HVG_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
OUT_PATH = ROOT / "results" / "baselines" / "correlation" / "jiang" / "corr_nonctrl.pkl"

# fmt: off
# See the code at https://github.com/Chen-Li-17/CellPB for more details
with open(SPLIT_PATH, "rb") as f:
    split_dict = pickle.load(f)
perturbed_genes = sorted(list(set([i.split('_')[1] for split_i in range(10) for i in split_dict[split_i]["test"]])))

ad_allgene = sc.read(ALL_GENE_PATH)
ad_allgene.X = ad_allgene.layers["sum"].copy()
del ad_allgene.layers["sum"]
sc.pp.normalize_total(ad_allgene)
sc.pp.log1p(ad_allgene)

adata_5000 = sc.read(HVG_PATH)
var_names_5000 = list(adata_5000.var_names)

extra_genes = [g for g in perturbed_genes if g not in adata_5000.var_names and g in ad_allgene.var_names]
adata = ad.concat([adata_5000, ad_allgene[adata_5000.obs_names, extra_genes]], axis=1)
adata.obs = adata_5000.obs.copy()
adata.obs['cov_pert'] = adata.obs['covariate'].str.cat(adata.obs['perturbation'], sep='_')
var_names_5150 = list(adata.var_names)
del ad_allgene, adata_5000
gc.collect()

result = {}
for i in range(10):
    print(f"Processing split {i}")
    split = split_dict[i]
    test_conds, train_conds = split["test"], split["train"]
    adata_train = adata[adata.obs.cov_pert.isin(train_conds)]
    corr_mtx = np.corrcoef(adata_train.X.T)

    corr_pred = []
    for cond in test_conds:
        cov, pert = cond.split('_')
        if pert not in var_names_5150:
            print(f"{cond}: Gene {pert} is not included in the 5150 genes")
            continue
        pert_idx = var_names_5150.index(pert)
        
        ctrl_ad = adata[adata.obs.cov_pert == f'{cov}_control']
        pert_value = ctrl_ad.X[:, pert_idx].copy()
        pert_corr = corr_mtx[pert_idx, :]
        exp_change = np.dot(pert_value.reshape(-1, 1), pert_corr.reshape(1, -1))
        pred_ad = pd.DataFrame(ctrl_ad.X - exp_change, index=ctrl_ad.obs_names.str.replace('control', pert), columns=ctrl_ad.var_names)
        pred_ad['cov_pert'] = cond
        corr_pred.append(pred_ad)

    corr_pred = pd.concat(corr_pred)[var_names_5000+['cov_pert']]
    result[f'split_{i}'] = corr_pred

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(result, f)
