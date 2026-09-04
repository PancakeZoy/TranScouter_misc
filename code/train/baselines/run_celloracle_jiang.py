import scanpy as sc
import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
import pickle
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl"
CTRL_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_ctrl.h5ad"
HVG_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
OUT_DIR = ROOT / "results" / "baselines" / "celloracle" / "jiang"

# %matplotlib inline

# fmt: off
with open(SPLIT_PATH, "rb") as f:
    split_dict = pickle.load(f)
perturbed_genes = sorted(list(set([i.split('_')[1] for split_i in range(10) for i in split_dict[split_i]["test"]])))

raw_ad_full = sc.read(CTRL_PATH)
raw_ad_full.X = raw_ad_full.layers['sum'].copy()
del raw_ad_full.layers['sum']

norm_ad_full = raw_ad_full.copy()
sc.pp.normalize_total(norm_ad_full)
sc.pp.log1p(norm_ad_full)

adata_5000 = sc.read(HVG_PATH)
adata_5000 = adata_5000[adata_5000.obs.perturbation == 'control']

extra_genes = [g for g in perturbed_genes if g not in adata_5000.var_names and g in raw_ad_full.var_names]
adata = ad.concat([adata_5000, norm_ad_full[adata_5000.obs_names, extra_genes]], axis=1)
adata.obs = adata_5000.obs.copy()
gc.collect()

i = int(sys.argv[1])
if i not in range(10):
    raise ValueError("Input must be an integer from 0-9")
split = split_dict[i]
test_conds, test_cov = split["test"], split["test_cov"]
co_pred = []
for ctrl_cov in test_cov:
    ########################## CellOracle Setup #########################################
    ctrl_ad = adata[adata.obs.covariate == ctrl_cov]
    ctrl_ad.layers["normalized"] = ctrl_ad.X.copy()
    ctrl_ad.layers["raw_count"] = raw_ad_full[ctrl_ad.obs_names, ctrl_ad.var_names].X.copy()

    # The PAGA part is not needed in this case, because we only need gene expression vector, 
    # the intermediate output of celloracle. 
    # We don't need to project it onto any dimension reduction map.
    sc.tl.pca(ctrl_ad, svd_solver='arpack')
    sc.pp.neighbors(ctrl_ad, n_neighbors=4, n_pcs=min(20, ctrl_ad.n_obs-1))
    sc.tl.diffmap(ctrl_ad)
    sc.pp.neighbors(ctrl_ad, n_neighbors=10, use_rep='X_diffmap')
    sc.tl.louvain(ctrl_ad, resolution=0.8)

    base_GRN = co.data.load_human_promoter_base_GRN()
    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=ctrl_ad,
                                            cluster_column_name="louvain",
                                            embedding_name="X_pca")
    oracle.import_TF_data(TF_info_matrix=base_GRN)

    oracle.perform_PCA()
    n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    n_comps = min(n_comps, 50)
    n_cell = oracle.adata.n_obs
    k = max(5, int(0.025*n_cell))
    oracle.knn_imputation(n_pca_dims=n_comps, 
                          k=k, 
                          balanced=True, 
                          b_sight=min(k*8, n_cell-1), 
                          b_maxl=min(k*4, n_cell-1), 
                          n_jobs=4)

    links = oracle.get_links(cluster_name_for_GRN_unit="louvain", alpha=10, verbose_level=10)
    links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
    links.get_network_score()

    oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
    oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)
    
    ########################## CellOracle Simulation ###################################
    conds = [cond for cond in test_conds if ctrl_cov in cond]
    genes = [g.split('_')[1] for g in conds]
    for cond, g in zip(conds, genes):
        if g in adata.var_names and g in oracle.active_regulatory_genes:
            oracle_cond = oracle.copy()
            oracle_cond.simulate_shift(perturb_condition={g: 0.0}, n_propagation=3)
            cond_ad = oracle_cond.adata
            pred_ad = sc.AnnData(X=cond_ad.layers['simulated_count'], 
                                    obs=pd.DataFrame(index=[name.replace('control', g) for name in cond_ad.obs_names]),
                                    var=pd.DataFrame(index=cond_ad.var_names))
            pred_ad.obs['cov_pert'] = cond
            pred_ad.obs[['covariate', 'perturbation']] = pred_ad.obs['cov_pert'].str.split('_', expand=True)
            co_pred.append(pred_ad)
            print(cond)
            gc.collect()
        else:
            print(f"Gene {g} is not included in the baseGRN")
            
co_pred = sc.concat(co_pred, axis=0)
co_pred = co_pred[:, adata_5000.var_names]
OUT_DIR.mkdir(parents=True, exist_ok=True)
co_pred.write_h5ad(OUT_DIR / f"CoPred_{i}.h5ad")
