import pickle
import pandas as pd
import scanpy as sc
from pathlib import Path

# fmt: off
ROOT = Path(__file__).resolve().parents[2]
output_path = ROOT / "results/collected/jiang/unseen_pred_results.pkl"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(ROOT / "data/jiang/splits/split_dict_10Fold.pkl", "rb") as f:
    split_dict = pickle.load(f)
with open(ROOT / "results/transcouter/jiang/predictions.pkl", "rb") as f:
    trans_result = pickle.load(f)
with open(ROOT / "data/jiang/processed/top_degs_names.pkl", "rb") as f:
    top_degs_idx = pickle.load(f)
with open(ROOT / "results/baselines/correlation/jiang/corr_nonctrl.pkl", "rb") as f:
    corr_pred = pickle.load(f)
    
adata = sc.read_h5ad(ROOT / "data/jiang/processed/aggregate_sum_hvg.h5ad")
adata.obs['cov_pert'] = adata.obs['covariate'].str.cat(adata.obs['perturbation'], sep='_')

pred_trans = []
pred_avg = []
pred_co = []
pred_corr = []
for i in range(10):
    print(f"Processing split {i}")
    test_conds, masked_perts, test_cov = split_dict[i]['test'], split_dict[i]['masked_perts'], split_dict[i]['test_cov']
    test_conds = pd.Series(test_conds, index=test_conds).str.split('_', expand=True)
    test_conds.columns = ['covariate', 'perturbation']
    test_conds['cov_unseen'] = test_conds['covariate'].isin(test_cov)
    test_conds['pert_unseen'] = test_conds['perturbation'].isin(masked_perts)
    unseen_conds = test_conds[test_conds.cov_unseen & test_conds.pert_unseen]

    # TranScouter
    pred_trans_i = trans_result[f'split_{i}']['Prediction']
    pred_trans_i = pred_trans_i[pred_trans_i.cov_pert.isin(unseen_conds.index)].groupby('cov_pert').mean()
    pred_trans.append(pred_trans_i)
    
    # Simple average across all conditions of the same perturbation
    avg_i = []
    for cond in unseen_conds.index:
        cov, pert = cond.split('_')
        avg_cond = adata[(~adata.obs.covariate.isin(test_cov)) & (adata.obs.perturbation == pert)].X.mean(axis=0)
        avg_i.append(pd.Series(avg_cond, index=adata.var_names, name=cond))
    avg_i = pd.concat(avg_i, axis=1).T
    pred_avg.append(avg_i)

    # CellOracle
    pred_co_i = sc.read_h5ad(ROOT / f"results/baselines/celloracle/jiang/CoPred_{i}.h5ad")
    pred_co_i = pd.DataFrame(pred_co_i.X, index=pred_co_i.obs.index, 
                             columns=pred_co_i.var_names).assign(cov_pert=pred_co_i.obs.cov_pert)
    pred_co_i['cov_pert'] = pred_co_i['cov_pert'].astype(str)
    pred_co_i = pred_co_i.groupby('cov_pert').mean()
    pred_co_i = pred_co_i.loc[[i for i in unseen_conds.index if i in pred_co_i.index]]
    pred_co.append(pred_co_i)
    
    # Correlation based method
    corr_pred_i = corr_pred[f'split_{i}'].groupby('cov_pert').mean()
    corr_pred_i = corr_pred_i.loc[[i for i in unseen_conds.index if i in corr_pred_i.index]]
    pred_corr.append(corr_pred_i)
    
pred_trans = pd.concat(pred_trans).clip(lower=0)
pred_avg = pd.concat(pred_avg).loc[pred_trans.index]
pred_corr = pd.concat(pred_corr).loc[pred_trans.index]
pred_co = pd.concat(pred_co)

true = adata[adata.obs.cov_pert.isin(pred_trans.index)]
true = pd.DataFrame(true.X, index=true.obs.index, columns=true.var_names).assign(cov_pert=true.obs.cov_pert)
true = true.groupby('cov_pert').mean()
true = true.loc[pred_trans.index]

ctrl = adata[adata.obs.perturbation == 'control']
ctrl = pd.DataFrame(ctrl.X, index=ctrl.obs.index, columns=ctrl.var_names).assign(cov_pert=ctrl.obs.cov_pert)
ctrl = ctrl.groupby('cov_pert').mean()

data = {'TranScouter': pred_trans, 'Avg': pred_avg, 'CellOracle': pred_co, 
        'Correlation': pred_corr, 'True': true, 'Ctrl': ctrl}
with open(output_path, "wb") as f:
    pickle.dump(data, f)
