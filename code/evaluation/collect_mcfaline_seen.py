import pickle
from pathlib import Path

import pandas as pd
import scanpy as sc

# fmt: off
ROOT = Path(__file__).resolve().parents[2]
output_path = ROOT / "results/collected/mcfaline/seen_pred_results.pkl"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(ROOT / "data/mcfaline/splits/split_dict_5Fold.pkl", "rb") as f:
    split_dict = pickle.load(f)
with open(ROOT / "results/transcouter/mcfaline/predictions.pkl", "rb") as f:
    trans_result = pickle.load(f)

adata = sc.read_h5ad(ROOT / "data/mcfaline/processed/mcfaline.h5ad")

pred_trans = []
pred_avg = []
pred_scgen = []

for i in range(5):
    print(f"Processing split {i}")
    test_conds, masked_perts, test_cov = split_dict[i]['test'], split_dict[i]['masked_perts'], split_dict[i]['test_cov']
    test_conds = pd.Series(test_conds, index=test_conds).str.split('_', expand=True)
    test_conds.columns = ['covariate', 'perturbation']
    test_conds['cov_unseen'] = test_conds['covariate'].isin(test_cov)
    test_conds['pert_unseen'] = test_conds['perturbation'].isin(masked_perts)
    eval_conds = test_conds[test_conds.cov_unseen & (~test_conds.pert_unseen)]

    # TranScouter
    pred_trans_i = trans_result[f'split_{i}']['Prediction']
    pred_trans_i = pred_trans_i[pred_trans_i.cov_pert.isin(eval_conds.index)].groupby('cov_pert').mean()
    pred_trans.append(pred_trans_i)
    
    # Simple average across all conditions of the same perturbation
    avg_i = []
    for cond in eval_conds.index:
        cov, pert = cond.split('_')
        avg_cond = adata[(~adata.obs.covariate.isin(test_cov)) & (adata.obs.perturbation == pert)].X.toarray().mean(axis=0)
        avg_i.append(pd.Series(avg_cond, index=adata.var_names, name=cond))
    avg_i = pd.concat(avg_i, axis=1).T
    pred_avg.append(avg_i)

    # scGen
    pred_scgen_i = sc.read_h5ad(ROOT / f"results/baselines/scgen/mcfaline/mcfaline_{i}.h5ad")
    pred_scgen_i = pd.DataFrame(pred_scgen_i.X, index=pred_scgen_i.obs.index, 
                                columns=pred_scgen_i.var_names).assign(cov_pert=pred_scgen_i.obs.cov_pert)
    pred_scgen_i['cov_pert'] = pred_scgen_i['cov_pert'].astype(str)
    pred_scgen_i = pred_scgen_i.groupby('cov_pert').mean()
    pred_scgen.append(pred_scgen_i)
    
pred_trans = pd.concat(pred_trans).clip(lower=0)
pred_avg = pd.concat(pred_avg).loc[pred_trans.index]
pred_scgen = pd.concat(pred_scgen).loc[pred_trans.index]

true = adata[adata.obs.cov_pert.isin(pred_trans.index)]
true = pd.DataFrame(true.X.toarray(), index=true.obs.index, columns=true.var_names).assign(cov_pert=true.obs.cov_pert)
true = true.groupby('cov_pert').mean()
true = true.loc[pred_trans.index]

ctrl = adata[adata.obs.perturbation == 'control']
ctrl = pd.DataFrame(ctrl.X.toarray(), index=ctrl.obs.index, columns=ctrl.var_names).assign(cov_pert=ctrl.obs.cov_pert)
ctrl = ctrl.groupby('cov_pert').mean()

data = {'TranScouter': pred_trans, 'Avg': pred_avg, 'scGen': pred_scgen, 
        'True': true, 'Ctrl': ctrl}

with open(output_path, "wb") as f:
    pickle.dump(data, f)
