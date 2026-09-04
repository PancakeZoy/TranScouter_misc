import scanpy as sc
import trvaep
import numpy as np
import random
import torch
import pickle
import pandas as pd
import anndata as ad
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADATA_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
SPLIT_PATH = ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl"
OUT_DIR = ROOT / "results" / "baselines" / "trvae" / "jiang"


# fmt: off
def set_seeds(seed=24):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def main(split_i):
	set_seeds()
	adata = sc.read_h5ad(ADATA_PATH)
	adata.obs['cov_pert'] = adata.obs[['covariate', 'perturbation']].apply(lambda x: '_'.join(x), axis=1)
	with open(SPLIT_PATH, "rb") as f:
		split_dict = pickle.load(f)

	test, val, train, masked_perts, test_cov = split_dict[split_i].values()
	test_df = pd.DataFrame([i.split('_') + [i] for i in test], columns=['covariate', 'perturbation', 'cov_pert'])
	test_df = test_df[~test_df.perturbation.isin(masked_perts)]
	test_df = test_df[test_df.perturbation != 'control']

	train_adata = adata[~adata.obs.cov_pert.isin(test_df.cov_pert)].copy()
	test_perts = test_df.perturbation.unique().tolist()+['control']
	train_adata = train_adata[train_adata.obs.perturbation.isin(test_perts)]
	train_adata = train_adata[~train_adata.obs.covariate.isin(test_df.covariate.unique())].copy()

	condition_key = "perturbation"
	n_conditions = train_adata.obs[condition_key].unique().shape[0]

	model = trvaep.CVAE(train_adata.n_vars, num_classes=n_conditions)
	trainer = trvaep.Trainer(model, train_adata, condition_key=condition_key)
	trainer.train_trvae()

	pred_split = []
	for index, row in test_df.iterrows():
		set_seeds()
		cov_i, pert_i, covpert_i = row.values
		adata_source = adata[(adata.obs["covariate"] == cov_i) & (adata.obs[condition_key] == "control")]
		x, y, target = adata_source.X.toarray().astype(np.float32), adata_source.obs[condition_key].tolist(), pert_i
		predicted_X = model.predict(x=x, y=y, target=target)

		pred_adata = ad.AnnData(X=predicted_X.mean(axis=0, keepdims=True), var=adata_source.var.copy())
		pred_adata.obs['covariate'] = cov_i
		pred_adata.obs["perturbation"] = pert_i
		pred_adata.obs["cov_pert"] = covpert_i
		pred_split.append(pred_adata)
	pred_split = sc.concat(pred_split)
	pred_split.obs_names = pred_split.obs.cov_pert.tolist()
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	pred_split.write_h5ad(OUT_DIR / f"jiang_{split_i}.h5ad")


parser = argparse.ArgumentParser()
parser.add_argument('split_i', type=int, help='Split index for cross validation')
args = parser.parse_args()
split_i = args.split_i

main(split_i)
