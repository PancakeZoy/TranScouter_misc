import scanpy as sc
import scgen
import pickle
import pandas as pd
import warnings
import logging
import numpy as np
import random
import torch
from pathlib import Path
from lightning.pytorch import seed_everything
import scvi

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("scvi").setLevel(logging.WARNING)


ROOT = Path(__file__).resolve().parents[3]
ADATA_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
SPLIT_PATH = ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl"
OUT_DIR = ROOT / "results" / "baselines" / "scgen" / "jiang"


# fmt: off
def set_seeds(seed=24):
    seed_everything(seed, workers=True)
    scvi.settings.seed = seed
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
    train_adata = train_adata[train_adata.obs.perturbation.isin(test_perts)].copy()

    scgen.SCGEN.setup_anndata(train_adata, batch_key="perturbation", labels_key="covariate")
    model = scgen.SCGEN(train_adata)
    model.train(
        max_epochs=100,
        deterministic=True,
        train_size=0.8,
        batch_size=512,
        early_stopping=True,
        early_stopping_patience=25
    )

    pred_split = []
    for index, row in test_df.iterrows():
        cov_i, pert_i, covpert_i = row.values
        set_seeds()
        pred, delta = model.predict(
            ctrl_key='control',
            stim_key=pert_i,
            celltype_to_predict=cov_i
        )
        pred.obs['covariate'] = cov_i
        pred.obs['perturbation'] = pert_i
        pred.obs['cov_pert'] = covpert_i
        pred.obs = pred.obs[['covariate', 'perturbation', 'cov_pert', 'bulk', 'donor']].copy()
        pred_split.append(pred)
    pred_split = sc.concat(pred_split)
    pred_split.obs_names = [idx.replace("control", row["perturbation"]) for idx, row in pred_split.obs.iterrows()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_split.write_h5ad(OUT_DIR / f"jiang_{split_i}.h5ad")


for split_i in range(10):
    main(split_i)
