import scanpy as sc
import pandas as pd
import pickle
import argparse
import itertools
import numpy as np
import random
import torch
import gc
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from transcouter import TranScouter, ScouterData

ADATA_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
EMBD_PATH = ROOT / "data" / "embeddings" / "GenePT_V1.pickle"
SPLIT_PATH = ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl"
OUT_DIR = ROOT / "results" / "ablation" / "train_test_distance"

def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#################### Hyperparameters ####################
hidden_enc_embd = (512, 256)
hidden_enc_ctrl = (2048, 1024)
bottle_dim = 512
hidden_dec = (1024, 2048)
use_batch_norm = True
use_layer_norm = False
dropout_rate = 0.0
use_sampling = False
w_kld = 0.0
condition = "control"
batch_size = 512
w_norm = 0.0
w_direction = 0.25
lr = 0.0001
if_nonzero = False
###################### Load Data ##########################
adata = sc.read_h5ad(ADATA_PATH)
with open(EMBD_PATH, "rb") as f:
    embd = pd.DataFrame(pickle.load(f)).T
    embd.rename(index={"H1-0": "H1F0"}, inplace=True)

set_seeds()
pertdata = ScouterData(
    adata=adata,
    embd=embd,
    key_pert="perturbation",
    key_cov="covariate",
    ctrl_value="control",
    key_var_gnames="gene_name",
)

pertdata.setup_ad("embd_index")
pertdata.gene_ranks(pval_cutoff=0.1)
pertdata.get_nonzero_genes()

with open(SPLIT_PATH, "rb") as f:
    split_dict = pickle.load(f)
n_split = len(split_dict)

del adata, embd
###################### Train & Evaluate ##########################
metric_df_ls = []
result_dict_ls = {}
for split in range(n_split):
    gc.collect()
    set_seeds()
    train_conds, val_conds, test_conds, masked_perts, test_cov = (
        split_dict[split]["train"],
        split_dict[split]["val"],
        split_dict[split]["test"],
        split_dict[split]["masked_perts"],
        split_dict[split]["test_cov"],
    )
    pertdata.split_Train_Val_Test(train_conds=train_conds, val_conds=val_conds, test_conds=test_conds)  # fmt: skip
    scouter_model = TranScouter(pertdata)
    scouter_model.data_init(key_stratify=["covariate", "bulk"])
    scouter_model.model_init(
        hidden_enc_embd=hidden_enc_embd,
        hidden_enc_ctrl=hidden_enc_ctrl,
        bottle_dim=bottle_dim,
        hidden_dec=hidden_dec,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate,
        use_sampling=use_sampling,
        condition=condition,
    )
    scouter_model.train(
        batch_size=batch_size,
        w_norm=w_norm,
        w_direction=w_direction,
        w_kld=w_kld,
        lr=lr,
        if_nonzero=if_nonzero,
    )
    metric_df, result_dict = scouter_model.evaluate()
    
    metric_df["cov_pert"] = metric_df.index
    metric_df[["covariate", "perturbation"]] = metric_df["cov_pert"].str.split("_", expand=True)  # fmt: skip
    metric_df["cov_seen"] = ~metric_df["covariate"].isin(test_cov)
    metric_df["pert_seen"] = ~metric_df["perturbation"].isin(masked_perts)
    metric_df_ls.append(metric_df.assign(split=split))

    result_dict_ls[f"split_{split}"] = result_dict

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / f"model_split{split}.pt"
    torch.save(scouter_model.network.state_dict(), model_path)

metric_df = pd.concat(metric_df_ls)
metric_df.to_csv(OUT_DIR / "Jiang.csv")

with open(OUT_DIR / "Jiang.pkl", "wb") as f:
    pickle.dump(result_dict_ls, f)
