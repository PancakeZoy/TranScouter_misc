import scanpy as sc
import pandas as pd
import pickle
import numpy as np
import random
import torch
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from transcouter import TranScouter, ScouterData

ADATA_PATH = ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad"
EMBD_PATH = ROOT / "data" / "embeddings" / "GenePT_V1.pickle"
SPLIT_DIR = ROOT / "data" / "jiang" / "splits" / "trainsize"
OUT_DIR = ROOT / "results" / "ablation" / "trainsize"

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

with open(SPLIT_DIR / f"TrainSize_SplitDict_{args.i}_10Fold.pkl", "rb") as f:
    split_dict = pickle.load(f)
n_split = len(split_dict)
###################### Train & Evaluate ################
metric_df_ls = []
result_dict_ls = {}
for split in range(n_split):
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

metric_df = pd.concat(metric_df_ls)
OUT_DIR.mkdir(parents=True, exist_ok=True)
train_size = 30 - 5 * args.i
metric_df.to_csv(OUT_DIR / f"Jiang_{train_size}.csv")
with open(OUT_DIR / f"Jiang_{train_size}.pkl", "wb") as f:
    pickle.dump(result_dict_ls, f)
