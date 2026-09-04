import scanpy as sc
import pandas as pd
import pickle
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
SPLIT_PATH = ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl"
OUT_DIR = ROOT / "results" / "ablation" / "architecture"


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#################### Hyperparameters ####################
embd_dict = {
    "GenePT_V1": ROOT / "data" / "embeddings" / "GenePT_V1.pickle",
    "GenePT_V2": ROOT / "data" / "embeddings" / "GenePT_V2.pickle",
    "scELMo": ROOT / "data" / "embeddings" / "scELMo.pickle",
}
embd_source = "GenePT_V1"
hidden_enc_embd = ()
hidden_enc_ctrl = (2048, 1024)
bottle_hidden_dec = [512, (1024, 2048)]
batch_layer_norm = (True, False)
dropout_rate = 0.0
use_sampling_wkld = (False, 0.0)
condition = "control"
batch_size = 512
w_norm = 0.0
w_direction = 0.25
lr = 0.0001
if_nonzero = False
re_cond = False
use_batch_norm, use_layer_norm = batch_layer_norm
bottle_dim, hidden_dec = bottle_hidden_dec
use_sampling, w_kld = use_sampling_wkld
###################### Load Data ##########################
adata = sc.read_h5ad(ADATA_PATH)
embd_path = embd_dict[embd_source]
with open(embd_path, "rb") as f:
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
        re_cond=re_cond,
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
metric_df.to_csv(OUT_DIR / "Jiang.csv")

with open(OUT_DIR / "Jiang.pkl", "wb") as f:
    pickle.dump(result_dict_ls, f)
