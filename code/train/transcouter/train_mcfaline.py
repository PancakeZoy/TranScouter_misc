import gc
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from transcouter import ScouterData, TranScouter  # noqa: E402


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#################### Final McFaline Hyperparameters ####################
embd_source = "GenePT_V1"
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
w_norm = 0.25
w_direction = 0.5
lr = 0.0001
if_nonzero = False
n_epochs = 50

###################### Load Data ##########################
adata_path = ROOT / "data/mcfaline/processed/mcfaline.h5ad"
embd_path = ROOT / f"data/embeddings/{embd_source}.pickle"
split_path = ROOT / "data/mcfaline/splits/split_dict_5Fold.pkl"
output_dir = ROOT / "results/transcouter/mcfaline"
output_dir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(adata_path)

rename_dict = {
    "COQ8A": "ADCK3",
    "COQ8B": "ADCK4",
    "GRK2": "ADRBK1",
    "GRK3": "ADRBK2",
    "STKLD1": "C9orf96",
    "HASPIN": "GSG2",
    "CILK1": "ICK",
    "MAP3K21": "KIAA1804",
    "STK26": "MST4",
    "NIM1K": "NIM1",
    "PAK5": "PAK7",
    "COL1A2": "SGK494",
    "MAP3K20": "ZAK",
}
with open(embd_path, "rb") as f:
    embd = pd.DataFrame(pickle.load(f)).T
    embd.rename(index=rename_dict, inplace=True)

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
pertdata.key_cov_pert = "cov_pert"
pertdata.key_uns_DegNames = "top_degs_names"
pertdata.key_uns_DegIdx = "top_degs_idx"
pertdata.key_uns_NonzeroIdx = "nonzero_gene_idx"

with open(split_path, "rb") as f:
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
    scouter_model.data_init(key_stratify=["covariate"])
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
        n_epochs=n_epochs,
    )
    metric_df, result_dict = scouter_model.evaluate()

    metric_df["cov_pert"] = metric_df.index
    metric_df[["covariate", "perturbation"]] = metric_df["cov_pert"].str.split("_", expand=True)  # fmt: skip
    metric_df["cov_seen"] = ~metric_df["covariate"].isin(test_cov)
    metric_df["pert_seen"] = ~metric_df["perturbation"].isin(masked_perts)
    metric_df_ls.append(metric_df.assign(split=split))

    result_dict_ls[f"split_{split}"] = result_dict

metric_df = pd.concat(metric_df_ls)
metric_df.to_csv(output_dir / "metrics.csv")

with open(output_dir / "predictions.pkl", "wb") as f:
    pickle.dump(result_dict_ls, f)
