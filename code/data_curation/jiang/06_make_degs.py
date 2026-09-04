import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from transcouter import ScouterData  # noqa: E402


adata = sc.read_h5ad(ROOT / "data/jiang/processed/aggregate_sum_hvg.h5ad")

# DEG calling does not use embeddings, but ScouterData expects a DataFrame.
perturbations = adata.obs["perturbation"].unique().tolist()
embd = pd.DataFrame(np.zeros((len(perturbations), 1)), index=perturbations)

pertdata = ScouterData(
    adata=adata,
    embd=embd,
    key_pert="perturbation",
    key_cov="covariate",
    ctrl_value="control",
    key_var_gnames="gene_name",
)
pertdata.gene_ranks(pval_cutoff=0.1)
pertdata.get_nonzero_genes()

output_dir = ROOT / "data/jiang/processed"
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "top_degs_names.pkl", "wb") as f:
    pickle.dump(pertdata.adata.uns["top_degs_names"], f)
with open(output_dir / "top_degs_idx.pkl", "wb") as f:
    pickle.dump(pertdata.adata.uns["top_degs_idx"], f)
with open(output_dir / "adata_uns.pkl", "wb") as f:
    pickle.dump(pertdata.adata.uns, f)
