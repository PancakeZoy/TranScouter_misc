from pathlib import Path

import scanpy as sc

ROOT = Path(__file__).resolve().parents[3]

adata = sc.read_h5ad(ROOT / "data/jiang/processed/aggregate_sum.h5ad")
ctrl = adata[adata.obs["perturbation"] == "control"].copy()
ctrl.write_h5ad(ROOT / "data/jiang/processed/aggregate_sum_ctrl.h5ad")
