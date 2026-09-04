import scanpy as sc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

adata = sc.read_h5ad(ROOT / "data/jiang/processed/aggregate_sum.h5ad")
adata.X = adata.layers["sum"]
del adata.layers["sum"]

adata.obs["covariate"] = adata.obs[["cell_type", "treatment"]].agg("+".join, axis=1)
adata.obs["bulk"] = adata.obs[["sample_id", "batch"]].agg("_".join, axis=1)
adata.obs["donor"] = adata.obs["bulk"].str.split("_").str[2]
adata.obs = adata.obs[["covariate", "bulk", "perturbation", "donor"]]

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata, batch_key="covariate", flavor="seurat_v3", n_top_genes=5000
)
adata = adata[:, adata.var.highly_variable]
adata.write_h5ad(ROOT / "data/jiang/processed/aggregate_sum_hvg.h5ad")
