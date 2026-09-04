import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import gc
from pathlib import Path
from perturbench.analysis.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[3]
data_directory = ROOT / "data/jiang/raw"

adata_paths = [
    "Seurat_object_IFNB_Perturb_seq.h5ad",
    "Seurat_object_IFNG_Perturb_seq.h5ad",
    "Seurat_object_INS_Perturb_seq.h5ad",
    "Seurat_object_TGFB_Perturb_seq.h5ad",
    "Seurat_object_TNFA_Perturb_seq.h5ad",
]
adata_paths = [data_directory / path for path in adata_paths]

adata_list = []
for adata_path in adata_paths:
    print(adata_path)
    adata = sc.read_h5ad(adata_path)
    adata.X = adata.raw.X.copy()
    adata.raw = None

    adata.obs.cell_type = [x.lower() for x in adata.obs.cell_type]
    adata.obs.cell_type.value_counts()

    adata.obs["treatment"] = adata_path.name.split("_")[2]
    adata.obs.treatment.value_counts()

    condition_remap = {
        "NT": "control",
    }
    adata.obs["condition"] = adata.obs.gene.copy()
    adata.obs.condition = [condition_remap.get(x, x) for x in adata.obs.condition]
    adata.obs["condition"] = adata.obs.condition.astype("category")
    adata.obs["perturbation"] = adata.obs.condition.copy()

    adata.obs["ncounts"] = adata.obs["nCount_RNA"].copy()
    adata.obs["ngenes"] = adata.obs["nFeature_RNA"].copy()
    adata.obs["perturbation_type"] = "CRISPRi"

    adata_list.append(adata)
    del adata
    gc.collect()


adata_merged = ad.concat(adata_list)
adata_merged.obs_names_make_unique()

del adata_list
gc.collect()

adata_merged.obs["dataset"] = "jiang24"

required_cols = [
    "condition",
    "cell_type",
    "treatment",
    "perturbation_type",
    "dataset",
    "ngenes",
    "ncounts",
]

for col in required_cols:
    assert col in adata_merged.obs.columns
    if np.any(adata_merged.obs[col].isnull()):
        print(col)
    if np.any(adata_merged.obs[col].isna()):
        print(col)

adata_merged = preprocess(
    adata_merged,
    perturbation_key="condition",
    covariate_keys=["cell_type", "treatment"],
)

adata_merged.write_h5ad(data_directory / "jiang24_processed.h5ad")
