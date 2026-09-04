import scanpy as sc
import pickle
from pathlib import Path
from scipy.sparse import diags


ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "data" / "mcfaline" / "raw" / "mcfaline23_gxe_processed.h5ad"
FINAL_PATH = ROOT / "data" / "mcfaline" / "processed" / "mcfaline.h5ad"
SPLIT_PATH = ROOT / "data" / "mcfaline" / "splits" / "split_dict_5Fold.pkl"
OUT_PATH = ROOT / "data" / "mcfaline" / "processed" / "mcfaline_extra.h5ad"


adata = sc.read(RAW_PATH)
adata.obs["perturbation"] = adata.obs["perturbation"].str.replace(
    "RcontrolSEL", "RNASEL"
)
del adata.raw

# Filter cells based on ncounts and ngenes
mask1 = adata.obs["sample"] != "sci3_A172_MMR_HPRT1_CROPseq"
mask2 = adata.obs["gRNA_id"] != "NA"
mask3 = adata.obs.dose.isin([0.0, 10.0])
mask4 = (adata.obs.ncounts > 2000) & (adata.obs.ncounts < 8000)
mask5 = (adata.obs.ngenes > 500) & (adata.obs.ngenes < 4000)
adata = adata[mask1 & mask2 & mask3 & mask4 & mask5].copy()

# Filter cells based on mt percentage
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=False, inplace=True
)
mask5 = adata.obs["pct_counts_mt"] < 25
adata = adata[mask5].copy()

# Filter perturbation whose cell size are too small
adata.obs = adata.obs.drop(
    columns=[
        "ngenes",
        "cell",
        "sample",
        "Size_Factor",
        "n.umi",
        "new_cell",
        "perturbation_type",
        "dataset",
        "gene_dose",
        "pert_cl_tr",
        "condition",
        "condition_plus_treatment",
        "PCR_plate",
        "drug_dose",
    ]
)
adata.obs["covariate"] = adata.obs[["cell_type", "treatment"]].agg("+".join, axis=1)
adata.obs["cov_pert"] = adata.obs[["covariate", "perturbation"]].agg("_".join, axis=1)
covpert_counts = adata.obs.cov_pert.value_counts()
valid_covperts = covpert_counts[covpert_counts > 10].index.tolist()
adata = adata[adata.obs.cov_pert.isin(valid_covperts)].copy()

# Filter genes
adata_degs_normalized = sc.read(FINAL_PATH)
var_degs = list(adata_degs_normalized.var_names)
adata_degs_raw = adata[adata_degs_normalized.obs_names, var_degs]
raw_cellcounts = adata_degs_raw.X.sum(axis=1).A1
factor = 1e4 / raw_cellcounts

with open(SPLIT_PATH, "rb") as f:
    split_dict = pickle.load(f)
perturbed_genes = sorted(list(set([i.split('_')[1] for split_i in range(5) for i in split_dict[split_i]["test"]])))
extra_genes = [g for g in perturbed_genes if g not in adata_degs_normalized.var_names and g in adata.var_names]

# Normalize extra genes
adata_extra = adata[adata_degs_normalized.obs_names, extra_genes]
adata_extra.X = diags(factor) @ adata_extra.X
sc.pp.log1p(adata_extra)
keep_cols = [
    "orig.ident",
    "dose",
    "gRNA_id",
    "cell_type",
    "treatment",
    "covariate",
    "perturbation",
    "cov_pert",
]
adata_extra.obs = adata_extra.obs[keep_cols]
adata_extra.var = adata_extra.var[["gene_id", "gene_name"]]
del adata_extra.uns["log1p"]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
adata_extra.write_h5ad(OUT_PATH)
