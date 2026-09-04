import pandas as pd
import scanpy as sc
from tqdm import tqdm
import warnings
from pathlib import Path
from pandas.errors import PerformanceWarning


ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "data" / "mcfaline" / "raw" / "mcfaline23_gxe_processed.h5ad"
OUT_PATH = ROOT / "data" / "mcfaline" / "processed" / "mcfaline.h5ad"


def interleave_rows(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Interleave rows from df1 and df2, one by one.
    If one DataFrame is shorter, then after interleaving as far as possible,
    append the remaining rows of the longer DataFrame.
    """
    n1, n2 = len(df1), len(df2)
    nmin = min(n1, n2)

    if n1 == 0 and n2 == 0:
        return df1.copy()

    interleaved_rows = []
    for i in range(nmin):
        interleaved_rows.append(df1.iloc[[i]])
        interleaved_rows.append(df2.iloc[[i]])
    if n1 > nmin:
        interleaved_rows.append(df1.iloc[nmin:])
    if n2 > nmin:
        interleaved_rows.append(df2.iloc[nmin:])

    return pd.concat(interleaved_rows, ignore_index=True)


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
sc.pp.filter_genes(adata, min_cells=10)
mask_gene = (~adata.var.mt) & (~adata.var.ribo) & (~adata.var.hb)
adata = adata[:, mask_gene].copy()

degs_dict = {}
for cov in adata.obs.covariate.unique():
    print(f"Processing {cov}")
    subset = adata[adata.obs.covariate == cov].copy()
    sc.pp.normalize_total(subset)
    sc.pp.log1p(subset)
    sc.pp.filter_genes(subset, min_cells=5)
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=PerformanceWarning)
        sc.tl.rank_genes_groups(
            subset,
            groupby="perturbation",
            reference="control",
            rankby_abs=True,
            method="wilcoxon",
        )
    df = sc.get.rank_genes_groups_df(subset, group=None)
    df = df[df.pvals < 0.05]

    subset_ctrl = subset[subset.obs.perturbation == "control"].copy()
    for pert in tqdm(df.group.unique(), desc=f"{cov}:"):
        temp_df = df[df.group == pert].copy()
        pval_degs = temp_df["names"].tolist()
        ctrl_mean = subset_ctrl[:, pval_degs].X.toarray().mean(axis=0)
        pert_mean = (
            subset[subset.obs.perturbation == pert, pval_degs].X.toarray().mean(axis=0)
        )
        diff_mean = pert_mean - ctrl_mean
        temp_df["change"] = diff_mean

        subset_pos = temp_df[temp_df.change > 0].sort_values("change", ascending=False)
        subset_neg = temp_df[temp_df.change < 0].sort_values("change", ascending=True)
        k_degs = interleave_rows(subset_pos, subset_neg).head(20).names.tolist()
        key = f"{cov}_{pert}"
        degs_dict[key] = k_degs

all_degs = set()
for deg_list in degs_dict.values():
    all_degs.update(deg_list)
all_degs = list(all_degs)

adata = adata[:, all_degs].copy()
var_idx = adata.var_names
degs_idx = {
    cov_pert: list(var_idx.get_indexer(deg_names))
    for cov_pert, deg_names in degs_dict.items()
}
adata.uns["top_degs_names"] = degs_dict
adata.uns["top_degs_idx"] = degs_idx
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
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
adata.obs = adata.obs[keep_cols]
adata.var = adata.var[["gene_id", "gene_name"]]
del adata.uns["log1p"]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(OUT_PATH)
