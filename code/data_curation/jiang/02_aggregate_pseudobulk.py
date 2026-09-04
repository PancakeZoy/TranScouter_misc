import anndata as ad
import scanpy as sc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Load the h5ad file
file_path = ROOT / "data/jiang/raw/jiang24_processed.h5ad"
adata = ad.read_h5ad(file_path)

# Reduce .obs to only contain specified columns
columns_to_keep = ["cell_type", "treatment", "perturbation", "sample_ID", "Batch_info"]
adata.obs = adata.obs[columns_to_keep]

# Aggregation
agg_funs = ["sum"]
agg_adata = sc.get.aggregate(adata, by=columns_to_keep, func=agg_funs, layer="counts")

# rename .obs.columns
rename_dict = {"sample_ID": "sample_id", "Batch_info": "batch"}
agg_adata.obs = agg_adata.obs.rename(columns=rename_dict)


# Save the filtered AnnData object locally
output_path = ROOT / "data/jiang/processed/aggregate_sum.h5ad"
output_path.parent.mkdir(parents=True, exist_ok=True)
agg_adata.write(output_path)

print(f"Filtered data saved to {output_path}")
