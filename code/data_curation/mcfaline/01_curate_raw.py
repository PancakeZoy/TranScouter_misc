import scanpy as sc
import numpy as np
import anndata as ad
import gc
from pathlib import Path
from scipy.sparse import csr_matrix

from perturbench.analysis.utils import get_ensembl_mappings
from perturbench.analysis.preprocess import preprocess


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "mcfaline" / "raw"
OUT_PATH = DATA_DIR / "mcfaline23_gxe_processed.h5ad"
OBS_META_PATH = DATA_DIR / "obs_meta.csv"
VAR_META_PATH = DATA_DIR / "var_meta.csv"

id_to_gene = get_ensembl_mappings()
id_to_gene = {k:v for k,v in id_to_gene.items() if isinstance(v, str) and v != ''}
len(id_to_gene.keys())

data_paths = [
    DATA_DIR / "gxe1.h5ad",
    DATA_DIR / "gxe2_A172.h5ad",
    DATA_DIR / "gxe2_T98G.h5ad",
    DATA_DIR / "gxe2_U87MG.h5ad",
]

adata_list = []
for path in data_paths:
    adata_list.append(sc.read_h5ad(path))
adata = ad.concat(adata_list)

adata.X = csr_matrix(adata.X)

adata.obs_names_make_unique()

del adata_list
gc.collect()

print(adata.obs.dose.value_counts())

adata.obs['drug_dose'] = adata.obs.dose.copy()
adata.obs.cell_type = [x.lower() for x in adata.obs.cell_type]
adata.obs.cell_type = adata.obs.cell_type.astype('category')
adata.obs.cell_type.value_counts()

adata.obs.rename(columns = {
    'nCount_RNA': 'ncounts',
    'nFeature_RNA': 'ngenes',
}, inplace=True)
adata.obs['perturbation_type'] = 'CRISPRi'
adata.obs['dataset'] = 'mcfaline23'


# Rename Perturbations
adata.obs['gene_id'] = [x.replace(',', '+') for x in adata.obs.gene_id]

gene_controls = ['NA', 'NTC', 'random']
for ctrl in gene_controls:
    adata.obs['gene_id'] = [x.replace(ctrl, 'control') for x in adata.obs['gene_id']]
print(adata.obs.gene_id.value_counts())

single_gene_perts = [x for x in adata.obs.gene_id.unique() if '+' not in x]
adata = adata[adata.obs.gene_id.isin(single_gene_perts),:]

drug_controls = ['vehicle', 'dmso']
for ctrl in drug_controls:
    adata.obs['treatment'] = [x.replace(ctrl, 'none') for x in adata.obs['treatment']]
print(adata.obs.treatment.value_counts())


gene_dose = []
for gene in adata.obs.gene_id:
    ngenes = len(gene.split('+'))
    dose = '+'.join(['1']*ngenes)
    gene_dose.append(dose)
    
adata.obs['gene_dose'] = gene_dose
print(adata.obs.gene_dose.value_counts())

adata.obs['perturbation'] = adata.obs.gene_id.astype('category').copy()
print(adata.obs.perturbation.value_counts())

adata.obs['pert_cl_tr'] = adata.obs['perturbation'].astype(str) + '_' + adata.obs['cell_type'].astype(str) + '_' + adata.obs['treatment'].astype(str)
pert_cl_tr_counts = adata.obs.pert_cl_tr.value_counts()
pert_cl_tr_keep = list(pert_cl_tr_counts.loc[pert_cl_tr_counts >= 20].index)
print(len(pert_cl_tr_keep))

adata = adata[adata.obs.pert_cl_tr.isin(pert_cl_tr_keep)]

adata.obs['condition'] = adata.obs['perturbation'].copy()
adata.obs.condition = adata.obs.condition.astype('category')
adata.obs.perturbation = adata.obs.perturbation.astype('category')

adata.var['gene_id'] = [x.split('.')[0] for x in adata.var_names]
adata = adata[:,[x in id_to_gene for x in adata.var['gene_id']]]

adata.var['gene_name'] = [str(id_to_gene[x]) for x in adata.var['gene_id']]
adata = adata[:,[x != '' for x in adata.var['gene_name']]]

adata.var_names = adata.var.gene_name.astype(str).copy()

adata = adata[:,['nan' not in x for x in adata.var_names]]

duplicated_genes = adata.var.index.duplicated()
adata = adata[:,~duplicated_genes]


required_cols = [
    'condition',
    'cell_type',
    'treatment',
    'perturbation_type',
    'dataset',
    'ngenes',
    'ncounts',
]

for col in required_cols:
    assert col in adata.obs.columns
    if np.any(adata.obs[col].isnull()):
        print(col)
    if np.any(adata.obs[col].isna()):
        print(col)


adata = adata.copy()
gc.collect()

condition_plus_treatment = []
for condition, treatment in zip(adata.obs.condition, adata.obs.treatment):
    if treatment == 'none':
        condition_plus_treatment.append(str(condition))
    else:
        condition_plus_treatment.append(str(condition) + '+' + str(treatment))

adata.obs['condition_plus_treatment'] = condition_plus_treatment
adata.obs['condition_plus_treatment'] = adata.obs['condition_plus_treatment'].astype('category')
print(adata.obs.condition_plus_treatment.value_counts())

unique_obs = adata.obs.loc[:,['condition', 'cell_type', 'treatment']].drop_duplicates()
print(unique_obs.treatment.value_counts())

treatments_remove = [
    'temozolomide',
    'thioguanine'
]

adata = adata[~adata.obs.treatment.isin(treatments_remove)].to_memory()
print(adata)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(OUT_PATH)
adata.obs.to_csv(OBS_META_PATH)
adata.var.to_csv(VAR_META_PATH)

