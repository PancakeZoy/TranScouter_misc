# TranScouter

This repository contains the code and reproducibility workflow for the TranScouter manuscript. It includes the local `transcouter/` implementation, data-processing scripts, training scripts, evaluation utilities, and figure notebooks.

Run commands from the repository root:

```bash
cd /path/to/TranScouter_misc
```

No package build step is required. The scripts import the local `transcouter/` module directly.

## Repository layout

- `transcouter/`: TranScouter model and data utilities.
- `data/`: raw data, processed data, embeddings, split files, and data-derived metadata. Large files are ignored by Git; see `data/README.md`.
- `results/`: model predictions, baseline outputs, ablation outputs, and figure-ready collected result files.
- `code/data_curation/`: Jiang24 and McFaline preprocessing scripts.
- `code/train/transcouter/`: TranScouter training scripts for the paper settings.
- `code/train/baselines/`: benchmark and baseline scripts.
- `code/train/ablation/`: ablation training and analysis scripts.
- `code/evaluation/`: scripts that collect predictions into figure-ready result files.
- `figures/main/`: notebooks for main figures.
- `figures/supplement/`: notebooks for supplementary figures.

## Environment

Use a Python environment with the scientific Python stack and the method-specific packages required by the scripts you run. Core dependencies include `numpy`, `pandas`, `scipy`, `scanpy`, `anndata`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`, and `adjustText`. Additional benchmark scripts require their corresponding packages, such as `scgen`, `trvaep`, and `celloracle`.

## Data preparation

Place the raw data files and embedding files under `data/` as described in `data/README.md`. Large local artifacts such as `.h5ad`, `.pkl`, `.pickle`, and model checkpoint files are excluded from Git by default.

### Jiang24

```bash
python code/data_curation/jiang/01_curate_singlecell.py
python code/data_curation/jiang/02_aggregate_pseudobulk.py
python code/data_curation/jiang/03_make_hvg.py
python code/data_curation/jiang/04_make_splits.py

for i in 1 2 3 4 5; do
  python code/data_curation/jiang/05_make_trainsize_splits.py $i
done

python code/data_curation/jiang/06_make_degs.py
python code/data_curation/jiang/07_make_celloracle_ctrl.py
```

### McFaline

```bash
python code/data_curation/mcfaline/01_curate_raw.py
python code/data_curation/mcfaline/02_make_final_h5ad.py
python code/data_curation/mcfaline/03_make_splits.py
python code/data_curation/mcfaline/04_make_extra_genes.py
```

## Model and baseline outputs

The following commands generate the TranScouter predictions and method outputs used by the evaluation and figure notebooks.

Large result files are not stored in Git. To reproduce the manuscript figures directly, place the separately provided precomputed outputs under `results/`; otherwise, run the training and baseline scripts below to generate them. Because neural-network training can depend on CPU/GPU hardware, backend libraries, and low-level numerical behavior, regenerated outputs may not be bit-identical to the precomputed outputs.

### TranScouter

```bash
python code/train/transcouter/train_jiang.py
python code/train/transcouter/train_mcfaline.py
```

### Benchmarks and baselines

```bash
python code/train/baselines/train_scgen_jiang.py
python code/train/baselines/train_scgen_mcfaline.py

for i in 0 1 2 3 4 5 6 7 8 9; do
  python code/train/baselines/train_trvae_jiang.py $i
done

python code/train/baselines/run_correlation_jiang.py
python code/train/baselines/run_correlation_mcfaline.py

for i in 0 1 2 3 4 5 6 7 8 9; do
  python code/train/baselines/run_celloracle_jiang.py $i
done

python code/train/baselines/run_delta_average.py
```

### Ablations

```bash
python code/train/ablation/train_jiang_architecture_ablation.py
python code/train/ablation/train_jiang_random_pairing.py

for i in 1 2 3 4 5; do
  python code/train/ablation/train_jiang_trainsize.py $i
  python code/train/ablation/analyze_jiang_trainsize.py $i
done

python code/train/ablation/train_jiang_train_test_distance.py
```

## Figure-ready result collection

Run these after TranScouter and the required baselines have produced their outputs:

```bash
python code/evaluation/collect_jiang_seen.py
python code/evaluation/collect_jiang_unseen.py
python code/evaluation/collect_mcfaline_seen.py
python code/evaluation/collect_mcfaline_unseen.py
```

## Figures

Open the notebooks under `figures/main/` and `figures/supplement/` to reproduce the manuscript figures. The notebooks display figures inline by default.

Each figure notebook has a top-level switch:

```python
SAVE_FIGURES = False
```

Change it to `True` to write image files to the corresponding `figures/main/output/` or `figures/supplement/output/` directory.
