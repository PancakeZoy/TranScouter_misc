import os
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "baselines" / "delta_average"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm


PLOT_DATA_PATH = OUT_DIR / "delta_avg_compare_plot_data.pkl"
JIANG_SIMILARITY_CSV = OUT_DIR / "diagnosis_true_delta_similarity.csv"
MCFALINE_TRANSFERABILITY_CSV = OUT_DIR / "diagnosis_mcfaline_transferability.csv"
METHOD_ORDER = ["TranScouter", "DeltaAverage"]

DATASETS = {
    "jiang": {
        "adata_path": ROOT / "data" / "jiang" / "processed" / "aggregate_sum_hvg.h5ad",
        "split_path": ROOT / "data" / "jiang" / "splits" / "split_dict_10Fold.pkl",
        "top_degs_path": ROOT / "data" / "jiang" / "processed" / "top_degs_names.pkl",
    },
    "mcfaline": {
        "adata_path": ROOT / "data" / "mcfaline" / "processed" / "mcfaline.h5ad",
        "split_path": ROOT / "data" / "mcfaline" / "splits" / "split_dict_5Fold.pkl",
        "top_degs_path": None,
    },
}

SCENARIOS = [
    {
        "title": "Jiang24 Seen Perturbation",
        "dataset": "jiang",
        "scenario": "seen",
        "results_path": ROOT / "results" / "collected" / "jiang" / "seen_pred_results.pkl",
    },
    {
        "title": "Jiang24 Unseen Perturbation",
        "dataset": "jiang",
        "scenario": "unseen",
        "results_path": ROOT / "results" / "collected" / "jiang" / "unseen_pred_results.pkl",
    },
    {
        "title": "McFaline Seen Perturbation",
        "dataset": "mcfaline",
        "scenario": "seen",
        "results_path": ROOT / "results" / "collected" / "mcfaline" / "seen_pred_results.pkl",
    },
    {
        "title": "McFaline Unseen Perturbation",
        "dataset": "mcfaline",
        "scenario": "unseen",
        "results_path": ROOT / "results" / "collected" / "mcfaline" / "unseen_pred_results.pkl",
    },
]


def load_adata(path):
    adata = sc.read_h5ad(path)
    adata.obs["covariate"] = adata.obs["covariate"].astype(str)
    adata.obs["perturbation"] = adata.obs["perturbation"].astype(str)
    adata.obs["cov_pert"] = adata.obs["covariate"].str.cat(
        adata.obs["perturbation"], sep="_"
    )
    return adata


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_split_values(split_dict):
    if not isinstance(split_dict, dict):
        return list(split_dict)

    def sort_key(key):
        if isinstance(key, int):
            return key
        text = str(key)
        suffix = text.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else text

    return [split_dict[key] for key in sorted(split_dict, key=sort_key)]


def condition_mean_table(adata):
    labels = adata.obs["cov_pert"].astype(str)
    codes, index = pd.factorize(labels, sort=False)
    selector = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(index), len(codes)),
    )
    sums = selector @ adata.X
    if sparse.issparse(sums):
        sums = sums.toarray()
    means = np.asarray(sums) / np.bincount(codes)[:, None]
    return pd.DataFrame(means, index=pd.Index(index, name="cov_pert"), columns=adata.var_names)


def condition_mean_metadata(mean_index):
    meta = mean_index.to_series(index=mean_index).str.rsplit("_", n=1, expand=True)
    meta.columns = ["covariate", "perturbation"]
    return meta


def split_eval_map(split_dict, scenario):
    cov_pert_to_test_cov = {}
    for split in iter_split_values(split_dict):
        test_df = pd.Series(split["test"], index=split["test"]).str.rsplit(
            "_", n=1, expand=True
        )
        test_df.columns = ["covariate", "perturbation"]
        test_df["cov_unseen"] = test_df.covariate.isin(split["test_cov"])
        test_df["pert_unseen"] = test_df.perturbation.isin(split["masked_perts"])
        if scenario == "seen":
            eval_conds = test_df[test_df.cov_unseen & (~test_df.pert_unseen)]
        elif scenario == "unseen":
            eval_conds = test_df[test_df.cov_unseen & test_df.pert_unseen]
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        for cov_pert in eval_conds.index:
            cov_pert_to_test_cov[cov_pert] = split["test_cov"]
    return cov_pert_to_test_cov


def delta_average_from_means(mean_table, mean_meta, target_cov, pert, test_cov):
    train_rows = mean_meta[
        (~mean_meta.covariate.isin(test_cov)) & (mean_meta.perturbation == pert)
    ]
    deltas = []
    for cov_pert, row in train_rows.iterrows():
        ctrl_key = f"{row.covariate}_control"
        if ctrl_key not in mean_table.index:
            continue
        deltas.append(
            mean_table.loc[cov_pert].to_numpy(dtype=float)
            - mean_table.loc[ctrl_key].to_numpy(dtype=float)
        )

    target_ctrl_key = f"{target_cov}_control"
    if len(deltas) == 0 or target_ctrl_key not in mean_table.index:
        return None

    target_ctrl = mean_table.loc[target_ctrl_key].to_numpy(dtype=float)
    return target_ctrl + np.vstack(deltas).mean(axis=0)


def direction_mismatch_rate(pred, true, ctrl):
    return np.mean(np.sign(pred - ctrl) != np.sign(true - ctrl))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_dataset_inputs(dataset_name):
    config = DATASETS[dataset_name]
    adata = load_adata(config["adata_path"])
    if config["top_degs_path"] is None:
        top_degs_names = dict(adata.uns["top_degs_names"])
    else:
        top_degs_names = load_pickle(config["top_degs_path"])

    inputs = {
        "mean_table": condition_mean_table(adata),
        "top_degs_names": top_degs_names,
        "split_dict": load_pickle(config["split_path"]),
    }
    inputs["mean_meta"] = condition_mean_metadata(inputs["mean_table"].index)
    del adata
    return inputs


def compute_condition_metrics(scenario_config, dataset_inputs):
    results = load_pickle(scenario_config["results_path"])
    mean_table = dataset_inputs["mean_table"]
    mean_meta = dataset_inputs["mean_meta"]
    top_degs_names = dataset_inputs["top_degs_names"]
    split_dict = dataset_inputs["split_dict"]
    var_names = pd.Index(results["TranScouter"].columns)
    cov_pert_to_test_cov = split_eval_map(split_dict, scenario_config["scenario"])

    delta_preds = {}
    for cov_pert in tqdm(
        results["TranScouter"].index,
        desc=f"{scenario_config['title']}: delta-average",
        leave=False,
    ):
        if cov_pert not in cov_pert_to_test_cov:
            continue
        cov, pert = cov_pert.rsplit("_", 1)
        pred = delta_average_from_means(
            mean_table, mean_meta, cov, pert, cov_pert_to_test_cov[cov_pert]
        )
        if pred is not None:
            delta_preds[cov_pert] = np.clip(pred, a_min=0, a_max=None)
    delta_average = pd.DataFrame.from_dict(
        delta_preds, orient="index", columns=mean_table.columns
    )
    delta_average = delta_average.reindex(columns=var_names)

    preds = {
        "TranScouter": results["TranScouter"],
        "DeltaAverage": delta_average,
    }
    true = results["True"]
    ctrl = results["Ctrl"]

    records = []
    for cov_pert in tqdm(
        results["TranScouter"].index,
        desc=f"{scenario_config['title']}: metrics",
        leave=False,
    ):
        if cov_pert not in cov_pert_to_test_cov:
            continue
        cov, pert = cov_pert.rsplit("_", 1)
        ctrl_key = f"{cov}_control"
        if cov_pert not in true.index or ctrl_key not in ctrl.index:
            continue

        degs = [g for g in top_degs_names.get(cov_pert, []) if g in var_names]
        if len(degs) == 0:
            continue

        true_i = true.loc[cov_pert]
        ctrl_i = ctrl.loc[ctrl_key]
        for method in METHOD_ORDER:
            df = preds[method]
            if cov_pert not in df.index:
                continue
            pred_i = df.loc[cov_pert]

            true_vals = true_i[degs].to_numpy(dtype=float)
            pred_vals = pred_i[degs].to_numpy(dtype=float)
            ctrl_vals = ctrl_i[degs].to_numpy(dtype=float)
            finite = (
                np.isfinite(true_vals)
                & np.isfinite(pred_vals)
                & np.isfinite(ctrl_vals)
            )
            if finite.sum() == 0:
                continue

            records.append(
                {
                    "cov_pert": cov_pert,
                    "covariate": cov,
                    "perturbation": pert,
                    "celltype": cov.split("+", 1)[0].upper(),
                    "treatment": cov.split("+", 1)[1],
                    "method": method,
                    "MSE_degs": mse(true_vals[finite], pred_vals[finite]),
                    "DMR_degs": direction_mismatch_rate(
                        pred_vals[finite], true_vals[finite], ctrl_vals[finite]
                    ),
                }
            )

    metrics = pd.DataFrame(records)
    print(f"\n{scenario_config['title']}")
    print(metrics.groupby("method")[["MSE_degs", "DMR_degs"]].median().reindex(METHOD_ORDER))
    return metrics


def build_plot_data(scenario_metrics):
    condition_metrics = []
    panel_summaries = []
    overall_summaries = []

    for scenario_config, metrics in scenario_metrics:
        scenario = scenario_config["title"]
        metrics_out = metrics.copy()
        metrics_out.insert(0, "scenario", scenario)
        condition_metrics.append(metrics_out)

        for metric_name in ["MSE_degs", "DMR_degs"]:
            panel = (
                metrics.groupby(["celltype", "treatment", "method"], observed=False)[
                    metric_name
                ]
                .median()
                .reset_index()
                .rename(columns={metric_name: "value"})
            )
            panel.insert(0, "scenario", scenario)
            panel.insert(1, "metric", metric_name)
            panel_summaries.append(panel)

        overall = (
            metrics.groupby(["covariate", "method"], observed=False)[
                ["MSE_degs", "DMR_degs"]
            ]
            .median()
            .groupby("method")
            .mean()
            .reindex(METHOD_ORDER)
            .reset_index()
        )
        overall.insert(0, "scenario", scenario)
        overall_summaries.append(overall)

    return {
        "condition_metrics": pd.concat(condition_metrics, ignore_index=True),
        "panel_summary": pd.concat(panel_summaries, ignore_index=True),
        "overall_summary": pd.concat(overall_summaries, ignore_index=True),
        "configs": [
            {
                "title": config["title"],
                "dataset": config["dataset"],
                "scenario": config["scenario"],
                "method_order": METHOD_ORDER,
            }
            for config, _ in scenario_metrics
        ],
    }


def method_pair_table(condition_metrics, scenario, values):
    sub = condition_metrics[
        (condition_metrics.scenario == scenario)
        & (condition_metrics.method.isin(METHOD_ORDER))
    ]
    table = sub.pivot_table(
        index=[
            "scenario",
            "cov_pert",
            "covariate",
            "perturbation",
            "celltype",
            "treatment",
        ],
        columns="method",
        values=values,
        aggfunc="first",
    ).reset_index()

    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            "_".join([str(x) for x in col if str(x)]) for col in table.columns
        ]
    return table


def compute_jiang_true_delta_similarity(condition_metrics, dataset_inputs, results_by_scenario):
    mean_table = dataset_inputs["mean_table"]
    mean_meta = dataset_inputs["mean_meta"]
    top_degs_names = dataset_inputs["top_degs_names"]
    split_dict = dataset_inputs["split_dict"]
    var_names = pd.Index(mean_table.columns)

    records = []
    for scenario_config in [c for c in SCENARIOS if c["dataset"] == "jiang"]:
        scenario = scenario_config["title"]
        pair = method_pair_table(condition_metrics, scenario, "DMR_degs")
        pair = pair.dropna(subset=["TranScouter", "DeltaAverage"])
        pair["delta_minus_trans"] = pair["DeltaAverage"] - pair["TranScouter"]
        pair["delta_advantage"] = pair["TranScouter"] - pair["DeltaAverage"]
        pair["delta_better"] = pair["delta_minus_trans"] < 0

        cov_pert_to_test_cov = split_eval_map(split_dict, scenario_config["scenario"])
        results = results_by_scenario[scenario]
        true = results["True"]
        ctrl = results["Ctrl"]

        for row in tqdm(
            pair.itertuples(index=False),
            total=len(pair),
            desc=f"{scenario}: true-delta similarity",
            leave=False,
        ):
            cov_pert = row.cov_pert
            if cov_pert not in cov_pert_to_test_cov:
                continue
            cov = row.covariate
            pert = row.perturbation
            ctrl_key = f"{cov}_control"
            if cov_pert not in mean_table.index or ctrl_key not in mean_table.index:
                continue

            degs = [g for g in top_degs_names.get(cov_pert, []) if g in var_names]
            if not degs:
                continue

            cached_ctrl = ctrl.loc[ctrl_key, degs].to_numpy(dtype=float)
            target_delta = true.loc[cov_pert, degs].to_numpy(dtype=float) - cached_ctrl
            target_ctrl_for_prediction = mean_table.loc[ctrl_key, degs].to_numpy(dtype=float)
            train_rows = mean_meta[
                (~mean_meta.covariate.isin(cov_pert_to_test_cov[cov_pert]))
                & (mean_meta.perturbation == pert)
            ]

            train_deltas = []
            for train_cov_pert, train_row in train_rows.iterrows():
                train_ctrl = f"{train_row.covariate}_control"
                if train_ctrl not in mean_table.index:
                    continue
                train_deltas.append(
                    mean_table.loc[train_cov_pert, degs].to_numpy(dtype=float)
                    - mean_table.loc[train_ctrl, degs].to_numpy(dtype=float)
                )
            if not train_deltas:
                continue

            train_deltas = np.vstack(train_deltas)
            avg_train_delta = train_deltas.mean(axis=0)
            clipped_delta = np.clip(
                target_ctrl_for_prediction + avg_train_delta, a_min=0, a_max=None
            )
            clipped_delta = clipped_delta - cached_ctrl
            pair_sign = (
                np.sign(train_deltas) == np.sign(target_delta)[None, :]
            ).mean(axis=1)
            pair_corr = np.array([corr(d, target_delta) for d in train_deltas])
            finite_pair_corr = pair_corr[np.isfinite(pair_corr)]

            records.append(
                {
                    "scenario": scenario,
                    "cov_pert": cov_pert,
                    "covariate": cov,
                    "perturbation": pert,
                    "celltype": row.celltype,
                    "treatment": row.treatment,
                    "n_train_covs": len(train_deltas),
                    "avg_delta_sign_match": (
                        np.sign(avg_train_delta) == np.sign(target_delta)
                    ).mean(),
                    "clipped_delta_sign_match": (
                        np.sign(clipped_delta) == np.sign(target_delta)
                    ).mean(),
                    "mean_pair_sign_match": pair_sign.mean(),
                    "mean_pair_corr": (
                        finite_pair_corr.mean() if len(finite_pair_corr) else np.nan
                    ),
                    "TranScouter_DMR": row.TranScouter,
                    "DeltaAverage_DMR": row.DeltaAverage,
                    "delta_advantage": row.delta_advantage,
                    "delta_better": row.delta_better,
                }
            )
    return pd.DataFrame(records)


def compute_mcfaline_transferability(condition_metrics, dataset_inputs, results_by_scenario):
    mean_table = dataset_inputs["mean_table"]
    mean_meta = dataset_inputs["mean_meta"]
    top_degs_names = dataset_inputs["top_degs_names"]
    split_dict = dataset_inputs["split_dict"]
    var_names = pd.Index(mean_table.columns)

    records = []
    for scenario_config in [c for c in SCENARIOS if c["dataset"] == "mcfaline"]:
        scenario = scenario_config["title"]
        pair = method_pair_table(condition_metrics, scenario, ["DMR_degs", "MSE_degs"])
        pair = pair.rename(
            columns={
                "DMR_degs_TranScouter": "TranScouter_DMR",
                "DMR_degs_DeltaAverage": "DeltaAverage_DMR",
                "MSE_degs_TranScouter": "TranScouter_MSE",
                "MSE_degs_DeltaAverage": "DeltaAverage_MSE",
            }
        )
        pair = pair.dropna(
            subset=[
                "TranScouter_DMR",
                "DeltaAverage_DMR",
                "TranScouter_MSE",
                "DeltaAverage_MSE",
            ]
        )
        pair["delta_dmr_advantage"] = pair["TranScouter_DMR"] - pair["DeltaAverage_DMR"]
        pair["trans_dmr_advantage"] = pair["DeltaAverage_DMR"] - pair["TranScouter_DMR"]
        pair["trans_mse_advantage"] = pair["DeltaAverage_MSE"] - pair["TranScouter_MSE"]
        pair["delta_dmr_better"] = pair["DeltaAverage_DMR"] < pair["TranScouter_DMR"]
        pair["trans_dmr_better"] = pair["TranScouter_DMR"] < pair["DeltaAverage_DMR"]
        pair["delta_mse_better"] = pair["DeltaAverage_MSE"] < pair["TranScouter_MSE"]
        pair["trans_mse_better"] = pair["TranScouter_MSE"] < pair["DeltaAverage_MSE"]

        cov_pert_to_test_cov = split_eval_map(split_dict, scenario_config["scenario"])
        results = results_by_scenario[scenario]
        true = results["True"]
        ctrl = results["Ctrl"]

        for row in tqdm(
            pair.itertuples(index=False),
            total=len(pair),
            desc=f"{scenario}: transferability",
            leave=False,
        ):
            cov_pert = row.cov_pert
            if cov_pert not in cov_pert_to_test_cov:
                continue
            cov = row.covariate
            pert = row.perturbation
            ctrl_key = f"{cov}_control"
            if cov_pert not in mean_table.index or ctrl_key not in mean_table.index:
                continue

            degs = [g for g in top_degs_names.get(cov_pert, []) if g in var_names]
            if not degs:
                continue

            target_delta = (
                true.loc[cov_pert, degs].to_numpy(dtype=float)
                - ctrl.loc[ctrl_key, degs].to_numpy(dtype=float)
            )
            target_ctrl_for_prediction = mean_table.loc[ctrl_key, degs].to_numpy(dtype=float)
            train_rows = mean_meta[
                (~mean_meta.covariate.isin(cov_pert_to_test_cov[cov_pert]))
                & (mean_meta.perturbation == pert)
            ]

            train_deltas = []
            for train_cov_pert, train_row in train_rows.iterrows():
                train_ctrl = f"{train_row.covariate}_control"
                if train_ctrl not in mean_table.index:
                    continue
                train_deltas.append(
                    mean_table.loc[train_cov_pert, degs].to_numpy(dtype=float)
                    - mean_table.loc[train_ctrl, degs].to_numpy(dtype=float)
                )
            if not train_deltas:
                continue

            train_deltas = np.vstack(train_deltas)
            avg_train_delta = train_deltas.mean(axis=0)
            clipped_delta = np.clip(
                target_ctrl_for_prediction + avg_train_delta, a_min=0, a_max=None
            )
            clipped_delta = clipped_delta - ctrl.loc[ctrl_key, degs].to_numpy(dtype=float)
            pair_sign = (
                np.sign(train_deltas) == np.sign(target_delta)[None, :]
            ).mean(axis=1)
            pair_corr = np.array([corr(d, target_delta) for d in train_deltas])
            finite_pair_corr = pair_corr[np.isfinite(pair_corr)]

            records.append(
                {
                    "scenario": scenario,
                    "cov_pert": cov_pert,
                    "covariate": cov,
                    "perturbation": pert,
                    "celltype": row.celltype,
                    "treatment": row.treatment,
                    "n_train_covs": len(train_deltas),
                    "avg_delta_sign_match": (
                        np.sign(avg_train_delta) == np.sign(target_delta)
                    ).mean(),
                    "clipped_delta_sign_match": (
                        np.sign(clipped_delta) == np.sign(target_delta)
                    ).mean(),
                    "mean_pair_sign_match": pair_sign.mean(),
                    "mean_pair_corr": (
                        finite_pair_corr.mean() if len(finite_pair_corr) else np.nan
                    ),
                    "TranScouter_DMR": row.TranScouter_DMR,
                    "DeltaAverage_DMR": row.DeltaAverage_DMR,
                    "TranScouter_MSE": row.TranScouter_MSE,
                    "DeltaAverage_MSE": row.DeltaAverage_MSE,
                    "delta_dmr_advantage": row.delta_dmr_advantage,
                    "trans_dmr_advantage": row.trans_dmr_advantage,
                    "trans_mse_advantage": row.trans_mse_advantage,
                    "delta_dmr_better": row.delta_dmr_better,
                    "trans_dmr_better": row.trans_dmr_better,
                    "delta_mse_better": row.delta_mse_better,
                    "trans_mse_better": row.trans_mse_better,
                }
            )
    return pd.DataFrame(records)


def main():
    dataset_inputs = {}
    for dataset_name in DATASETS:
        print(f"Loading {dataset_name} inputs")
        dataset_inputs[dataset_name] = load_dataset_inputs(dataset_name)

    results_by_scenario = {}
    scenario_metrics = []
    for scenario_config in SCENARIOS:
        title = scenario_config["title"]
        print(f"\nProcessing {title}")
        results_by_scenario[title] = load_pickle(scenario_config["results_path"])
        metrics = compute_condition_metrics(
            scenario_config, dataset_inputs[scenario_config["dataset"]]
        )
        scenario_metrics.append((scenario_config, metrics))

    plot_data = build_plot_data(scenario_metrics)
    with open(PLOT_DATA_PATH, "wb") as f:
        pickle.dump(plot_data, f)
    print(f"\nSaved: {PLOT_DATA_PATH}")

    jiang_similarity = compute_jiang_true_delta_similarity(
        plot_data["condition_metrics"],
        dataset_inputs["jiang"],
        results_by_scenario,
    )
    jiang_similarity.to_csv(JIANG_SIMILARITY_CSV, index=False)
    print(f"Saved: {JIANG_SIMILARITY_CSV}")

    mcfaline_transferability = compute_mcfaline_transferability(
        plot_data["condition_metrics"],
        dataset_inputs["mcfaline"],
        results_by_scenario,
    )
    mcfaline_transferability.to_csv(MCFALINE_TRANSFERABILITY_CSV, index=False)
    print(f"Saved: {MCFALINE_TRANSFERABILITY_CSV}")


if __name__ == "__main__":
    main()
