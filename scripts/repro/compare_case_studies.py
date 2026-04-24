#!/usr/bin/env python3
"""Create notebook-aligned paper case-study comparisons.

This script follows the exact baseline/optimized cohort definitions used in the
paper's case-study notebooks. It complements ``summarize_case_studies.py``,
which provides broader grouped summaries over the available case-study rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMMON_METRICS = [
    "epoch_time_max",
    "compute_time_frac_epoch",
    "reader_time_frac_epoch",
    "o_reader_time_frac_self",
    "reader_posix_lustre_read_time_frac_epoch",
    "reader_posix_lustre_read_bw_mean",
    "data_loader_time_frac_epoch",
    "o_data_loader_time_frac_self",
    "checkpoint_time_frac_epoch",
    "o_checkpoint_time_frac_self",
    "checkpoint_posix_lustre_write_time_frac_epoch",
    "checkpoint_posix_ssd_write_time_frac_epoch",
]

PLOT_CASES = {
    "cosmoflow": {
        "prefix": "shap_cosmoflow_v100",
        "families": ["all", "posix"],
    },
    "unet3d": {
        "prefix": "shap_unet3d_v100",
        "families": ["all", "posix"],
    },
    "megatron_deepspeed": {
        "prefix": "shap_deepspeed",
        "families": ["all"],
    },
}

COSMO_PERF_COLS = [
    "compute_time_frac_epoch",
    "reader_posix_lustre_read_bw_mean",
    "reader_posix_lustre_read_bw_std",
    "o_epoch_time_frac_self",
    "o_fetch_data_time_frac_self",
    "o_fetch_data_time_frac_epoch",
    "o_data_loader_time_frac_self",
    "o_reader_time_frac_self",
    "u_data_loader_time_frac_self",
    "u_reader_time_frac_self",
    "u_reader_posix_lustre_time_frac_self",
    "config_num_nodes",
    "config_num_processes",
    "config_reader_read_threads",
]

COSMO_TIME_COLS = [
    "epoch_time_max",
    "compute_time_max",
    "fetch_data_time_max",
    "data_loader_time_max",
    "reader_time_max",
    "reader_posix_lustre_time_max",
    "reader_posix_lustre_data_time_max",
    "reader_posix_lustre_metadata_time_max",
    "reader_posix_lustre_open_time_max",
    "reader_posix_lustre_seek_time_max",
    "reader_posix_lustre_stat_time_max",
    "o_epoch_time_max",
    "o_fetch_data_time_max",
    "o_reader_time_max",
    "u_reader_time_max",
    "u_reader_posix_lustre_time_max",
]

COSMO_THREAD_ROWS = [
    {"metric": "compute_time_frac_epoch", "label": "Compute Utilization", "unit": "%", "higher_is_better": True},
    {"metric": "epoch_time_max", "label": "Training Time", "unit": "s"},
    {"metric": "o_fetch_data_time_max", "label": "Fetch Data Overhead Time", "unit": "s"},
    {"metric": "reader_time_max", "label": "Reader Time", "unit": "s"},
    {"metric": "reader_posix_lustre_time_max", "label": "Reader POSIX Time", "unit": "s"},
    {
        "metric": "u_reader_posix_lustre_time_frac_self",
        "label": "Unovlp. Reader POSIX Time",
        "unit": "%",
    },
    {
        "metric": "reader_posix_lustre_read_bw_mean",
        "label": "Reader POSIX BW",
        "unit": "GB/s",
        "higher_is_better": True,
    },
]

COSMO_FILE_ROWS = [
    {"metric": "compute_time_frac_epoch", "label": "Compute Utilization", "unit": "%", "higher_is_better": True},
    {"metric": "epoch_time_max", "label": "Training Time", "unit": "s"},
    {"metric": "o_fetch_data_time_max", "label": "Fetch Data Overhead Time", "unit": "s"},
    {"metric": "reader_time_max", "label": "Reader Time", "unit": "s"},
    {"metric": "reader_posix_lustre_time_max", "label": "Reader POSIX Time", "unit": "s"},
    {
        "metric": "reader_posix_lustre_read_bw_mean",
        "label": "Reader POSIX BW",
        "unit": "GB/s",
        "higher_is_better": True,
    },
]

COSMO_FILE_PREPROCESS_SECONDS = 521.0

UNET_PERF_COLS = [
    "compute_time_frac_epoch",
    "reader_posix_lustre_read_bw_mean",
    "reader_posix_lustre_read_bw_std",
    "fetch_data_time_frac_parent",
    "data_loader_fork_count_mean",
    "data_loader_fork_ops_mean",
    "data_loader_time_frac_parent",
    "data_loader_item_count_mean",
    "data_loader_item_ops_mean",
    "reader_time_frac_parent",
    "o_epoch_time_frac_self",
    "o_fetch_data_time_frac_self",
    "o_data_loader_time_frac_self",
    "o_reader_time_frac_self",
    "config_num_nodes",
    "config_num_processes",
    "config_reader_read_threads",
]

UNET_TIME_COLS = [
    "epoch_time_max",
    "compute_time_max",
    "fetch_data_time_max",
    "data_loader_time_max",
    "data_loader_fork_time_max",
    "data_loader_init_time_max",
    "reader_time_max",
    "reader_posix_lustre_time_max",
    "reader_posix_lustre_data_time_max",
    "reader_posix_lustre_metadata_time_max",
    "reader_posix_lustre_open_time_max",
    "reader_posix_lustre_seek_time_max",
    "reader_posix_lustre_stat_time_max",
    "checkpoint_time_max",
    "o_fetch_data_time_max",
    "o_data_loader_time_max",
    "o_reader_time_max",
    "u_data_loader_time_max",
    "u_data_loader_fork_time_max",
    "u_reader_time_max",
    "u_reader_posix_lustre_time_max",
]

UNET_ROWS = [
    {"metric": "compute_time_frac_epoch", "label": "Compute Utilization", "unit": "%", "higher_is_better": True},
    {"metric": "epoch_time_max", "label": "Training Time", "unit": "s"},
    {"metric": "fetch_data_time_max", "label": "Fetch Data Time", "unit": "s"},
    {"metric": "data_loader_time_max", "label": "Data Loader Time", "unit": "s"},
    {"metric": "reader_time_max", "label": "Reader Time", "unit": "s"},
    {"metric": "reader_posix_lustre_time_max", "label": "Reader POSIX Time", "unit": "s"},
]

DEEPSPEED_PERF_COLS = [
    "compute_time_frac_epoch",
    "checkpoint_posix_lustre_write_bw_mean",
    "checkpoint_posix_lustre_write_bw_std",
    "checkpoint_posix_ssd_write_bw_mean",
    "checkpoint_posix_ssd_write_bw_std",
    "checkpoint_time_frac_epoch",
    "o_epoch_time_frac_self",
    "o_fetch_data_time_frac_self",
    "o_fetch_data_time_frac_epoch",
    "o_data_loader_time_frac_self",
    "o_reader_time_frac_self",
    "u_data_loader_time_frac_self",
    "u_reader_time_frac_self",
    "u_reader_posix_lustre_time_frac_self",
    "o_checkpoint_time_frac_self",
    "u_checkpoint_posix_lustre_time_frac_self",
    "config_num_nodes",
    "config_reader_read_threads",
]

DEEPSPEED_TIME_COLS = [
    "epoch_time_max",
    "compute_time_max",
    "fetch_data_time_max",
    "data_loader_time_max",
    "reader_time_max",
    "reader_posix_lustre_time_max",
    "checkpoint_time_max",
    "checkpoint_time_std",
    "checkpoint_time_frac_epoch",
    "checkpoint_posix_lustre_time_max",
    "checkpoint_posix_lustre_read_time_max",
    "checkpoint_posix_lustre_write_time_max",
    "checkpoint_posix_lustre_sync_time_max",
    "checkpoint_posix_ssd_time_max",
    "checkpoint_posix_ssd_read_time_max",
    "checkpoint_posix_ssd_write_time_max",
    "o_fetch_data_time_max",
    "o_data_loader_time_max",
    "o_checkpoint_time_max",
    "u_data_loader_time_max",
    "u_reader_time_max",
    "u_reader_posix_lustre_time_max",
    "u_checkpoint_posix_lustre_time_max",
    "u_checkpoint_posix_ssd_time_max",
]

DEEPSPEED_ROWS = [
    {"metric": "compute_time_frac_epoch", "label": "Compute Utilization", "unit": "%", "higher_is_better": True},
    {"metric": "epoch_time_max", "label": "Training Time", "unit": "s"},
    {"metric": "fetch_data_time_max", "label": "Fetch Data Time", "unit": "s"},
    {"metric": "checkpoint_time_max", "label": "Checkpoint (Ckpt.) Time", "unit": "s"},
    {"metric": "checkpoint_posix_lustre_time_max", "label": "Ckpt. POSIX Time (Lustre)", "unit": "s"},
    {"metric": "checkpoint_posix_ssd_time_max", "label": "Ckpt. POSIX Time (SSD)", "unit": "s"},
]


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def load_workload_runs(globals_dir: Path) -> pd.DataFrame:
    path = globals_dir / "ml_workload_all.parquet"
    if not path.exists():
        raise SystemExit(f"Required workload parquet not found: {path}")
    return pd.read_parquet(path)


def load_epoch_data(globals_dir: Path, framework: str) -> pd.DataFrame:
    suffix = {
        "all": "ml_data_all.parquet",
        "pytorch": "ml_data_pytorch_all.parquet",
        "tensorflow": "ml_data_tensorflow_all.parquet",
    }[framework]
    path = globals_dir / "epoch" / suffix
    if not path.exists():
        raise SystemExit(f"Required input parquet not found: {path}")
    return pd.read_parquet(path)


def load_case_data(globals_dir: Path, framework: str, run_pattern: str) -> pd.DataFrame:
    runs = load_workload_runs(globals_dir)
    data = load_epoch_data(globals_dir, framework)
    case_df = data[data["run_id"].str.contains(run_pattern, case=False, na=False)].copy()
    run_config_cols = [col for col in runs.columns if col.startswith("config_")]
    data_config_cols = [col for col in case_df.columns if col.startswith("config_")]
    merged = case_df.drop(columns=data_config_cols, errors="ignore").merge(
        runs[run_config_cols], left_on="run_id", right_index=True
    )
    return add_dataset_format(merged)


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def add_dataset_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dataset_format = pd.Series("hdf5", index=df.index, dtype="object")
    dataset_format[bool_col(df, "config_dataset_format_tfrecord")] = "tfrecord"
    dataset_format[bool_col(df, "config_dataset_format_mmap_indexed_binary")] = "mmap_indexed_binary"
    dataset_format[bool_col(df, "config_dataset_format_npz")] = "npz"
    df["dataset_format"] = dataset_format
    return df


def existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def filter_eq_if_present(df: pd.DataFrame, col: str, value: object) -> pd.DataFrame:
    if col not in df.columns:
        return df
    filtered = df[df[col].eq(value)]
    return filtered if not filtered.empty else df


def summarize_grouped(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        raise SystemExit("No rows matched the requested case-study filters.")
    group_cols = existing(df, group_cols)
    metric_cols = existing(df, metrics)

    agg_spec = {
        "epoch_row_count": ("run_id", "size"),
        "run_count": ("run_id", "nunique"),
    }
    for metric in metric_cols:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_sem"] = (
            metric,
            lambda s: float(s.std() / np.sqrt(s.count())) if s.count() > 1 else 0.0,
        )

    out = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    out = out.sort_values(group_cols).reset_index(drop=True)
    if "epoch_time_max_mean" in out.columns:
        baseline = out["epoch_time_max_mean"].max()
        out["speedup_vs_slowest_epoch_mean"] = baseline / out["epoch_time_max_mean"]
    return out


def write_grouped_outputs(case: str, summary: pd.DataFrame, output_dir: Path, label_cols: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{case}_summary.csv"
    summary.to_csv(csv_path, index=False)
    outputs = [str(csv_path)]
    plot_path = output_dir / f"{case}_epoch_time_summary.png"
    if write_grouped_bar_plot(summary, plot_path, label_cols, "epoch_time_max_mean", f"{case} epoch-time summary"):
        outputs.append(str(plot_path))
    return outputs


def write_grouped_bar_plot(summary: pd.DataFrame, path: Path, label_cols: list[str], y_col: str, title: str) -> bool:
    if y_col not in summary.columns:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot {path}: {exc}")
        return False

    plot_df = summary.copy()
    label_cols = [col for col in label_cols if col in plot_df.columns]
    plot_df["label"] = plot_df[label_cols].astype(str).agg(" / ".join, axis=1)
    plot_df = plot_df.sort_values(y_col, ascending=False).head(40)

    fig_height = max(4, min(18, 0.32 * len(plot_df) + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(plot_df["label"], plot_df[y_col])
    ax.invert_yaxis()
    ax.set_xlabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def filter_rows(
    df: pd.DataFrame,
    *,
    equals: dict[str, object] | None = None,
    not_equals: dict[str, object] | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in (equals or {}).items():
        if col in df.columns:
            mask &= df[col].eq(value)
    for col, value in (not_equals or {}).items():
        if col in df.columns:
            mask &= df[col].ne(value)
    return df[mask].copy()


def shared_columns(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in baseline_df.columns and col in optimized_df.columns]


def aggregate_notebook_comparison(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    perf_cols: list[str],
    time_cols: list[str],
) -> pd.DataFrame:
    if baseline_df.empty or optimized_df.empty:
        raise SystemExit("No rows matched one of the requested case-study cohorts.")

    perf_cols = shared_columns(baseline_df, optimized_df, perf_cols)
    time_cols = shared_columns(baseline_df, optimized_df, time_cols)
    comparison = pd.DataFrame(
        {
            "baseline": baseline_df[perf_cols].replace(0, np.nan).mean(),
            "optimized": optimized_df[perf_cols].replace(0, np.nan).mean(),
        }
    )
    for metric in time_cols:
        comparison.loc[metric] = {
            "baseline": baseline_df.groupby("run_id")[metric].sum().mean(),
            "optimized": optimized_df.groupby("run_id")[metric].sum().mean(),
        }
    return comparison.fillna(0.0)


def scale_value(value: float, unit: str) -> float:
    if unit == "%":
        return float(value) * 100.0
    if unit == "GB/s":
        return float(value) / (1024 ** 3)
    return float(value)


def compute_improvement(baseline_value: float, optimized_value: float, higher_is_better: bool) -> tuple[float, float]:
    if higher_is_better:
        if baseline_value == 0:
            return float("nan"), float("nan")
        return optimized_value / baseline_value, ((optimized_value - baseline_value) / baseline_value) * 100.0
    if baseline_value == 0 or optimized_value == 0:
        return float("nan"), float("nan")
    return baseline_value / optimized_value, ((baseline_value - optimized_value) / baseline_value) * 100.0


def build_metric_rows(
    comparison_name: str,
    baseline_label: str,
    optimized_label: str,
    comparison_df: pd.DataFrame,
    row_specs: list[dict[str, object]],
    baseline_run_count: int,
    optimized_run_count: int,
    note: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in row_specs:
        metric = str(spec["metric"])
        if metric not in comparison_df.index:
            continue
        higher_is_better = bool(spec.get("higher_is_better", False))
        unit = str(spec["unit"])
        baseline_raw = float(comparison_df.loc[metric, "baseline"])
        optimized_raw = float(comparison_df.loc[metric, "optimized"])
        baseline_value = scale_value(baseline_raw, unit)
        optimized_value = scale_value(optimized_raw, unit)
        ratio, percent_improvement = compute_improvement(
            baseline_raw if unit not in {"%", "GB/s"} else baseline_value,
            optimized_raw if unit not in {"%", "GB/s"} else optimized_value,
            higher_is_better=higher_is_better,
        )
        rows.append(
            {
                "comparison": comparison_name,
                "metric": metric,
                "metric_label": str(spec["label"]),
                "unit": unit,
                "baseline_label": baseline_label,
                "optimized_label": optimized_label,
                "baseline_value": baseline_value,
                "optimized_value": optimized_value,
                "ratio": ratio,
                "percent_improvement": percent_improvement,
                "baseline_run_count": baseline_run_count,
                "optimized_run_count": optimized_run_count,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def build_run_inventory(
    comparison_name: str,
    baseline_label: str,
    optimized_label: str,
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for run_id in sorted(baseline_df["run_id"].unique()):
        rows.append(
            {
                "comparison": comparison_name,
                "cohort": baseline_label,
                "run_id": run_id,
            }
        )
    for run_id in sorted(optimized_df["run_id"].unique()):
        rows.append(
            {
                "comparison": comparison_name,
                "cohort": optimized_label,
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def write_comparison_outputs(case: str, summary: pd.DataFrame, inventory: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{case}_comparison.csv"
    summary.to_csv(summary_path, index=False)
    outputs = [str(summary_path)]

    inventory_path = output_dir / f"{case}_comparison_run_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    outputs.append(str(inventory_path))

    plot_path = output_dir / f"{case}_comparison_epoch_time_summary.png"
    if write_comparison_bar_plot(summary, plot_path, f"{case} training-time summary"):
        outputs.append(str(plot_path))
    return outputs


def write_comparison_bar_plot(summary: pd.DataFrame, path: Path, title: str) -> bool:
    plot_df = summary[summary["metric"].eq("epoch_time_max")].copy()
    if plot_df.empty:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot {path}: {exc}")
        return False

    plot_df = plot_df.reset_index(drop=True)
    y = np.arange(len(plot_df))
    height = 0.34

    fig_height = max(4, min(10, 1.0 + 0.9 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(y - height / 2, plot_df["baseline_value"], height=height, label="baseline")
    ax.barh(y + height / 2, plot_df["optimized_value"], height=height, label="optimized")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["comparison"])
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def diagnostic_plot_inventory(case: str, results_dir: Path, output_dir: Path) -> list[str]:
    if case not in PLOT_CASES:
        return []
    spec = PLOT_CASES[case]
    rows = []
    for family in spec["families"]:
        for quality in ("good", "bad"):
            for target in ("center", "width"):
                path = results_dir / "plots" / f"{spec['prefix']}_{family}_{quality}_{target}.png"
                rows.append(
                    {
                        "case": case,
                        "plot_family": family,
                        "quality": quality,
                        "target": target,
                        "path": str(path),
                        "exists": path.exists(),
                        "size_bytes": path.stat().st_size if path.exists() else 0,
                    }
                )
    inventory = pd.DataFrame(rows)
    output_path = output_dir / f"{case}_comparison_diagnostic_plot_inventory.csv"
    inventory.to_csv(output_path, index=False)
    return [str(output_path)]


def summarize_motivation(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    del results_dir
    df = load_case_data(globals_dir, "pytorch", "unet3d_v100")
    case_df = filter_eq_if_present(df, "config_dataset_num_files_train", 168)
    case_df = filter_eq_if_present(case_df, "config_train_epochs", 5)
    summary = summarize_grouped(
        case_df,
        [
            "dataset_format",
            "config_reader_read_threads",
            "config_dataset_num_files_train",
            "config_train_epochs",
        ],
        COMMON_METRICS,
    )
    return write_grouped_outputs("motivation", summary, output_dir, ["dataset_format", "config_reader_read_threads"])


def summarize_cosmoflow(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_case_data(globals_dir, "tensorflow", "cosmoflow")
    baseline = {
        "config_num_nodes": 4,
        "config_num_processes": 4,
        "config_reader_read_threads": 1,
        "config_dataset_num_files_train": 524288,
    }
    reader_threads_opt = {
        "config_num_nodes": 4,
        "config_num_processes": 4,
        "config_reader_read_threads": 4,
        "config_dataset_num_files_train": 524288,
    }
    file_layout_opt = {
        "config_num_nodes": 4,
        "config_num_processes": 4,
        "config_reader_read_threads": 1,
        "config_dataset_num_files_train": 5242,
    }

    baseline_df = filter_rows(df, equals=baseline)
    reader_threads_df = filter_rows(df, equals=reader_threads_opt)
    file_layout_df = filter_rows(df, equals=file_layout_opt)

    thread_cmp = aggregate_notebook_comparison(baseline_df, reader_threads_df, COSMO_PERF_COLS, COSMO_TIME_COLS)
    file_cmp = aggregate_notebook_comparison(baseline_df, file_layout_df, COSMO_PERF_COLS, COSMO_TIME_COLS)

    thread_rows = build_metric_rows(
        "default_layout_reader_threads",
        "baseline: 524288 files / 1 reader",
        "optimized: 524288 files / 4 readers",
        thread_cmp,
        COSMO_THREAD_ROWS,
        baseline_df["run_id"].nunique(),
        reader_threads_df["run_id"].nunique(),
    )
    file_rows = build_metric_rows(
        "dataset_relayout",
        "baseline: 524288 files / 1 reader",
        "optimized: 5242 files / 1 reader",
        file_cmp,
        COSMO_FILE_ROWS,
        baseline_df["run_id"].nunique(),
        file_layout_df["run_id"].nunique(),
        note="Paper prose additionally cites a 521 s dataset-relayout preprocessing cost.",
    )

    baseline_time = float(file_cmp.loc["epoch_time_max", "baseline"])
    optimized_time = float(file_cmp.loc["epoch_time_max", "optimized"]) + COSMO_FILE_PREPROCESS_SECONDS
    ratio, percent_improvement = compute_improvement(baseline_time, optimized_time, higher_is_better=False)
    file_rows = pd.concat(
        [
            file_rows,
            pd.DataFrame(
                [
                    {
                        "comparison": "dataset_relayout",
                        "metric": "epoch_time_with_preprocessing",
                        "metric_label": "Training Time + Preprocessing",
                        "unit": "s",
                        "baseline_label": "baseline: 524288 files / 1 reader",
                        "optimized_label": "optimized: 5242 files / 1 reader",
                        "baseline_value": baseline_time,
                        "optimized_value": optimized_time,
                        "ratio": ratio,
                        "percent_improvement": percent_improvement,
                        "baseline_run_count": baseline_df["run_id"].nunique(),
                        "optimized_run_count": file_layout_df["run_id"].nunique(),
                        "note": "Includes the 521 s dataset-relayout preprocessing cost cited in the paper.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    summary = pd.concat([thread_rows, file_rows], ignore_index=True)
    inventory = pd.concat(
        [
            build_run_inventory(
                "default_layout_reader_threads",
                "baseline: 524288 files / 1 reader",
                "optimized: 524288 files / 4 readers",
                baseline_df,
                reader_threads_df,
            ),
            build_run_inventory(
                "dataset_relayout",
                "baseline: 524288 files / 1 reader",
                "optimized: 5242 files / 1 reader",
                baseline_df,
                file_layout_df,
            ),
        ],
        ignore_index=True,
    )

    outputs = write_comparison_outputs("cosmoflow", summary, inventory, output_dir)
    outputs.extend(diagnostic_plot_inventory("cosmoflow", results_dir, output_dir))
    return outputs


def summarize_unet3d(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_case_data(globals_dir, "pytorch", "unet3d_v100")
    base_filters = {
        "config_num_nodes": 1,
        "config_reader_read_threads": 1,
        "config_dataset_format_npz": True,
        "config_train_epochs": 10,
        "config_dataset_num_files_train": 168,
        "config_reader_prefetch_workers": False,
    }
    opt_filters = dict(base_filters)
    opt_filters["config_reader_prefetch_workers"] = True
    if "config_reader_batch_size" in df.columns:
        base_filters["config_reader_batch_size"] = 4
        opt_filters["config_reader_batch_size"] = 4

    baseline_df = filter_rows(df, equals=base_filters)
    optimized_df = filter_rows(df, equals=opt_filters)
    comparison = aggregate_notebook_comparison(baseline_df, optimized_df, UNET_PERF_COLS, UNET_TIME_COLS)
    summary = build_metric_rows(
        "npz_prefetch_workers",
        "baseline: npz / no prefetch",
        "optimized: npz / prefetch",
        comparison,
        UNET_ROWS,
        baseline_df["run_id"].nunique(),
        optimized_df["run_id"].nunique(),
    )
    inventory = build_run_inventory(
        "npz_prefetch_workers",
        "baseline: npz / no prefetch",
        "optimized: npz / prefetch",
        baseline_df,
        optimized_df,
    )
    outputs = write_comparison_outputs("unet3d", summary, inventory, output_dir)
    outputs.extend(diagnostic_plot_inventory("unet3d", results_dir, output_dir))
    return outputs


def summarize_megatron_deepspeed(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_case_data(globals_dir, "pytorch", "deepspeed")
    baseline_filters = {
        "config_num_nodes": 32,
        "config_train_computation_time": 2.44,
        "config_train_epochs": 5,
        "config_checkpoint_scr": False,
    }
    optimized_filters = {
        "config_num_nodes": 32,
        "config_train_computation_time": 2.44,
        "config_train_epochs": 5,
        "config_checkpoint_scr": True,
        "config_checkpoint_scr_flush_async": True,
    }
    baseline_df = filter_rows(df, equals=baseline_filters)
    optimized_df = filter_rows(df, equals=optimized_filters)
    comparison = aggregate_notebook_comparison(baseline_df, optimized_df, DEEPSPEED_PERF_COLS, DEEPSPEED_TIME_COLS)
    summary = build_metric_rows(
        "checkpoint_scr_async",
        "baseline: Lustre checkpoint",
        "optimized: SCR async checkpoint",
        comparison,
        DEEPSPEED_ROWS,
        baseline_df["run_id"].nunique(),
        optimized_df["run_id"].nunique(),
    )
    inventory = build_run_inventory(
        "checkpoint_scr_async",
        "baseline: Lustre checkpoint",
        "optimized: SCR async checkpoint",
        baseline_df,
        optimized_df,
    )
    outputs = write_comparison_outputs("megatron_deepspeed", summary, inventory, output_dir)
    outputs.extend(diagnostic_plot_inventory("megatron_deepspeed", results_dir, output_dir))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--globals-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--case",
        choices=["all", "motivation", "cosmoflow", "unet3d", "megatron_deepspeed"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    globals_dir = resolve_path(args.globals_dir or Path("globals"), repo_root)
    results_dir = resolve_path(args.results_dir or Path("results"), repo_root)
    output_dir = resolve_path(args.output_dir or Path("results/repro_compare"), repo_root)

    selected = ["motivation", "cosmoflow", "unet3d", "megatron_deepspeed"] if args.case == "all" else [args.case]
    runners = {
        "motivation": summarize_motivation,
        "cosmoflow": summarize_cosmoflow,
        "unet3d": summarize_unet3d,
        "megatron_deepspeed": summarize_megatron_deepspeed,
    }
    outputs: dict[str, list[str]] = {}
    for case in selected:
        outputs[case] = runners[case](globals_dir, results_dir, output_dir)

    manifest_path = output_dir / "case_study_comparison_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
