#!/usr/bin/env python3
"""Create generic grouped summaries for case-study workloads.

This script provides a broad reviewer-facing summary over the available case
study rows in the packaged parquet files. It does not try to recreate the
paper's exact baseline/optimized pairings; that path lives in
``compare_case_studies.py``.
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


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def load_epoch_data(globals_dir: Path, framework: str) -> pd.DataFrame:
    suffix = {
        "all": "ml_data_all.parquet",
        "pytorch": "ml_data_pytorch_all.parquet",
        "tensorflow": "ml_data_tensorflow_all.parquet",
    }[framework]
    path = globals_dir / "epoch" / suffix
    if not path.exists():
        raise SystemExit(f"Required input parquet not found: {path}")
    df = pd.read_parquet(path)
    return add_dataset_format(df)


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


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
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


def write_outputs(case: str, summary: pd.DataFrame, output_dir: Path, label_cols: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{case}_summary.csv"
    summary.to_csv(csv_path, index=False)
    outputs = [str(csv_path)]
    plot_path = output_dir / f"{case}_epoch_time_summary.png"
    if write_bar_plot(summary, plot_path, label_cols, "epoch_time_max_mean", f"{case} epoch-time summary"):
        outputs.append(str(plot_path))
    return outputs


def write_bar_plot(summary: pd.DataFrame, path: Path, label_cols: list[str], y_col: str, title: str) -> bool:
    if y_col not in summary.columns:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency path
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
    output_path = output_dir / f"{case}_diagnostic_plot_inventory.csv"
    inventory.to_csv(output_path, index=False)
    return [str(output_path)]


def summarize_motivation(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_epoch_data(globals_dir, "pytorch")
    case_df = df[df["workload_name"].eq("unet3d_v100")]
    case_df = filter_eq_if_present(case_df, "config_dataset_num_files_train", 168)
    case_df = filter_eq_if_present(case_df, "config_train_epochs", 5)
    summary = summarize(
        case_df,
        [
            "dataset_format",
            "config_reader_read_threads",
            "config_dataset_num_files_train",
            "config_train_epochs",
        ],
        COMMON_METRICS,
    )
    return write_outputs("motivation", summary, output_dir, ["dataset_format", "config_reader_read_threads"])


def summarize_cosmoflow(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_epoch_data(globals_dir, "tensorflow")
    case_df = df[df["workload_name"].eq("cosmoflow_v100")]
    case_df = filter_eq_if_present(case_df, "config_train_epochs", 5)
    summary = summarize(
        case_df,
        [
            "config_num_nodes",
            "config_reader_read_threads",
            "config_dataset_num_files_train",
            "config_train_epochs",
        ],
        COMMON_METRICS,
    )
    outputs = write_outputs(
        "cosmoflow",
        summary,
        output_dir,
        ["config_num_nodes", "config_reader_read_threads", "config_dataset_num_files_train"],
    )
    outputs.extend(diagnostic_plot_inventory("cosmoflow", results_dir, output_dir))
    return outputs


def summarize_unet3d(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_epoch_data(globals_dir, "pytorch")
    case_df = df[df["workload_name"].eq("unet3d_v100")]
    case_df = filter_eq_if_present(case_df, "config_train_epochs", 5)
    summary = summarize(
        case_df,
        [
            "dataset_format",
            "config_reader_read_threads",
            "config_dataset_num_files_train",
            "config_reader_prefetch_workers",
            "config_train_epochs",
        ],
        COMMON_METRICS,
    )
    outputs = write_outputs(
        "unet3d",
        summary,
        output_dir,
        ["dataset_format", "config_reader_read_threads", "config_dataset_num_files_train"],
    )
    outputs.extend(diagnostic_plot_inventory("unet3d", results_dir, output_dir))
    return outputs


def summarize_megatron_deepspeed(globals_dir: Path, results_dir: Path, output_dir: Path) -> list[str]:
    df = load_epoch_data(globals_dir, "pytorch")
    case_df = df[df["workload_name"].str.contains("deepspeed", case=False, na=False)]
    summary = summarize(
        case_df,
        [
            "workload_name",
            "config_num_nodes",
            "config_train_computation_time",
            "config_checkpoint_scr",
            "config_checkpoint_scr_flush",
            "config_checkpoint_scr_flush_async",
            "config_train_epochs",
        ],
        COMMON_METRICS,
    )
    outputs = write_outputs(
        "megatron_deepspeed",
        summary,
        output_dir,
        [
            "config_num_nodes",
            "config_train_computation_time",
            "config_checkpoint_scr",
            "config_checkpoint_scr_flush_async",
        ],
    )
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
    output_dir = resolve_path(args.output_dir or Path("results/repro"), repo_root)

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

    manifest_path = output_dir / "case_study_summary_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
