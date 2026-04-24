#!/usr/bin/env python3
"""Build the reviewer-facing DLIO full sweep configuration.

Created for artifact reviewers: this converts the postprocessed workload
metadata into an explicit Flux-runner CSV with one row per runnable
configuration and a repeat count.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


SCR_CHECKPOINT_MECHANISM = (
    "dlio_benchmark.checkpointing.scr_pytorch_checkpointing.SCRPyTorchCheckpointing"
)

PLAN_COLUMNS = [
    "plan_component",
    "workload_group",
    "workload_name",
    "observed_workload_names",
    "source_versions",
    "node_values",
    "ppn",
    "reader_threads_values",
    "reader_batch_size",
    "reader_prefetch_workers",
    "reader_transfer_size",
    "dataset_format_values",
    "num_files_values",
    "num_samples_values",
    "record_length_bytes",
    "checkpoint_steps",
    "checkpoint_mechanism",
    "scr_cache_size",
    "scr_file_buf_size",
    "scr_flush",
    "scr_flush_async",
    "train_epochs",
    "total_training_steps",
    "chunk_size",
    "computation_time",
    "repeat_count",
]

GROUP_COLS = [
    "version",
    "workload_name",
    "config_num_nodes",
    "config_num_processes",
    "config_reader_read_threads",
    "config_reader_batch_size",
    "config_reader_prefetch_workers",
    "config_reader_transfer_size",
    "dataset_format",
    "config_dataset_num_files_train",
    "config_dataset_num_samples_per_file",
    "config_dataset_record_length_bytes",
    "config_train_epochs",
    "config_train_computation_time",
    "config_checkpoint_scr",
    "config_checkpoint_scr_cache_size",
    "config_checkpoint_scr_file_buf_size",
    "config_checkpoint_scr_flush",
    "config_checkpoint_scr_flush_async",
]


def bool_string(value: object) -> str:
    return "true" if bool(value) else "false"


def value_string(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return bool_string(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def dataset_format(row: pd.Series) -> str:
    if bool(row["config_dataset_format_tfrecord"]):
        return "tfrecord"
    if bool(row["config_dataset_format_mmap_indexed_binary"]):
        return "mmap_indexed_binary"
    if bool(row["config_dataset_format_npz"]):
        return "npz"
    return "hdf5"


def runnable_workload_name(name: str) -> str:
    if name.lower() == "megatron_deepspeed_llnl":
        return "megatron_deepspeed_LLNL"
    return name


def workload_group(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("bert"):
        return "bert"
    if lowered.startswith("cosmoflow"):
        return "cosmoflow"
    if lowered.startswith("resnet50"):
        return "resnet50"
    if lowered.startswith("unet3d"):
        return "unet3d"
    if lowered.startswith("llama_70b"):
        return "llama_70b"
    if lowered.startswith("llama_8b"):
        return "llama_8b"
    if lowered.startswith("llama_7b"):
        return "llama_7b"
    if "megatron_deepspeed" in lowered:
        return "megatron_deepspeed"
    return lowered


def plan_component(name: str, source_versions: list[str]) -> str:
    lowered = name.lower()
    if "custom" not in source_versions:
        return "source_scale_ci_style"
    if lowered.startswith(("bert", "resnet50")):
        return "custom_bert_resnet_model_sweeps"
    if lowered.startswith("cosmoflow"):
        return "custom_cosmoflow_case_study"
    if lowered.startswith("unet3d"):
        return "custom_unet3d_case_study"
    if lowered.startswith("llama"):
        return "custom_llama_checkpoint_probes"
    if "megatron_deepspeed" in lowered:
        return "custom_megatron_deepspeed_case_study"
    return "custom_targeted_measurement"


def checkpoint_steps(name: str) -> str:
    if "megatron_deepspeed" in name.lower():
        return "100"
    return ""


def total_training_steps(name: str) -> str:
    if "megatron_deepspeed" in name.lower():
        return "1000"
    return ""


def checkpoint_mechanism(scr_enabled: object) -> str:
    return SCR_CHECKPOINT_MECHANISM if bool(scr_enabled) else ""


def build_run_plan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["observed_workload_name"] = df["workload_name"]
    df["workload_name"] = df["workload_name"].map(runnable_workload_name)
    df["dataset_format"] = df.apply(dataset_format, axis=1)

    rows = []
    grouped = df.groupby(GROUP_COLS, dropna=False, sort=True)
    for keys, group in grouped:
        key = dict(zip(GROUP_COLS, keys))
        source_versions = sorted(group["version"].dropna().astype(str).unique())
        observed_names = sorted(group["observed_workload_name"].dropna().astype(str).unique())
        workload_name = key["workload_name"]
        scr_enabled = key["config_checkpoint_scr"]
        rows.append(
            {
                "plan_component": plan_component(workload_name, source_versions),
                "workload_group": workload_group(workload_name),
                "workload_name": workload_name,
                "observed_workload_names": ";".join(observed_names),
                "source_versions": ";".join(source_versions),
                "node_values": value_string(key["config_num_nodes"]),
                "ppn": value_string(key["config_num_processes"]),
                "reader_threads_values": value_string(key["config_reader_read_threads"]),
                "reader_batch_size": value_string(key["config_reader_batch_size"]),
                "reader_prefetch_workers": value_string(key["config_reader_prefetch_workers"]),
                "reader_transfer_size": value_string(key["config_reader_transfer_size"]),
                "dataset_format_values": key["dataset_format"],
                "num_files_values": value_string(key["config_dataset_num_files_train"]),
                "num_samples_values": value_string(key["config_dataset_num_samples_per_file"]),
                "record_length_bytes": value_string(key["config_dataset_record_length_bytes"]),
                "checkpoint_steps": checkpoint_steps(workload_name),
                "checkpoint_mechanism": checkpoint_mechanism(scr_enabled),
                "scr_cache_size": value_string(key["config_checkpoint_scr_cache_size"])
                if bool(scr_enabled)
                else "",
                "scr_file_buf_size": value_string(key["config_checkpoint_scr_file_buf_size"])
                if bool(scr_enabled)
                else "",
                "scr_flush": "1" if bool(key["config_checkpoint_scr_flush"]) else "",
                "scr_flush_async": "1" if bool(key["config_checkpoint_scr_flush_async"]) else "",
                "train_epochs": value_string(key["config_train_epochs"]),
                "total_training_steps": total_training_steps(workload_name),
                "chunk_size": "",
                "computation_time": value_string(key["config_train_computation_time"]),
                "repeat_count": len(group),
            }
        )

    return pd.DataFrame(rows, columns=PLAN_COLUMNS).sort_values(PLAN_COLUMNS[:-1]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "component": "source_scale_ci_style",
            "selection": "version in dev6/dev7",
            "measurement_count": int(df[df["version"].isin(["dev6", "dev7"])].shape[0]),
            "purpose": "One-epoch source-scale validation runs with fixed PPN and reader settings.",
            "repeat_rationale": "The dev6/dev7 collections repeat the source-scale sweep across two trace-generation versions.",
        },
        {
            "component": "custom_bert_resnet_model_sweeps",
            "selection": "custom BERT and ResNet-50",
            "measurement_count": int(
                df[
                    (df["version"].eq("custom"))
                    & (df["workload_name"].str.startswith(("bert", "resnet50")))
                ].shape[0]
            ),
            "purpose": "Targeted reader-thread, file-count, sample-count, PPN, and epoch-count sweeps for model coverage.",
            "repeat_rationale": "Selected configurations are repeated to capture variance in the model-training corpus.",
        },
        {
            "component": "custom_cosmoflow_case_study",
            "selection": "custom CosmoFlow V100",
            "measurement_count": int(
                df[
                    (df["version"].eq("custom"))
                    & (df["workload_name"].str.startswith("cosmoflow"))
                ].shape[0]
            ),
            "purpose": "CosmoFlow reader-thread and file-layout optimization study.",
            "repeat_rationale": "Baseline, relayout, and reader-thread configurations are repeated for aggregate optimization summaries.",
        },
        {
            "component": "custom_llama_checkpoint_probes",
            "selection": "custom Llama checkpoint-heavy variants",
            "measurement_count": int(
                df[
                    (df["version"].eq("custom"))
                    & (df["workload_name"].str.startswith("llama"))
                ].shape[0]
            ),
            "purpose": "Checkpoint-heavy Llama probes that broaden the diagnostic training set.",
            "repeat_rationale": "Small repeated probes cover SCR and non-SCR checkpoint behavior.",
        },
        {
            "component": "custom_unet3d_case_study",
            "selection": "custom U-Net 3D",
            "measurement_count": int(
                df[
                    (df["version"].eq("custom"))
                    & (df["workload_name"].str.startswith("unet3d"))
                ].shape[0]
            ),
            "purpose": "U-Net 3D format, file-count, reader-thread, prefetch, and epoch-count optimization study.",
            "repeat_rationale": "Baseline, format-switch, and asynchronous-loader configurations are repeated for aggregate optimization summaries.",
        },
        {
            "component": "custom_megatron_deepspeed_case_study",
            "selection": "custom Megatron-DeepSpeed",
            "measurement_count": int(
                df[
                    (df["version"].eq("custom"))
                    & (df["workload_name"].str.contains("megatron_deepspeed", case=False, na=False))
                ].shape[0]
            ),
            "purpose": "Megatron-DeepSpeed PPN, compute-time, node-count, SCR cache, buffer, and async-flush checkpointing study.",
            "repeat_rationale": "Baseline and SCR/asynchronous checkpoint configurations are repeated for aggregate optimization summaries.",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {
        "component": "total",
        "selection": "all components",
        "measurement_count": int(summary["measurement_count"].sum()),
        "purpose": "Full measurement count expected after postprocessing.",
        "repeat_rationale": "Sum of all source-scale, custom sweep, and case-study measurements.",
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--workload-table", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--expected-measurements", type=int, default=571)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    workload_table = resolve(args.workload_table or Path("globals/ml_workload_all.parquet"), repo_root)
    output_csv = resolve(args.output_csv or Path("scripts/repro/dlio_sweep_config_full.csv"), repo_root)
    summary_csv = resolve(
        args.summary_csv or Path("results/repro/dlio_sweep_config_full_summary.csv"),
        repo_root,
    )

    df = pd.read_parquet(workload_table)
    plan = build_run_plan(df)
    if int(plan["repeat_count"].sum()) != args.expected_measurements:
        raise SystemExit(
            f"run-plan repeat count mismatch: expected {args.expected_measurements}, "
            f"got {int(plan['repeat_count'].sum())}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    build_summary(df).to_csv(summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"wrote {output_csv} ({len(plan)} rows, repeat_count sum {int(plan['repeat_count'].sum())})")
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
