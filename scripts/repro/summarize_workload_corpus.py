#!/usr/bin/env python3
"""Summarize the reproduction workload corpus.

Created for artifact reviewers: validate the compact base sweep, the explicit
full sweep configuration, and the generated workload table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_axis(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return [""]
    return [part.strip() for part in str(value).split(";") if part.strip()]


def expanded_sweep_count(sweep_csv: Path) -> tuple[int, pd.DataFrame]:
    sweep = pd.read_csv(sweep_csv).fillna("")
    rows = []
    total = 0
    for _, row in sweep.iterrows():
        axis_count = 1
        for col in (
            "node_values",
            "reader_threads_values",
            "dataset_format_values",
            "num_files_values",
            "num_samples_values",
        ):
            axis_count *= len(split_axis(row[col]))
        repeat_count = int(row["repeat_count"] or 1)
        expanded_count = axis_count * repeat_count
        total += expanded_count
        rows.append(
            {
                "workload_group": row["workload_group"],
                "workload_name": row["workload_name"],
                "axis_configurations": axis_count,
                "repeat_count": repeat_count,
                "expanded_configurations": expanded_count,
            }
        )
    return total, pd.DataFrame(rows)


def summarize_workload_table(workload_table: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(workload_table)
    config_cols = ["workload_name"] + [c for c in df.columns if c.startswith("config_")]
    unique_configs = df.drop_duplicates(config_cols)
    repeat_counts = (
        df.groupby(config_cols, dropna=False)
        .size()
        .reset_index(name="measurement_count")
        .sort_values("measurement_count", ascending=False)
    )
    repeated = repeat_counts[repeat_counts["measurement_count"] > 1].copy()
    workload_counts = (
        df.groupby("workload_name", dropna=False)
        .size()
        .reset_index(name="measurement_count")
        .sort_values("workload_name")
    )
    summary = {
        "workload_table": str(workload_table),
        "workload_measurements": int(len(df)),
        "unique_workload_configurations": int(len(unique_configs)),
        "repeated_configuration_groups": int(len(repeated)),
        "repeat_measurement_excess": int((repeat_counts["measurement_count"] - 1).clip(lower=0).sum()),
    }
    return summary, workload_counts, repeated


def check_expected(name: str, actual: int, expected: int | None) -> None:
    if expected is None:
        return
    if actual != expected:
        raise SystemExit(f"{name} mismatch: expected {expected}, got {actual}")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-sweep-csv", type=Path, default=None)
    parser.add_argument("--full-sweep-csv", type=Path, default=None)
    parser.add_argument("--run-plan-csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sweep-csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workload-table", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-workload-table", action="store_true")
    parser.add_argument("--expected-base-configurations", type=int, default=263)
    parser.add_argument("--expected-full-measurements", type=int, default=571)
    parser.add_argument("--expected-plan-measurements", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-measurements", type=int, default=None)
    parser.add_argument("--expected-unique-configurations", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    base_sweep_csv = resolve_path(
        args.base_sweep_csv or args.sweep_csv or Path("scripts/repro/dlio_sweep_config_base.csv"),
        repo_root,
    )
    run_plan_csv = resolve_path(
        args.full_sweep_csv or args.run_plan_csv or Path("scripts/repro/dlio_sweep_config_full.csv"),
        repo_root,
    )
    workload_table = resolve_path(
        args.workload_table or Path("globals/ml_workload_all.parquet"),
        repo_root,
    )
    output_dir = resolve_path(args.output_dir or Path("results/repro"), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_count, sweep_summary = expanded_sweep_count(base_sweep_csv)
    check_expected("base sweep configuration count", base_count, args.expected_base_configurations)
    sweep_summary.to_csv(output_dir / "base_sweep_summary.csv", index=False)

    plan_count, run_plan_summary = expanded_sweep_count(run_plan_csv)
    expected_full = args.expected_plan_measurements or args.expected_full_measurements
    check_expected("full sweep measurement count", plan_count, expected_full)
    run_plan_summary.to_csv(output_dir / "full_sweep_config_expanded_summary.csv", index=False)

    summary: dict[str, object] = {
        "base_sweep_configurations": int(base_count),
        "base_sweep_csv": str(base_sweep_csv),
        "full_sweep_config_measurements": int(plan_count),
        "full_sweep_config_csv": str(run_plan_csv),
        "workload_table_present": workload_table.exists(),
        "note": (
            "The base sweep count expands the compact Table II axes. The full sweep "
            "configuration count is obtained by summing repeat_count in the explicit "
            "CSV, which pins runtime settings and targeted repeats."
        ),
    }

    if workload_table.exists():
        workload_summary, workload_counts, repeated = summarize_workload_table(workload_table)
        workload_counts.to_csv(output_dir / "workload_measurement_counts.csv", index=False)
        repeated.to_csv(output_dir / "repeated_configuration_counts.csv", index=False)
        check_expected("workload measurement count", workload_summary["workload_measurements"], args.expected_measurements)
        check_expected(
            "unique workload configuration count",
            workload_summary["unique_workload_configurations"],
            args.expected_unique_configurations,
        )
        if args.expected_unique_configurations is None:
            workload_summary.pop("unique_workload_configurations", None)
        summary.update(workload_summary)
    elif args.require_workload_table:
        raise SystemExit(f"Workload table not found: {workload_table}")

    write_json(output_dir / "workload_corpus_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
