import os
from typing import List

import numpy as np
import pandas as pd
import csv

from dfdiagnoser_ml.common import GLOBALS_DIR


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def summarize_evaluated(
    framework: str,
    view_type: str = "epoch",
    dataset_name: str = "ml_data_{framework}_holdout_full_evaluated.parquet",
    top_k: int = 3,
    include_q_bands: bool = False,
    q_low: int = 25,
    q_high: int = 75,
) -> str:
    """Create a concise CSV summary from evaluated parquet.

    Keeps key metrics and top-k attributions for mean/center/width (optionally q_low/q_high).
    """
    dataset_dir = f"{GLOBALS_DIR}/{view_type}/datasets/evaluated"
    in_path = f"{dataset_dir}/" + dataset_name.format(framework=framework)
    df = pd.read_parquet(in_path)

    # Base identifiers/context if present
    base_cols = _pick_existing(df, [
        "run_id", "workload_name", "config_id", "epoch", "time_range", "proc_name",
    ])

    # Target columns if present
    target_cols = _pick_existing(df, [
        "compute_time_frac_epoch", "epoch_time_max",
    ])

    # Predictions and metrics
    metric_cols = _pick_existing(df, [
        "y_pred_mean", "mean_abs_err", "pred_center", "pred_width", "ams_ps_alpha2", "ams_alpha2_mean_robust",
        "quant_overlap_strict", "quant_iqs", "quant_ams_alpha2",
        "quant_winkler_obs_0p5", "quant_picp_obs", "quant_cwc_obs_0p5",
    ])

    # Optional q-bands
    q_cols = _pick_existing(df, [f"q{q_low}_pred", f"q{q_high}_pred"]) if include_q_bands else []

    # Top-k attribution columns (feature + value, layer + value)
    def top_cols(prefix: str) -> List[str]:
        cols: List[str] = []
        for i in range(1, top_k + 1):
            cols.append(f"top_{prefix}_feature_{i}")
            cols.append(f"top_{prefix}_feature_{i}_value")
        for i in range(1, top_k + 1):
            cols.append(f"top_{prefix}_layer_{i}")
            cols.append(f"top_{prefix}_layer_{i}_value")
        return cols

    attr_cols = _pick_existing(df, top_cols("mean")) + \
                _pick_existing(df, top_cols("center")) + \
                _pick_existing(df, top_cols("width"))
    if include_q_bands:
        attr_cols += _pick_existing(df, top_cols(f"q{q_low}"))
        attr_cols += _pick_existing(df, top_cols(f"q{q_high}"))

    # True interval info
    true_cols = _pick_existing(df, ["true_q25", "true_q75", "true_center", "true_width", "is_true_width_zero", "is_true_width_tiny"])

    keep = base_cols + target_cols + true_cols + metric_cols + q_cols + attr_cols
    if not keep:
        raise RuntimeError("No expected columns found to summarize.")

    # Reset index to ensure alignment when selecting columns
    out = df.loc[:, keep].copy().reset_index(drop=True)

    # Save alongside evaluated parquet
    base, _ = os.path.splitext(os.path.basename(in_path))
    out_path = f"{dataset_dir}/{base}_summary.csv"
    out.to_csv(out_path, index=False, na_rep="NA", quoting=csv.QUOTE_NONNUMERIC)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize evaluated diagnostic parquet files to concise CSV")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], required=True)
    parser.add_argument("--view_type", type=str, default="epoch")
    parser.add_argument("--dataset_name", type=str, default="ml_data_{framework}_holdout_full_evaluated.parquet")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--include_q_bands", action="store_true")
    parser.add_argument("--q_low", type=int, default=25)
    parser.add_argument("--q_high", type=int, default=75)
    args = parser.parse_args()

    summarize_evaluated(
        framework=args.framework,
        view_type=args.view_type,
        dataset_name=args.dataset_name,
        top_k=args.top_k,
        include_q_bands=args.include_q_bands,
        q_low=args.q_low,
        q_high=args.q_high,
    )

