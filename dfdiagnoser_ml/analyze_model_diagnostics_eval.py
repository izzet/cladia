import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dfdiagnoser_ml.common import GLOBALS_DIR

def _dir_parts(view_type: str, framework: str, target_col: str, feature_groups: List[int] | None, q_method: str, q_low: int, q_high: int, posix_only: bool = False) -> Tuple:
    path_infix = f"{framework}_{view_type}_{target_col}"
    fg_suffix = ""
    if feature_groups:
        fg_suffix = "fg" + "".join(str(int(x)) for x in feature_groups)
        path_infix = f"{path_infix}_{fg_suffix}"
    posix_suffix = ""
    if posix_only:
        posix_suffix = "_posix"
    q_suffix = f"{q_method}_{q_low}_{q_high}"
    return path_infix, posix_suffix, q_suffix, fg_suffix

def _group_top_columns(cols: List[str]) -> Dict[Tuple[str, str], List[str]]:
    groups: Dict[Tuple[str, str], List[str]] = {}
    for c in cols:
        if not c.startswith("top_"):
            continue
        parts = c.split("_")
        # Expected: top_{prefix}_{kind}_{i} or top_{prefix}_{kind}_{i}_value
        if len(parts) < 4:
            continue
        prefix = parts[1]  # mean, q25, q75, center, width
        kind = parts[2]    # feature, layer
        key = (prefix, kind)
        groups.setdefault(key, []).append(c)
    for k in groups:
        groups[k].sort()
    return groups


def analyze_evaluated(
    framework: str,
    view_type: str = "epoch",
    target_col: str = "compute_time_frac_epoch",
    dataset_name: str = "ml_data_{framework}_holdout_full_{infix}_evaluated.parquet",
    top_k_expected: int = 8,
    show_examples: int = 5,
    q_method: str = "mc",
    q_low: int = 25,
    q_high: int = 75,    
    feature_groups: List[int] | None = None,
    posix_only = False
) -> None:
    dataset_dir = f"{GLOBALS_DIR}/{view_type}/datasets/evaluated"
    path_infix, posix_suffix, q_suffix, _ = _dir_parts(view_type, framework, target_col, feature_groups, q_method, q_low, q_high, posix_only)
    infix = f"{path_infix}_{q_suffix}{posix_suffix}"
    in_path = f"{dataset_dir}/" + dataset_name.format(framework=framework, infix=infix)
    print(f"[AN] Loading evaluated parquet: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"[AN] shape={df.shape}")

    # Basic columns presence
    basics = [
        "y_pred_mean", "mean_abs_err", f"q{q_low}_pred", f"q{q_high}_pred",
        "pred_center", "pred_width", "ams_ps_alpha2",
        "quant_overlap_strict", "quant_winkler_0p5", "quant_iqs", "quant_ams_alpha2", "quant_picp", "quant_r2_center", "quant_r2_width",
    ]
    for b in basics:
        print(f"[AN] has {b}: {b in df.columns}")

    # Width distribution
    if "pred_width" in df.columns:
        w = pd.to_numeric(df["pred_width"], errors="coerce").to_numpy()
        finite = w[np.isfinite(w)]
        if finite.size:
            qs = np.percentile(finite, [0, 25, 50, 75, 90, 95, 99])
            print(f"[AN] pred_width quantiles (0,25,50,75,90,95,99): {np.round(qs, 6).tolist()}")
            print(f"[AN] pred_width mean={float(np.mean(finite)):.6g}, std={float(np.std(finite)):.6g}")
        else:
            print("[AN] pred_width has no finite values")

    # True width diagnostics
    if "true_width" in df.columns:
        tw = pd.to_numeric(df["true_width"], errors="coerce").to_numpy()
        finite = tw[np.isfinite(tw)]
        nz = int(np.sum(finite <= 1e-12))
        nt = int(np.sum(finite <= 5e-3))
        print(f"[AN] true_width zero-count={nz}, tiny(<=0.005)={nt}")
        if finite.size:
            qs = np.percentile(finite, [0, 25, 50, 75, 90, 95, 99])
            print(f"[AN] true_width quantiles (0,25,50,75,90,95,99): {np.round(qs, 6).tolist()}")

    # Top-* columns presence and non-null coverage
    groups = _group_top_columns(list(df.columns))
    if not groups:
        print("[AN] No 'top_*' columns found. This suggests SHAP attribution columns were not written.")
    else:
        for (prefix, kind), cols in sorted(groups.items()):
            # Count slots and values
            slot_cols = [c for c in cols if c.endswith(tuple(str(i) for i in range(1, top_k_expected+1))) and not c.endswith("_value")]
            val_cols = [c for c in cols if c.endswith("_value")]
            n_slots = len(slot_cols)
            n_vals = len(val_cols)
            # Non-null rate for top1 slot
            top1 = f"top_{prefix}_{kind}_1"
            nn = int(df[top1].notna().sum()) if top1 in df.columns else 0
            print(f"[AN] top columns for {prefix}/{kind}: slots={n_slots}, values={n_vals}, non_null_top1={nn}")

    # Metric consistency checks (legacy and new *_ds fields)
    for col in [
        "quant_winkler_obs_0p5", "quant_picp_obs", "quant_cwc_obs_0p5", "quant_iqs", "quant_ams_alpha2",
        "quant_winkler_obs_0p5_ds", "quant_picp_obs_ds", "quant_cwc_obs_0p5_ds", "quant_iqs_ds", "quant_ams_alpha2_ds",
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            uniq = s.dropna().unique()
            if len(uniq) == 1:
                print(f"[AN] Warning: {col} is constant: {uniq[0]}")
            na_rate = float(s.isna().mean())
            if na_rate > 0:
                print(f"[AN] Note: {col} NaN rate={na_rate:.2%}")

    # New dataset-level metrics summary
    def _print_ds(col: str, label: str) -> None:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").iloc[0]
            print(f"[AN] {label}: {val}")

    _print_ds("quant_overlap_strict", "Overlap(strict)_ds")
    _print_ds("quant_iqs_ds", "IQS_ds")
    _print_ds("quant_ams_alpha2_ds", "AMS@alpha2_ds")
    _print_ds("quant_winkler_obs_0p5_ds", "Winkler@0.5_obs_ds")
    _print_ds("quant_picp_obs_ds", "PICP_obs_ds")
    _print_ds("quant_cwc_obs_0p5_ds", "CWC@0.5_obs_ds")
    _print_ds("quant_r2_center", "R2_center_ds")
    _print_ds("quant_r2_width", "R2_width_ds")

    # Observed scalar availability rate
    if "obs_scalar_available" in df.columns:
        rate = float(pd.to_numeric(df["obs_scalar_available"], errors="coerce").mean())
        print(f"[AN] observed-scalar availability rate: {rate:.2%}")

        # Example rows with any non-null top entries
        any_top_cols = sorted([c for cols in groups.values() for c in cols])
        mask_any = df[any_top_cols].notna().any(axis=1) if any_top_cols else pd.Series([], dtype=bool)
        n_any = int(mask_any.sum()) if hasattr(mask_any, "sum") else 0
        print(f"[AN] rows with any non-null top attribution: {n_any}")
        if n_any and show_examples > 0:
            print("[AN] example rows (subset of columns):")
            subset_cols = [c for c in any_top_cols if c.endswith("_1") or c.endswith("_1_value")][:20]
            print(df.loc[mask_any, subset_cols].head(show_examples).to_string(index=False))


if __name__ == "__main__":
    import argparse

    quantile_profiles = {
        "iqr": (25, 75),
        "tail": (90, 95),
    }

    parser = argparse.ArgumentParser(description="Analyze evaluated diagnostic parquet files")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], required=True)
    parser.add_argument("--view_type", type=str, default="epoch")
    parser.add_argument("--target_col", type=str, default="compute_time_frac_epoch")
    parser.add_argument("--dataset_name", type=str, default="ml_data_{framework}_holdout_full_{infix}_evaluated.parquet")
    parser.add_argument("--top_k_expected", type=int, default=8)
    parser.add_argument("--show_examples", type=int, default=5)
    parser.add_argument("--quantile_profile", type=str, choices=sorted(quantile_profiles), default="iqr")
    parser.add_argument("--q_method", type=str, default="mc", help=argparse.SUPPRESS)
    parser.add_argument("--q_low", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--q_high", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--feature_groups", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--posix_only", action="store_true")
    args = parser.parse_args()
    q_low, q_high = quantile_profiles[args.quantile_profile]
    if args.q_low is not None:
        q_low = args.q_low
    if args.q_high is not None:
        q_high = args.q_high

    analyze_evaluated(
        framework=args.framework,
        view_type=args.view_type,
        target_col=args.target_col,
        dataset_name=args.dataset_name,
        top_k_expected=args.top_k_expected,
        show_examples=args.show_examples,
        q_low=q_low,
        q_high=q_high,
        q_method=args.q_method,
        feature_groups=args.feature_groups,
        posix_only=args.posix_only,
    )

