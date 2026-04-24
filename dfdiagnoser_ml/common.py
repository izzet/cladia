import os
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional

GLOBALS_DIR = os.environ.get("DFD_REPRO_GLOBALS_DIR", os.environ.get("DFD_GLOBALS_DIR", "globals"))
LAYER_NAMES: List[str] = [
    "app",
    "training",
    "epoch",
    "compute",
    "fetch_data",
    "data_loader",
    "data_loader_fork",
    "reader",
    "reader_posix_lustre",
    "checkpoint",
    "checkpoint_posix_lustre",
    "checkpoint_posix_ssd",
    "other_posix",
    "other_posix_lustre",
    "other_posix_ssd",
]

GROUND_TRUTH_BOTTLENECKS = {
    "cosmoflow_v100_all": {
        "feature_prefixes": ["reader_count", "reader_posix_lustre_read"],
        "layers": ["config", "reader", "reader_posix_lustre"],
    },
    "cosmoflow_v100_posix": {
        "feature_prefixes": ["reader_count", "reader_posix_lustre_read"],
        "layers": ["config", "reader", "reader_posix_lustre"],
    },
    "resnet50_v100_all": {
        "feature_prefixes": [
            "config_dataset_num_samples_per_file",
            "config_num_nodes",
            "reader_count",
        ],
        "layers": ["config", "reader"],
    },
    "unet3d_v100_all": {
        "feature_prefixes": ["data_loader_fork", "data_loader_count", "data_loader_item"],
        "layers": ["data_loader_fork", "data_loader"],
    },
    "unet3d_v100_posix": {
        "feature_prefixes": ["data_loader_fork", "data_loader_count", "data_loader_item"],
        "layers": ["data_loader_fork", "data_loader"],
    },
    "deepspeed_all": {
        "feature_prefixes": [
            "checkpoint_posix_lustre_write",
            "checkpoint_posix_ssd_write",
        ],
        "layers": ["checkpoint_posix_lustre", "checkpoint_posix_ssd"],
    },
    "deepspeed_posix": {
        "feature_prefixes": [
            "checkpoint_posix_lustre_write",
            "checkpoint_posix_ssd_write",
        ],
        "layers": ["checkpoint_posix_lustre", "checkpoint_posix_ssd"],
    },
}

def get_feature_group(feature: str) -> str:
    """Groups features based on layer and primary metric, with special handling for u_/o_ prefixes."""
    lk = layer_key(feature)
    if feature.startswith(("u_", "o_")):
        prefix = feature.split("_")[0]
        return f"{prefix}_{lk}"
    if feature.startswith(lk + "_"):
        remainder = feature.replace(lk + "_", "", 1)
        primary_metric = remainder.split("_")[0]
        return f"{lk}_{primary_metric}"
    return feature


def layer_key(feature_name: str) -> str:
    n = feature_name
    if n.startswith("u_") or n.startswith("o_"):
        n = n.split("_", 1)[1]
    candidates = sorted(LAYER_NAMES, key=len, reverse=True)
    for layer in candidates:
        if n.startswith(layer + "_") or n == layer:
            return layer
    parts = n.split("_")
    return parts[0] if parts else n


def select_epoch_features(
    df: pd.DataFrame, 
    target_col: str = "epoch_time_max", 
    posix_only=False,
    feature_groups: Optional[List[int]] = None,
) -> list:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    identifiers = {"run_id", "workload_name", "config_id", "time_range", "proc_name"}
    numeric_bool_cols = list(df.select_dtypes(include=[np.number, bool]).columns)

    config_cols = [c for c in numeric_bool_cols if c.startswith("config_")]
    config_cols = [c for c in config_cols if not c.startswith("config_checkpoint_scr")]
    config_cols.remove("config_dataset_record_length_bytes")
    config_cols.remove("config_reader_batch_size")
    config_cols.remove("config_reader_prefetch_workers")
    config_cols.remove("config_train_computation_time")
    config_cols.remove("config_train_epochs")

    hier_cols = [c for c in numeric_bool_cols if c.startswith("o_") or c.startswith("u_")]
    if 'time_frac' in target_col:
        hier_cols = [c for c in hier_cols if c.endswith("_time_frac_self")]

    bhvr_cols = [c for c in numeric_bool_cols if c.endswith("_mean") or c.endswith("_std") or c.endswith("_nunique")]
    bhvr_cols = [c for c in bhvr_cols if c not in hier_cols]
    if 'time' in target_col:
        bhvr_cols = [c for c in bhvr_cols if '_time_' not in c]
    if 'time' in target_col or 'size' in target_col:
        bhvr_cols = [c for c in bhvr_cols if '_bw_' not in c]
    if 'time' in target_col or 'count' in target_col:
        bhvr_cols = [c for c in bhvr_cols if '_ops_' not in c]

    special_cols = [c for c in numeric_bool_cols if c.startswith('dl_') or c.endswith('_ratio')]

    disallowed_prefixes = [
        "app_",
        "o_app_",
        "training_",
        "o_training_",
        "epoch_",
        "o_epoch_",
        "compute_",
        "fetch_data_count_",
        "fetch_data_time_",
        "data_loader_init_count_",   
        "checkpoint_count_",        
        "checkpoint_time_",
        "checkpoint_posix_lustre_count_",
        "checkpoint_posix_lustre_data_",
        "checkpoint_posix_lustre_intensity_",
        "checkpoint_posix_lustre_metadata_",
        "checkpoint_posix_lustre_other_",
        # "checkpoint_posix_lustre_read_",
        "checkpoint_posix_lustre_size_",
        "checkpoint_posix_ssd_count_",
        "checkpoint_posix_ssd_data_",
        "checkpoint_posix_ssd_intensity_",
        "checkpoint_posix_ssd_metadata_",
        "checkpoint_posix_ssd_other_",
        # "checkpoint_posix_ssd_read_",
        "checkpoint_posix_ssd_size_",
        "other_posix_",
        "reader_posix_lustre_count_",
        "reader_posix_lustre_data_",
        "reader_posix_lustre_intensity_",
        "reader_posix_lustre_metadata_",
        "reader_posix_lustre_size_",
        "reader_posix_lustre_sync_",
        "reader_posix_lustre_write_",
        "reader_posix_lustre_other_",
        "reader_posix_ssd_",
    ]
    allowed_prefixes = [
        # "checkpoint_posix_lustre_",
        # "checkpoint_posix_ssd_",
    ]

    # allowed = ["epoch"]
    allowed = []
    if feature_groups is None:
        allowed.extend(config_cols)
        allowed.extend(hier_cols)
        allowed.extend(bhvr_cols)
        allowed.extend(special_cols)
    else:
        for group in feature_groups:
            if group == 0:
                allowed.extend(config_cols)
            elif group == 1:
                allowed.extend(hier_cols)
            elif group == 2:
                allowed.extend(bhvr_cols)
            elif group == 3:
                allowed.extend(special_cols)

    if posix_only:
        allowed = [col for col in allowed if col.startswith("reader_posix_lustre") or col.startswith("checkpoint_posix_lustre") or col.startswith("checkpoint_posix_ssd")]

    filtered = []
    for col in allowed:
        if col == target_col or col in identifiers:
            continue
        if any(col.startswith(pfx) for pfx in disallowed_prefixes) and not any(col.startswith(pfx) for pfx in allowed_prefixes):
            continue
        if re.search(r"_q(\d+)_q(\d+)_", col):
            continue
        filtered.append(col)
    
    return sorted(set(filtered))


def prune_empty_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list) -> list:
    keep, drop = [], []
    for c in feature_cols:
        if c not in train_df.columns:
            drop.append(c)
            continue
        s_tr = train_df[c]
        if not s_tr.notna().any():
            drop.append(c)
        else:
            keep.append(c)
    if drop:
        print(f"Pruned {len(drop)} train-empty features (all-NaN/missing in train): {drop[:10]}{' ...' if len(drop)>10 else ''}")
    return keep


def drop_nonfinite_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    mask = np.isfinite(df[target_col].to_numpy())
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"Dropped {dropped} rows with non-finite {target_col}.")
    return df.loc[mask].copy()


def resolve_target_prefix(target_col: str) -> str:
    """Given a target column like 'epoch_time_max', return base prefix 'epoch_time'."""
    parts = str(target_col).rsplit("_", 1)
    return parts[0] if len(parts) == 2 else str(target_col)


def get_quantiles(
    df: pd.DataFrame,
    target_col: str,
    q_method: str = "mc",
    q_pair: tuple[int, int] = (25, 75),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_low, q_high) arrays for a target column.

    Search order:
    1) Exact names using full target_col: '{target_col}_q{low}'/'{target_col}_q{high}'
    2) Names using resolved prefix: '{prefix}_q{low}'/'{prefix}_q{high}'
    3) Derive from mean/std with prefix: '{prefix}_q{low}_q{high}_mean'/'{prefix}_q{low}_q{high}_std'
    4) Special case: compute_time_frac_epoch -> derive via add_compute_time_frac_epoch_quantiles
    """
    base_full = str(target_col)
    base_pref = resolve_target_prefix(target_col)
    q_low, q_high = q_pair
    # Case 1: exact full base
    q_low_full = f"{base_full}_q{q_low}"
    q_high_full = f"{base_full}_q{q_high}"
    if q_low_full in df.columns and q_high_full in df.columns:
        return df[q_low_full].to_numpy(), df[q_high_full].to_numpy()
    # Case 2: prefix base
    q_low_pref = f"{base_pref}_q{q_low}"
    q_high_pref = f"{base_pref}_q{q_high}"
    if q_low_pref in df.columns and q_high_pref in df.columns:
        return df[q_low_pref].to_numpy(), df[q_high_pref].to_numpy()
    # Case 3: derive from mean/std (prefix)
    mu_col = f"{base_pref}_q{q_low}_q{q_high}_mean"
    sd_col = f"{base_pref}_q{q_low}_q{q_high}_std"
    if mu_col in df.columns and sd_col in df.columns:
        from scipy.stats import norm  # type: ignore
        mu = df[mu_col].to_numpy()
        sd = df[sd_col].to_numpy()
        k_low = norm.ppf(q_low / 100.0)
        k_high = norm.ppf(q_high / 100.0)
        return mu + k_low * sd, mu + k_high * sd
    # Fallback for legacy q25/q75 mean/std columns
    if q_pair == (25, 75):
        mu_col_legacy = f"{base_pref}_q25_q75_mean"
        sd_col_legacy = f"{base_pref}_q25_q75_std"
        if mu_col_legacy in df.columns and sd_col_legacy in df.columns:
            from scipy.stats import norm  # type: ignore
            mu = df[mu_col_legacy].to_numpy()
            sd = df[sd_col_legacy].to_numpy()
            k = norm.ppf(0.75)
            return mu - k * sd, mu + k * sd
    # Case 4: special ratio target
    if base_pref == "compute_time_frac_epoch" or base_full == "compute_time_frac_epoch":
        low, high = q_pair
        print("Adding compute_time_frac_epoch_quantiles with method", q_method, "and q_pair", (low, high))
        tmp = add_compute_time_frac_epoch_quantiles(df, method=q_method, q_pair=(low, high))
        ylow = pd.to_numeric(tmp.get(f"compute_time_frac_epoch_q{low}"), errors="coerce").to_numpy()
        yhigh = pd.to_numeric(tmp.get(f"compute_time_frac_epoch_q{high}"), errors="coerce").to_numpy()
        return ylow, yhigh
    raise KeyError(f"Quantile targets not found: tried {q_low_full}/{q_high_full}, {q_low_pref}/{q_high_pref}, or dynamic mean/std columns.")


def add_special_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with additional DL-specialized diagnostic features.

    Metrics added:
    - dl_pipeline_balance_ratio = fetch_data_time_max / compute_time_max
    - dl_file_shard_coverage = reader_posix_lustre_file_name_nunique / config_dataset_num_files_train (clipped 0-1)
    - dl_data_loader_to_posix_amplification_ratio = data_loader_item_count_sum / reader_posix_lustre_read_count_sum
      Also adds alias dl_dataloader_items_per_posix_read with the same value.
    - dl_reader_preprocess_to_posix_read_ratio = reader_preprocess_time_max / reader_posix_lustre_read_time_max
    - dl_checkpoint_flush_pressure = (write+sync)/checkpoint (prefers *_time_frac_epoch; falls back to *_time_max)
    - reader_posix_lustre_read_small_io_ratio = (<64 KiB read bytes) / total read bytes
    - reader_posix_lustre_seek_to_read_ratio = seek_count_sum / read_count_sum
    - reader_posix_lustre_open_to_close_ratio = open_ops_sum / close_ops_sum
    """
    out = df.copy()

    def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
        n = pd.to_numeric(numer, errors="coerce")
        d = pd.to_numeric(denom, errors="coerce")
        d = d.mask(d == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = n / d
        if hasattr(r, "replace"):
            r = r.replace([np.inf, -np.inf], np.nan)
        return r.astype("Float64").fillna(0.0)

    # is_first_epoch
    if "epoch" in out.columns:
        out["dl_is_first_epoch"] = (out["epoch"] == 0) | (out["epoch"] == 1)

    # dl_pipeline_balance_ratio
    # if {"fetch_data_time_max", "compute_time_max"}.issubset(out.columns):
    #     out["dl_pipeline_balance_ratio"] = _safe_div(out["fetch_data_time_max"], out["compute_time_max"])

    # dl_file_shard_coverage
    if {"reader_posix_lustre_file_name_nunique", "config_dataset_num_files_train"}.issubset(out.columns):
        out["dl_file_shard_coverage"] = _safe_div(
            out["reader_posix_lustre_file_name_nunique"], out["config_dataset_num_files_train"]
        ).clip(0, 1)

    # dl_data_loader_to_posix_amplification_ratio (+ alias)
    # if {"data_loader_item_count_sum", "reader_posix_lustre_read_count_sum"}.issubset(out.columns):
    #     out["dl_dataloader_item_per_posix_read"] = _safe_div(out["data_loader_item_count_sum"], out["reader_posix_lustre_read_count_sum"])

    # dl_reader_preprocess_to_posix_read_ratio
    # if {"reader_preprocess_time_max", "reader_posix_lustre_read_time_max"}.issubset(out.columns):
    #     out["dl_reader_preprocess_to_posix_read_ratio"] = _safe_div(
    #         out["reader_preprocess_time_max"], out["reader_posix_lustre_read_time_max"]
    #     )

    # dl_checkpoint_flush_pressure
    # Prefer epoch-normalized shares; otherwise fall back to absolute times
    if {
        "checkpoint_posix_lustre_write_time_max",
        "checkpoint_posix_lustre_sync_time_max",
        "checkpoint_time_max",
    }.issubset(out.columns):
        out["dl_checkpoint_flush_pressure"] = _safe_div(
            out["checkpoint_posix_lustre_write_time_max"] + out["checkpoint_posix_lustre_sync_time_max"],
            out["checkpoint_time_max"],
        )

    # dl_checkpoint_posix_lustre_write_size_per_process
    if {"checkpoint_posix_lustre_write_size_sum", "config_num_processes"}.issubset(out.columns):
        out["dl_checkpoint_posix_lustre_write_size_per_process"] = _safe_div(
            out["checkpoint_posix_lustre_write_size_sum"], out["config_num_processes"]
        )

    # reader_posix_lustre_read_small_io_ratio (<64 KiB)
    small_bins = [
        "reader_posix_lustre_read_size_bin_0_4kib_sum",
        "reader_posix_lustre_read_size_bin_4kib_16kib_sum",
        "reader_posix_lustre_read_size_bin_16kib_64kib_sum",
    ]
    if "reader_posix_lustre_read_size_sum" in out.columns and any(c in out.columns for c in small_bins):
        small_sum = out[[c for c in small_bins if c in out.columns]].sum(axis=1)
        out["reader_posix_lustre_read_small_io_ratio"] = _safe_div(small_sum, out["reader_posix_lustre_read_size_sum"]).clip(0, 1)

    # reader_posix_lustre_seek_to_read_ratio
    if {"reader_posix_lustre_seek_count_sum", "reader_posix_lustre_read_count_sum"}.issubset(out.columns):
        out["reader_posix_lustre_seek_to_read_ratio"] = _safe_div(
            out["reader_posix_lustre_seek_count_sum"], out["reader_posix_lustre_read_count_sum"]
        )

    # reader_posix_lustre_open_to_close_ratio
    if {"reader_posix_lustre_open_ops_sum", "reader_posix_lustre_close_ops_sum"}.issubset(out.columns):
        out["reader_posix_lustre_open_to_close_ratio"] = _safe_div(
            out["reader_posix_lustre_open_ops_sum"], out["reader_posix_lustre_close_ops_sum"]
        )

    # dl_data_loader_item_per_worker
    # if {"data_loader_item_count_sum", "config_reader_read_threads"}.issubset(out.columns):
    #     out["dl_data_loader_item_per_worker"] = _safe_div(
    #         out["data_loader_item_count_sum"], out["config_reader_read_threads"]
    #     )

    # dl_data_loader_fork_per_worker
    if {"data_loader_fork_count_sum", "config_reader_read_threads"}.issubset(out.columns):
        out["dl_data_loader_fork_per_worker"] = _safe_div(
            out["data_loader_fork_count_sum"], out["config_reader_read_threads"]
        )

    # dl_reader_posix_lustre_read_count_per_worker
    # if {"reader_posix_lustre_read_count_sum", "config_reader_read_threads"}.issubset(out.columns):
    #     out["dl_reader_posix_lustre_read_count_per_worker"] = _safe_div(
    #         out["reader_posix_lustre_read_count_sum"], out["config_reader_read_threads"]
    #     )

    # dl_reader_posix_lustre_read_size_per_worker
    # if {"reader_posix_lustre_read_size_sum", "config_reader_read_threads"}.issubset(out.columns):
    #     out["dl_reader_posix_lustre_read_size_per_worker"] = _safe_div(
    #         out["reader_posix_lustre_read_size_sum"], out["config_reader_read_threads"]
    #     )

    return out


def add_compute_time_frac_epoch_quantiles(
    df: pd.DataFrame,
    method: str = "mc",
    samples: int = 2000,
    rho: float = 0.0,
    random_state: int | None = 42,
    q_pair: tuple[int, int] = (25, 75),
) -> pd.DataFrame:
    """Add compute_time_frac_epoch_q25 and _q75 using approximate ratio quantiles.

    We model Z = X / Y where X=compute_time, Y=epoch_time. Given only mean/std within q25–q75 bands
    for X and Y, exact ratio quantiles are not identifiable. We provide:
    - method="mc": Monte Carlo using (truncated) bivariate normal with corr=rho (default 0), clipped to >0.
    - method="delta": fast delta-method approximation assuming independence (rho ignored), then convert to q25/q75
      using a normal approximation around the ratio mean.

    Columns required:
      compute_time_q25_q75_mean, compute_time_q25_q75_std,
      epoch_time_q25_q75_mean,   epoch_time_q25_q75_std
    """
    out = df.copy()

    mu_x = pd.to_numeric(out.get("compute_time_q25_q75_mean"), errors="coerce")
    sd_x = pd.to_numeric(out.get("compute_time_q25_q75_std"), errors="coerce")
    mu_y = pd.to_numeric(out.get("epoch_time_q25_q75_mean"), errors="coerce")
    sd_y = pd.to_numeric(out.get("epoch_time_q25_q75_std"), errors="coerce")

    # retained for backward compatibility in delta branch when q_pair=(25,75)
    _ = 0.67448975  # unused placeholder

    q_low, q_high = q_pair
    z_low = np.full(len(out), np.nan, dtype=float)
    z_high = np.full(len(out), np.nan, dtype=float)

    if method == "mc":
        rng = np.random.RandomState(random_state)
        for i in range(len(out)):
            mx = float(mu_x.iat[i]) if pd.notna(mu_x.iat[i]) else np.nan
            sx = float(sd_x.iat[i]) if pd.notna(sd_x.iat[i]) else np.nan
            my = float(mu_y.iat[i]) if pd.notna(mu_y.iat[i]) else np.nan
            sy = float(sd_y.iat[i]) if pd.notna(sd_y.iat[i]) else np.nan
            if not np.isfinite([mx, sx, my, sy]).all():
                continue
            # Build covariance matrix
            cov = rho * sx * sy
            Sigma = np.array([[sx * sx, cov], [cov, sy * sy]], dtype=float)
            # Sample bivariate normal and clip to positive domain
            try:
                samples_xy = rng.multivariate_normal(mean=[mx, my], cov=Sigma, size=samples)
            except Exception:
                # Fallback to independent normals if covariance invalid
                samples_xy = np.column_stack([
                    rng.normal(loc=mx, scale=max(sx, 1e-12), size=samples),
                    rng.normal(loc=my, scale=max(sy, 1e-12), size=samples),
                ])
            x = np.clip(samples_xy[:, 0], a_min=1e-12, a_max=None)
            y = np.clip(samples_xy[:, 1], a_min=1e-12, a_max=None)
            z = x / y
            z_low[i] = float(np.nanpercentile(z, q_low))
            z_high[i] = float(np.nanpercentile(z, q_high))
    elif method == "delta":
        # Delta-method approximation for Var(X/Y) assuming independence
        # E[Z] ≈ mu_x / mu_y; Var[Z] ≈ (sd_x / mu_y)^2 + (mu_x * sd_y / mu_y^2)^2
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_z = (mu_x / mu_y).astype(float)
            var_z = (sd_x / mu_y) ** 2 + ((mu_x * sd_y) / (mu_y ** 2)) ** 2
            sd_z = np.sqrt(var_z)
        # Convert to q25/q75 under normal approx
        # scale k for requested percentiles under normal approx
        from scipy.stats import norm  # type: ignore
        k_low = norm.ppf(q_low/100.0)
        k_high = norm.ppf(q_high/100.0)
        z_low = (mu_z + k_low * sd_z).to_numpy()
        z_high = (mu_z + k_high * sd_z).to_numpy()
        # Clean invalid
        z_low[~np.isfinite(z_low)] = np.nan
        z_high[~np.isfinite(z_high)] = np.nan
    else:
        raise ValueError("method must be 'mc' or 'delta'")

    out[f"compute_time_frac_epoch_q{q_low}"] = z_low
    out[f"compute_time_frac_epoch_q{q_high}"] = z_high
    return out
