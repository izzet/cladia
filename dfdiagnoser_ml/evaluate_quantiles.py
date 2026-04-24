import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from typing import List, Dict, Any, Tuple
from math import sqrt
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from dfdiagnoser_ml.common import get_quantiles, layer_key, GROUND_TRUTH_BOTTLENECKS
from dfdiagnoser_ml.metrics import (
    lenient_overlap_accuracy_tolerant, lenient_overlap_accuracy,
    compute_winkler_score, compute_interval_quality_score,
    compute_prediction_interval_coverage_probability,
    compute_asymmetric_miss_score,
    median_absolute_percentage_error as mdape_metric,
)


def print_quantile_metrics(metrics: dict, label: str):
    """Helper to print a summary of quantile metrics."""
    print(f"--- Quantile evaluation ({label}) ---")
    print(f"  Overlap            : {metrics.get('overlap', -1):.3f}")
    print(f"  Overlap (tolerant) : {metrics.get('overlap_tolerant', -1):.3f}")
    print(f"  Winkler score      : {metrics.get('winkler', -1):.3f}")
    print(f"  IQS                : {metrics.get('iqs', -1):.3f}")
    print(f"  PICP               : {metrics.get('picp', -1):.3f}")
    print(f"  AMS (alpha=2)      : {metrics.get('ams_alpha2', -1):.3f}")
    print(f"  R2 Center          : {metrics.get('r2_center', -1):.3f}")
    print(f"  R2 Width           : {metrics.get('r2_width', -1):.3f}")
    print(f"  R2 Low             : {metrics.get('r2_low', -1):.3f}")
    print(f"  R2 High            : {metrics.get('r2_high', -1):.3f}")
    print(f"  Low MAE            : {metrics.get('low_mae', -1):.3f}")
    print(f"  Low MDE            : {metrics.get('low_mdea', -1):.3f}")
    print(f"  Low RMSE           : {metrics.get('low_rmse', -1):.3f}")
    print(f"  Low MDAPE          : {metrics.get('low_mdape', -1):.3%}")
    print(f"  High MAE           : {metrics.get('high_mae', -1):.3f}")
    print(f"  High MDE           : {metrics.get('high_mdea', -1):.3f}")
    print(f"  High RMSE          : {metrics.get('high_rmse', -1):.3f}")
    print(f"  High MDAPE         : {metrics.get('high_mdape', -1):.3%}")
    print("------------------------------------")


def _calculate_quantile_metrics(
    df: pd.DataFrame,
    q_low_pipe,
    q_high_pipe,
    feature_cols: List[str],
    target_col: str,
    q_method: str,
    q_pair: tuple,
) -> dict:
    """Helper to compute a set of quantile-based metrics for a given dataset."""
    q_low, q_high = q_pair
    X = df[feature_cols]
    
    # Get true intervals and handle NaNs
    y_low_true, y_high_true = get_quantiles(df, target_col, q_method=q_method, q_pair=(q_low, q_high))
    mask = np.isfinite(y_low_true) & np.isfinite(y_high_true)
    
    if not np.any(mask):
        return {}

    # Make predictions on valid data
    qlow_pred = q_low_pipe.predict(X[mask])
    qhigh_pred = q_high_pipe.predict(X[mask])
    y_true = np.column_stack([y_low_true[mask], y_high_true[mask]])
    y_pred = np.column_stack([np.minimum(qlow_pred, qhigh_pred), np.maximum(qlow_pred, qhigh_pred)])
    y_point = (y_true[:, 0] + y_true[:, 1]) / 2.0

    # Calculate metrics
    metrics = {}
    metrics["overlap"] = lenient_overlap_accuracy(y_true, y_pred)
    metrics["overlap_tolerant"] = lenient_overlap_accuracy_tolerant(y_true, y_pred, tolerance=0.1)
    metrics["winkler"] = compute_winkler_score(y_pred, y_point, alpha=0.5)
    metrics["iqs"] = compute_interval_quality_score(y_true, y_pred, lambda_weight=1.0)
    metrics["picp"] = compute_prediction_interval_coverage_probability(y_pred, y_point)
    metrics["ams_alpha2"] = compute_asymmetric_miss_score(y_true, y_pred, alpha=2.0)

    # R2 scores for center and width
    try:
        from sklearn.metrics import r2_score as _r2
        true_center = (y_true[:, 0] + y_true[:, 1]) / 2.0
        pred_center = (y_pred[:, 0] + y_pred[:, 1]) / 2.0
        true_width = y_true[:, 1] - y_true[:, 0]
        pred_width = y_pred[:, 1] - y_pred[:, 0]
        metrics["r2_center"] = float(_r2(true_center, pred_center))
        metrics["r2_width"] = float(_r2(true_width, pred_width))
        metrics["r2_low"] = float(_r2(y_true[:, 0], y_pred[:, 0]))
        metrics["r2_high"] = float(_r2(y_true[:, 1], y_pred[:, 1]))
    except Exception:
        metrics["r2_center"] = metrics["r2_width"] = metrics["r2_low"] = metrics["r2_high"] = float("nan")

    # Point-prediction metrics for low and high quantiles
    y_true_low, y_pred_low = y_true[:, 0], y_pred[:, 0]
    y_true_high, y_pred_high = y_true[:, 1], y_pred[:, 1]
    metrics["low_mae"] = mean_absolute_error(y_true_low, y_pred_low)
    metrics["low_mdea"] = median_absolute_error(y_true_low, y_pred_low)
    metrics["low_rmse"] = sqrt(mean_squared_error(y_true_low, y_pred_low))
    metrics["low_mdape"] = mdape_metric(y_true_low, y_pred_low)
    metrics["high_mae"] = mean_absolute_error(y_true_high, y_pred_high)
    metrics["high_mdea"] = median_absolute_error(y_true_high, y_pred_high)
    metrics["high_rmse"] = sqrt(mean_squared_error(y_true_high, y_pred_high))
    metrics["high_mdape"] = mdape_metric(y_true_high, y_pred_high)

    return metrics


def evaluate_quantiles(
    q_low_pipe,
    q_high_pipe,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    framework: str,
    label: str = "Test",
    q_method: str = "delta",
    q_pair: tuple = (25, 75),
    top_k: int = 3,
) -> dict:
    q_low, q_high = q_pair
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols]
    y_low_true, y_high_true = get_quantiles(df, target_col, q_method=q_method, q_pair=q_pair)
    mask = np.isfinite(y_low_true) & np.isfinite(y_high_true)
    X_eval = X[mask]

    metrics = _calculate_quantile_metrics(
        df, q_low_pipe, q_high_pipe, feature_cols, target_col, q_method, q_pair
    )

    print_quantile_metrics(metrics, label)

    # SHAP feature importance (only for holdout)
    if label == "Holdout":
        q_low_top_features, q_low_top_layers = [], []
        q_high_top_features, q_high_top_layers = [], []
        try:
            import shap  # type: ignore
            def _topk(pipe):
                if hasattr(pipe, "named_steps"):
                    imputer = pipe.named_steps.get("simpleimputer")
                    model = pipe.named_steps.get("gradientboostingregressor") or pipe.named_steps.get("randomforestregressor")
                    if imputer is None or model is None:
                        return [], []
                    Xt = imputer.transform(X_eval)
                    sv = shap.TreeExplainer(model).shap_values(Xt)
                else:
                    sv = shap.TreeExplainer(pipe).shap_values(X_eval)
                if isinstance(sv, list):
                    sv = sv[0]
                mean_abs = np.abs(sv).mean(axis=0)
                total = float(mean_abs.sum()) + 1e-12
                feat_pairs = [(cols[j], float(mean_abs[j] / total)) for j in range(len(cols))]
                feat_pairs.sort(key=lambda x: x[1], reverse=True)
                feat_top = feat_pairs[:top_k]
                layer_sums: Dict[str, float] = {}
                for j, name in enumerate(cols):
                    lk = layer_key(name)
                    layer_sums[lk] = layer_sums.get(lk, 0.0) + float(mean_abs[j])
                layer_pairs = [(k, v / total) for k, v in layer_sums.items()]
                layer_pairs.sort(key=lambda x: x[1], reverse=True)
                layer_top = layer_pairs[:top_k]
                return feat_top, layer_top
            q_low_top_features, q_low_top_layers = _topk(q_low_pipe)
            q_high_top_features, q_high_top_layers = _topk(q_high_pipe)
        except Exception as _exc:
            print(f"SHAP analysis failed: {_exc}")
        
        metrics["q_low_top_features"] = q_low_top_features
        metrics["q_low_top_layers"] = q_low_top_layers
        metrics["q_high_top_features"] = q_high_top_features
        metrics["q_high_top_layers"] = q_high_top_layers
    
    return metrics


def shap_holdout_pair_report_quantiles(
    q_low_pipe,
    q_high_pipe,
    holdout_df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int = 8,
    sample_rows: int | None = None,
    return_data: bool = False,
    target_prefix: str | None = None,
    q_method: str = "mc",
    q_pair: tuple[int, int] | None = None,
    k_list: List[int] | None = None,
) -> Dict[str, Any] | None:
    """Per holdout pair, compute SHAP for q25 and q75 GBMs; print layer/feature summaries and deltas.

    Enhancements:
    - Adds center and width views (from q25/q75 SHAP):
        center ≈ (sv75 + sv25)/2, width ≈ (sv75 - sv25)/2
    - Reports both normalized |SHAP| (importance share) and signed mean SHAP (direction)
    - Prints pair context: predicted medians and widths for bad/good and their deltas
    - Optional sampling per pair via sample_rows
    - Optionally returns a structured dict for downstream analysis/plotting
    """
    q_low, q_high = q_pair if q_pair is not None else (25, 75)

    if holdout_df.empty or "holdout_name" not in holdout_df.columns:
        print("No holdout pairs to report for quantiles.")
        return
    try:
        import shap  # type: ignore
    except Exception as exc:
        print(f"SHAP not available ({exc}); skipping quantile holdout pair report.")
        return

    try:
        imp25 = None
        imp75 = None
        if hasattr(q_low_pipe, "named_steps"):
            imp25 = q_low_pipe.named_steps["simpleimputer"]
            gbm25 = q_low_pipe.named_steps["gradientboostingregressor"]
            imp75 = q_high_pipe.named_steps["simpleimputer"]
            gbm75 = q_high_pipe.named_steps["gradientboostingregressor"]
        else:
            gbm25 = q_low_pipe
            gbm75 = q_high_pipe
    except Exception:
        print("Pipelines not accessible for SHAP; skipping.")
        return None

    holdout_df = holdout_df.copy()
    holdout_df["pair_id"] = holdout_df["holdout_name"].str.rsplit("_", n=1).str[0]

    results: Dict[str, Any] = {}
    warned_mismatch = False

    # Try to auto-detect target_prefix if not provided
    auto_prefix: str | None = None
    if target_prefix is None:
        cols_set = set(holdout_df.columns)
        prefixes_q25 = {c[:-4] for c in cols_set if c.endswith("_q25")}
        prefixes_q75 = {c[:-4] for c in cols_set if c.endswith("_q75")}
        common = sorted(prefixes_q25 & prefixes_q75)
        if len(common) == 1:
            auto_prefix = common[0]
        elif len(common) > 1:
            # Prefer epoch_time* if present
            candidates = [p for p in common if p.startswith("epoch_time")] or [p for p in common if p.startswith("compute_time")] or common
            auto_prefix = candidates[0]
    tpfx = target_prefix or auto_prefix

    k_list = sorted(set(k_list or [1, 3, 5, 10]))

    # --- Per-pair SHAP summary ---
    for pid, g in holdout_df.groupby("pair_id"):
        bad = g[g["holdout_name"].str.endswith("_bad")]
        good = g[g["holdout_name"].str.endswith("_good")]
        if len(bad) == 0 or len(good) == 0:
            continue
        cols = [c for c in feature_cols if c in g.columns]
        X_pair = g[cols]

        # Optional sampling to speed up large pairs
        if sample_rows is not None and len(X_pair) > sample_rows:
            X_pair = X_pair.sample(sample_rows, random_state=42)
            bad = bad.loc[bad.index.intersection(X_pair.index)]
            good = good.loc[good.index.intersection(X_pair.index)]

        # SHAP for q25
        Xt25 = imp25.transform(X_pair) if imp25 is not None else X_pair
        sv25 = shap.TreeExplainer(gbm25).shap_values(Xt25)
        if isinstance(sv25, list):
            sv25 = sv25[0]
        # SHAP for q75
        Xt75 = imp75.transform(X_pair) if imp75 is not None else X_pair
        sv75 = shap.TreeExplainer(gbm75).shap_values(Xt75)
        if isinstance(sv75, list):
            sv75 = sv75[0]

        # Feature alignment safety
        def _align(shap_matrix: np.ndarray, columns: List[str]) -> Tuple[np.ndarray, List[str]]:
            if shap_matrix.shape[1] != len(columns):
                limit = min(shap_matrix.shape[1], len(columns))
                if not warned_mismatch:
                    print(f"Warning: SHAP feature length mismatch (cols={len(columns)} vs contrib={shap_matrix.shape[1]}). Truncating to {limit}.")
                return shap_matrix[:, :limit], columns[:limit]
            return shap_matrix, columns

        sv25, cols25 = _align(sv25, cols)
        sv75, cols75 = _align(sv75, cols)
        # Use common aligned columns (take min length)
        limit = min(sv25.shape[1], sv75.shape[1])
        if limit != len(cols):
            cols = cols[:limit]
        sv25 = sv25[:, :limit]
        sv75 = sv75[:, :limit]

        # Index mapping for selecting bad/good rows in SHAP matrices
        pos_map = {idx: i for i, idx in enumerate(X_pair.index)}
        bad_pos = [pos_map[i] for i in bad.index if i in pos_map]
        good_pos = [pos_map[i] for i in good.index if i in pos_map]

        # Identify I/O-bound samples and their positions
        bad_io_bound = bad[bad['compute_time_frac_epoch'] < 0.8]
        good_io_bound = good[good['compute_time_frac_epoch'] < 0.8]
        bad_io_bound_pos = [pos_map[i] for i in bad_io_bound.index if i in pos_map]
        good_io_bound_pos = [pos_map[i] for i in good_io_bound.index if i in pos_map]

        # Prediction stats (median/width)
        def _pred_stats(df_part: pd.DataFrame) -> Tuple[float, float, float, float]:
            Xp = df_part[cols]
            q25p = q_low_pipe.predict(Xp)
            q75p = q_high_pipe.predict(Xp)
            med = float(((q25p + q75p) / 2.0).mean())
            wid = float((q75p - q25p).mean())
            return med, wid, float(q25p.mean()), float(q75p.mean())
        
        good_median, good_width, good_low, good_high = _pred_stats(good)
        bad_median, bad_width, bad_low, bad_high = _pred_stats(bad)
        delta_median = good_median - bad_median
        delta_width = good_width - bad_width
        delta_low = good_low - bad_low
        delta_high = good_high - bad_high

        print(f"\nQuantile SHAP for pair: {pid}")
        print(f"  Pred median: bad={bad_median:.3f}, good={good_median:.3f}, delta={delta_median:+.3f}")
        print(f"  Pred width:  bad={bad_width:.3f}, good={good_width:.3f}, delta={delta_width:+.3f}")
        print(f"  Pred low:    bad={bad_low:.3f}, good={good_low:.3f}, delta={delta_low:+.3f}")
        print(f"  Pred high:   bad={bad_high:.3f}, good={good_high:.3f}, delta={delta_high:+.3f}")

        # True interval stats
        good_true_low, good_true_high = get_quantiles(good, tpfx, q_method=q_method, q_pair=q_pair)
        bad_true_low, bad_true_high = get_quantiles(bad, tpfx, q_method=q_method, q_pair=q_pair)

        def report_for(name: str, sv_matrix: np.ndarray, bad_pos_list: List[int], good_pos_list: List[int]):
            contrib_abs = np.abs(sv_matrix)
            contrib_signed = sv_matrix
            # Align length handled earlier
            c2 = contrib_abs
            s2 = contrib_signed
            if not bad_pos_list or not good_pos_list:
                return
            # Aggregate
            def agg_norm(pos_list):
                mean_abs = c2[pos_list].mean(axis=0)
                mean_signed = s2[pos_list].mean(axis=0)
                total = float(mean_abs.sum()) + 1e-12
                layer_abs: Dict[str, float] = {}
                layer_signed: Dict[str, float] = {}
                for j, fname in enumerate(cols):
                    key = layer_key(fname)
                    layer_abs[key] = layer_abs.get(key, 0.0) + float(mean_abs[j])
                    layer_signed[key] = layer_signed.get(key, 0.0) + float(mean_signed[j])
                # Normalize abs to shares; signed stays in original units relative to mean abs scale
                layer_abs_norm = {k: v / total for k, v in layer_abs.items()}
                feat_abs_norm = (mean_abs / total)
                # Signed feature mean left as-is (could be reported alongside)
                return layer_abs_norm, feat_abs_norm, layer_signed, mean_signed

            lay_bad_abs, feat_bad_abs, lay_bad_signed, feat_bad_signed = agg_norm(bad_pos_list)
            lay_good_abs, feat_good_abs, lay_good_signed, feat_good_signed = agg_norm(good_pos_list)
            # Compact layer table: [layer, bad_share, good_share, delta, signed_delta]
            all_layers = sorted(set(lay_bad_abs) | set(lay_good_abs))
            layer_rows = []
            for lk in all_layers:
                bshare = lay_bad_abs.get(lk, 0.0)
                gshare = lay_good_abs.get(lk, 0.0)
                delta = bshare - gshare
                sd = lay_bad_signed.get(lk, 0.0) - lay_good_signed.get(lk, 0.0)
                layer_rows.append((lk, bshare, gshare, delta, sd))
            layer_rows.sort(key=lambda x: abs(x[3]), reverse=True)
            print(f"  [{name}] Layers (bad_share, good_share, delta, signed_delta):")
            for lk, bshare, gshare, ldelta, lsd in layer_rows[:top_k]:
                print(f"    {lk}: bad={bshare:.3f}, good={gshare:.3f}, delta={ldelta:+.3f}, s_delta={lsd:+.3f}")

            # Compact feature table: [feature, bad_share, good_share, delta, signed_delta]
            feat_rows = []
            fb_abs = feat_bad_abs.tolist()
            fg_abs = feat_good_abs.tolist()
            fb_sig = feat_bad_signed.tolist()
            fg_sig = feat_good_signed.tolist()
            for idx_f, fname in enumerate(cols):
                vb = float(fb_abs[idx_f])
                vg = float(fg_abs[idx_f])
                delta = vb - vg
                sd = float(fb_sig[idx_f] - fg_sig[idx_f])
                feat_rows.append((fname, vb, vg, delta, sd))
            feat_rows.sort(key=lambda x: abs(x[3]), reverse=True)
            print(f"  [{name}] Features (bad_share, good_share, delta, signed_delta):")
            for fname, vb, vg, fdelta, sd in feat_rows[:top_k]:
                print(f"    {fname}: bad={vb:.3f}, good={vg:.3f}, delta={fdelta:+.3f}, s_delta={sd:+.3f}")

            return {
                "layers_abs_bad": lay_bad_abs,
                "layers_abs_good": lay_good_abs,
                "layers_signed_bad": lay_bad_signed,
                "layers_signed_good": lay_good_signed,
                "features_abs_bad": dict(zip(cols, feat_bad_abs.tolist())),
                "features_abs_good": dict(zip(cols, feat_good_abs.tolist())),
                "features_signed_bad": dict(zip(cols, feat_bad_signed.tolist())),
                "features_signed_good": dict(zip(cols, feat_good_signed.tolist())),
            }

        def get_shap_summary_data(
            sv_matrix: np.ndarray, 
            feature_names: List[str], 
            bad_pos_list: List[int], 
            good_pos_list: List[int]
        ) -> Dict[str, pd.DataFrame]:
            """
            Calculates aggregated SHAP statistics for bad and good configurations.

            Returns a dictionary containing two DataFrames: 'bad' and 'good'.
            Each DataFrame is indexed by feature name and contains columns for
            mean_abs_shap and mean_shap.
            """
            results = {}
            for group_name, pos_list in [("bad", bad_pos_list), ("good", good_pos_list)]:
                if not pos_list:
                    # Return an empty DataFrame if no data for this group
                    results[group_name] = pd.DataFrame(columns=['mean_abs_shap', 'mean_shap'])
                    continue

                # Select the relevant rows from the SHAP matrix
                group_sv = sv_matrix[pos_list]
                
                # Calculate metrics
                mean_abs_shap = np.abs(group_sv).mean(axis=0)
                mean_shap = group_sv.mean(axis=0)

                # Create DataFrame
                df = pd.DataFrame({
                    'feature': feature_names,
                    'mean_abs_shap': mean_abs_shap,
                    'mean_shap': mean_shap
                })
                
                # Sort by importance (mean absolute value)
                df = df.sort_values(by='mean_abs_shap', ascending=False).set_index('feature')
                results[group_name] = df
                
            return results

        def get_shap_first_sample_data(
            sv_matrix: np.ndarray,
            feature_names: List[str],
            bad_pos_list: List[int],
            good_pos_list: List[int]
        ) -> Dict[str, pd.DataFrame]:
            """
            Extracts SHAP values for the first sample in bad and good configurations.

            Returns a dictionary containing two DataFrames: 'bad' and 'good'.
            Each DataFrame is indexed by feature name and contains columns for
            the raw shap_value and its absolute value (abs_shap_value).
            """
            results = {}
            for group_name, pos_list in [("bad", bad_pos_list), ("good", good_pos_list)]:
                if not pos_list:
                    # Return an empty DataFrame if no data for this group
                    results[group_name] = pd.DataFrame(columns=['abs_shap_value', 'shap_value'])
                    continue

                # Select the first sample from the list
                first_sample_pos = pos_list[0]
                sample_sv = sv_matrix[first_sample_pos]
                
                # Create DataFrame
                df = pd.DataFrame({
                    'feature': feature_names,
                    'abs_shap_value': np.abs(sample_sv),
                    'shap_value': sample_sv,
                })
                
                # Sort by importance (absolute value)
                df = df.sort_values(by='abs_shap_value', ascending=False).set_index('feature')
                results[group_name] = df
                
            return results

        # Pair context
        mb, mg, wb, wg = _pred_stats(good)

        res_q25 = report_for("q_low", sv25, bad_pos, good_pos)
        res_q75 = report_for("q_high", sv75, bad_pos, good_pos)
        sv_stability = sv25 - sv75
        res_stability = report_for("stability", sv_stability, bad_pos, good_pos)

        # Center and width views
        sv_center = (sv75 + sv25) / 2.0
        sv_width = (sv75 - sv25) / 2.0
        res_center = report_for("center", sv_center, bad_pos, good_pos)
        res_width = report_for("width", sv_width, bad_pos, good_pos)

        # --- Generate summaries (All Samples, I/O Bound, First Sample) ---
        top_s = 20
        sv_sum = sv25 + sv75

        def generate_full_summary_set(
            sv25_mat, sv75_mat, sv_center_mat, sv_width_mat, sv_stability_mat, sv_sum_mat,
            pos_bad, pos_good, feature_cols
        ):
            if not pos_bad and not pos_good:
                return {}
            
            # 1. Generate full, untruncated base summaries
            summary_q25 = get_shap_summary_data(sv25_mat, feature_cols, pos_bad, pos_good)
            summary_q75 = get_shap_summary_data(sv75_mat, feature_cols, pos_bad, pos_good)
            summary_center = get_shap_summary_data(sv_center_mat, feature_cols, pos_bad, pos_good)
            summary_width = get_shap_summary_data(sv_width_mat, feature_cols, pos_bad, pos_good)
            summary_stability = get_shap_summary_data(sv_stability_mat, feature_cols, pos_bad, pos_good)
            summary_sum = get_shap_summary_data(sv_sum_mat, feature_cols, pos_bad, pos_good)

            base_summaries = { "q_low": summary_q25, "q_high": summary_q75, "center": summary_center, "width": summary_width, "stability": summary_stability, "sum": summary_sum }
            final_summaries = {}

            # 2. Generate Layer-Aware summaries and Min summary
            summary_min, summary_sum_la, summary_min_la, summary_center_la, summary_width_la, summary_stability_la = {}, {}, {}, {}, {}, {}
            for group in ["bad", "good"]:
                # --- Min Summary ---
                df25, df75 = base_summaries["q_low"][group], base_summaries["q_high"][group]
                all_feats = df25.index.union(df75.index)
                df25 = df25.reindex(all_feats, fill_value=0); df75 = df75.reindex(all_feats, fill_value=0)
                min_df = pd.DataFrame(index=all_feats)
                min_df['mean_shap'] = np.minimum(df25['mean_shap'], df75['mean_shap'])
                min_df['mean_abs_shap'] = min_df['mean_shap'].abs()
                summary_min[group] = min_df

                # --- Layer-Aware Calculation ---
                def calculate_layer_aware(base_df, shap_col):
                    df = base_df.copy()
                    df['layer'] = df.index.map(layer_key)
                    layer_impact = df.groupby('layer')[shap_col].sum()
                    df['layer_weight'] = df['layer'].map(layer_impact)
                    df['layer_aware_score'] = df['layer_weight'] * df['mean_shap']
                    return df
                
                summary_sum_la[group] = calculate_layer_aware(base_summaries["sum"][group], 'mean_abs_shap')
                summary_min_la[group] = calculate_layer_aware(summary_min[group], 'mean_abs_shap')
                summary_center_la[group] = calculate_layer_aware(base_summaries["center"][group], 'mean_abs_shap')
                summary_width_la[group] = calculate_layer_aware(base_summaries["width"][group], 'mean_abs_shap')
                summary_stability_la[group] = calculate_layer_aware(base_summaries["stability"][group], 'mean_abs_shap')
            
            # 3. Consolidate, sort, and truncate all summaries
            for key, group_dfs in base_summaries.items():
                final_summaries[key] = {g: df.sort_values('mean_abs_shap', ascending=False).head(top_s) for g, df in group_dfs.items()}
            
            final_summaries["min"] = {g: df.sort_values('mean_shap', ascending=True).head(top_s) for g, df in summary_min.items()}
            final_summaries["sum_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in summary_sum_la.items()}
            final_summaries["min_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in summary_min_la.items()}
            final_summaries["center_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in summary_center_la.items()}
            final_summaries["width_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in summary_width_la.items()}
            final_summaries["stability_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in summary_stability_la.items()}
            
            return final_summaries

        # (generate_first_sample_summary_set is similar but uses different column names)
        def generate_first_sample_summary_set(
            sv25_mat, sv75_mat, sv_center_mat, sv_width_mat, sv_stability_mat, sv_sum_mat,
            pos_bad, pos_good, feature_cols
        ):
            if not pos_bad and not pos_good:
                return {}

            # 1. Generate full, untruncated base summaries
            sample_q25 = get_shap_first_sample_data(sv25_mat, feature_cols, pos_bad, pos_good)
            sample_q75 = get_shap_first_sample_data(sv75_mat, feature_cols, pos_bad, pos_good)
            sample_center = get_shap_first_sample_data(sv_center_mat, feature_cols, pos_bad, pos_good)
            sample_width = get_shap_first_sample_data(sv_width_mat, feature_cols, pos_bad, pos_good)
            sample_stability = get_shap_first_sample_data(sv_stability_mat, feature_cols, pos_bad, pos_good)
            sample_sum = get_shap_first_sample_data(sv_sum_mat, feature_cols, pos_bad, pos_good)

            base_summaries = { "q_low": sample_q25, "q_high": sample_q75, "center": sample_center, "width": sample_width, "stability": sample_stability, "sum": sample_sum }
            final_summaries = {}

            # 2. Generate Layer-Aware and Min summaries
            sample_min, sample_sum_la, sample_min_la, sample_center_la, sample_width_la, sample_stability_la = {}, {}, {}, {}, {}, {}
            for group in ["bad", "good"]:
                 # --- Min Summary ---
                df25, df75 = base_summaries["q_low"][group], base_summaries["q_high"][group]
                all_feats = df25.index.union(df75.index)
                df25 = df25.reindex(all_feats, fill_value=0); df75 = df75.reindex(all_feats, fill_value=0)
                min_df = pd.DataFrame(index=all_feats)
                min_df['shap_value'] = np.minimum(df25['shap_value'], df75['shap_value'])
                min_df['abs_shap_value'] = min_df['shap_value'].abs()
                sample_min[group] = min_df

                # --- Layer-Aware Calculation ---
                def calculate_layer_aware(base_df, shap_col):
                    df = base_df.copy()
                    df['layer'] = df.index.map(layer_key)
                    layer_impact = df.groupby('layer')[shap_col].sum()
                    df['layer_weight'] = df['layer'].map(layer_impact)
                    df['layer_aware_score'] = df['layer_weight'] * df['shap_value']
                    return df

                sample_sum_la[group] = calculate_layer_aware(base_summaries["sum"][group], 'abs_shap_value')
                sample_min_la[group] = calculate_layer_aware(sample_min[group], 'abs_shap_value')
                sample_center_la[group] = calculate_layer_aware(base_summaries["center"][group], 'abs_shap_value')
                sample_width_la[group] = calculate_layer_aware(base_summaries["width"][group], 'abs_shap_value')
                sample_stability_la[group] = calculate_layer_aware(base_summaries["stability"][group], 'abs_shap_value')

            # 3. Consolidate, sort, and truncate all summaries
            for key, group_dfs in base_summaries.items():
                final_summaries[key] = {g: df.head(top_s) for g, df in group_dfs.items()}
            
            final_summaries["min"] = {g: df.sort_values('shap_value', ascending=True).head(top_s) for g, df in sample_min.items()}
            final_summaries["sum_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in sample_sum_la.items()}
            final_summaries["min_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in sample_min_la.items()}
            final_summaries["center_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in sample_center_la.items()}
            final_summaries["width_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in sample_width_la.items()}
            final_summaries["stability_layer_aware"] = {g: df.sort_values('layer_aware_score', ascending=True).head(top_s) for g, df in sample_stability_la.items()}

            return final_summaries

        # 1. All samples
        summary_dataframes = generate_full_summary_set(
            sv25, sv75, sv_center, sv_width, sv_stability, sv_sum, bad_pos, good_pos, cols
        )

        # 2. I/O-bound samples
        io_bound_summary_dataframes = generate_full_summary_set(
            sv25, sv75, sv_center, sv_width, sv_stability, sv_sum, bad_io_bound_pos, good_io_bound_pos, cols
        )
        
        # 3. First sample
        first_sample_dataframes = generate_first_sample_summary_set(
             sv25, sv75, sv_center, sv_width, sv_stability, sv_sum, bad_pos, good_pos, cols
        )

        # --- REVISED CODE FOR BOTTLENECK HIT RATE (FEATURE & LAYER) ---
        target_is_higher_is_better = "frac" in (tpfx or "")

        # Get the ground truth dictionary for the current pair
        ground_truth_dict = GROUND_TRUTH_BOTTLENECKS.get(pid)
        
        nrows = len(bad_pos)
        if nrows > 0 and ground_truth_dict:
            ground_truth_prefixes = ground_truth_dict.get("feature_prefixes", [])
            ground_truth_layers = ground_truth_dict.get("layers", [])

            # Extract per-row SHAP matrices for bad rows
            bad_center_sv = sv_center[bad_pos]
            bad_width_sv = sv_width[bad_pos]
            bad_qlow_sv = sv25[bad_pos]
            bad_qhigh_sv = sv75[bad_pos]

            # 1) Center + Width bot hit (matched top-k budget)
            feature_hit_count_cw = {k: 0 for k in k_list}
            layer_hit_count_cw = {k: 0 for k in k_list}
            for i in range(nrows):
                center_row_sv = bad_center_sv[i, :]
                width_row_sv = bad_width_sv[i, :]
                center_score = -center_row_sv if target_is_higher_is_better else center_row_sv
                width_score = width_row_sv
                combined_cw = center_score + width_score
                order_cw = np.argsort(combined_cw)[::-1]
                for k in k_list:
                    top_idx = order_cw[:k]
                    top_features_cw = [cols[j] for j in top_idx]
                    if any(gtp in tf for gtp in ground_truth_prefixes for tf in top_features_cw):
                        feature_hit_count_cw[k] += 1
                    top_layers_cw = {layer_key(f) for f in top_features_cw}
                    if any(gtl in tl for gtl in ground_truth_layers for tl in top_layers_cw):
                        layer_hit_count_cw[k] += 1

            bot_hit_center_width = {
                "by_k": {
                    k: {
                        "feature_hit_count": feature_hit_count_cw[k],
                        "layer_hit_count": layer_hit_count_cw[k],
                        "nrows": nrows,
                    }
                    for k in k_list
                }
            }
            if 3 in k_list:
                print(f"  [Quantile Model Bot-Hit Center+Width for '{pid}'] feature={feature_hit_count_cw[3]}/{nrows}, layer={layer_hit_count_cw[3]}/{nrows}")

            # 2) Q_low + Q_high bot hit (matched top-k budget)
            feature_hit_count_q = {k: 0 for k in k_list}
            layer_hit_count_q = {k: 0 for k in k_list}
            for i in range(nrows):
                qlow_row_sv = bad_qlow_sv[i, :]
                qhigh_row_sv = bad_qhigh_sv[i, :]
                qlow_score = -qlow_row_sv if target_is_higher_is_better else qlow_row_sv
                qhigh_score = -qhigh_row_sv if target_is_higher_is_better else qhigh_row_sv
                combined_q = qlow_score + qhigh_score
                order_q = np.argsort(combined_q)[::-1]
                for k in k_list:
                    top_idx = order_q[:k]
                    top_features_q = [cols[j] for j in top_idx]
                    if any(gtp in tf for gtp in ground_truth_prefixes for tf in top_features_q):
                        feature_hit_count_q[k] += 1
                    top_layers_q = {layer_key(f) for f in top_features_q}
                    if any(gtl in tl for gtl in ground_truth_layers for tl in top_layers_q):
                        layer_hit_count_q[k] += 1

            bot_hit_q_low_high = {
                "by_k": {
                    k: {
                        "feature_hit_count": feature_hit_count_q[k],
                        "layer_hit_count": layer_hit_count_q[k],
                        "nrows": nrows,
                    }
                    for k in k_list
                }
            }
            if 3 in k_list:
                print(f"  [Quantile Model Bot-Hit Q_low+Q_high for '{pid}'] feature={feature_hit_count_q[3]}/{nrows}, layer={layer_hit_count_q[3]}/{nrows}")
        # --- END OF REVISED CODE ---

        # AMS-aware SHAP layer ranking: correlate width SHAP with under-miss component
        # Separate analysis for bad vs good configs
        try:
            def compute_ams_correlations(df_subset: pd.DataFrame, label: str) -> None:
                if df_subset.empty:
                    return
                X_sub = df_subset[cols]
                q25_sub = q_low_pipe.predict(X_sub)
                q75_sub = q_high_pipe.predict(X_sub)
                Pmin_sub = np.minimum(q25_sub, q75_sub)
                Pmax_sub = np.maximum(q25_sub, q75_sub)

                if tpfx is not None:
                    y25_sub, y75_sub = get_quantiles(df_subset, tpfx, q_method=q_method, q_pair=q_pair)
                    Tmin_sub = np.minimum(y25_sub, y75_sub)
                    Tmax_sub = np.maximum(y25_sub, y75_sub)
                    true_w_sub = np.maximum(Tmax_sub - Tmin_sub, 1e-12)
                    inter_sub = np.maximum(0.0, np.minimum(Tmax_sub, Pmax_sub) - np.maximum(Tmin_sub, Pmin_sub))
                    under_sub = np.maximum(0.0, true_w_sub - inter_sub) / true_w_sub
                    finite_rows_sub = np.isfinite(under_sub)
                    if finite_rows_sub.any():
                        # Get indices for this subset
                        sub_pos_map = {idx: i for i, idx in enumerate(df_subset.index)}
                        sub_pos_keep = [sub_pos_map[i] for i in df_subset.index[finite_rows_sub] if i in sub_pos_map]
                        if sub_pos_keep:
                            sv_width_sub = sv_width[sub_pos_keep]
                            under_sub_f = under_sub[finite_rows_sub]

                            # Compute per-layer correlations
                            layer_stats: Dict[str, Dict[str, float | list | int]] = {}
                            for j, fname in enumerate(cols):
                                lk = layer_key(fname)
                                if lk not in layer_stats:
                                    layer_stats[lk] = {"corr_under_width": [], "mean_center": 0.0, "mean_width": 0.0, "n": 0}
                                x = sv_width_sub[:, j]
                                if np.std(x) > 1e-9 and np.std(under_sub_f) > 1e-9:
                                    rho = float(np.corrcoef(x, under_sub_f)[0, 1])
                                    layer_stats[lk]["corr_under_width"].append(rho)
                                layer_stats[lk]["mean_center"] = float(layer_stats[lk]["mean_center"]) + float(np.mean(sv_center[sub_pos_keep, j]))
                                layer_stats[lk]["mean_width"] = float(layer_stats[lk]["mean_width"]) + float(np.mean(sv_width_sub[:, j]))
                                layer_stats[lk]["n"] = int(layer_stats[lk]["n"]) + 1

                            rows_corr: List[Tuple[str, float, float, float]] = []
                            for lk, d in layer_stats.items():
                                vals = d["corr_under_width"]  # type: ignore
                                rho_u = float(np.mean(vals)) if vals else 0.0
                                mc = float(d["mean_center"]) / max(int(d["n"]), 1)
                                mw = float(d["mean_width"]) / max(int(d["n"]), 1)
                                rows_corr.append((lk, rho_u, mc, mw))
                            rows_corr.sort(key=lambda x: abs(x[1]), reverse=True)

                            print(f"  [ams-{label}] Layers by corr(width SHAP, under-miss):")
                            for lk, rho_u, mc, mw in rows_corr[:top_k]:
                                print(f"    {lk}: corr_under_width={rho_u:+.3f}, mean_center={mc:+.3f}, mean_width={mw:+.3f}")

            # Compute separate correlations for bad vs good
            compute_ams_correlations(bad, "bad")
            compute_ams_correlations(good, "good")

            # Feature-level ranking with the same criterion - separated by bad/good
            def compute_ams_feature_correlations(df_subset: pd.DataFrame, label: str) -> None:
                if df_subset.empty:
                    return
                X_sub = df_subset[cols]
                q25_sub = q_low_pipe.predict(X_sub)
                q75_sub = q_high_pipe.predict(X_sub)
                Pmin_sub = np.minimum(q25_sub, q75_sub)
                Pmax_sub = np.maximum(q25_sub, q75_sub)

                if tpfx is not None:
                    y25_sub, y75_sub = get_quantiles(df_subset, tpfx, q_method=q_method, q_pair=q_pair)
                    Tmin_sub = np.minimum(y25_sub, y75_sub)
                    Tmax_sub = np.maximum(y25_sub, y75_sub)
                    true_w_sub = np.maximum(Tmax_sub - Tmin_sub, 1e-12)
                    inter_sub = np.maximum(0.0, np.minimum(Tmax_sub, Pmax_sub) - np.maximum(Tmin_sub, Pmin_sub))
                    under_sub = np.maximum(0.0, true_w_sub - inter_sub) / true_w_sub
                    finite_rows_sub = np.isfinite(under_sub)
                    if finite_rows_sub.any():
                        # Get indices for this subset
                        sub_pos_map = {idx: i for i, idx in enumerate(df_subset.index)}
                        sub_pos_keep = [sub_pos_map[i] for i in df_subset.index[finite_rows_sub] if i in sub_pos_map]
                        if sub_pos_keep:
                            sv_width_sub = sv_width[sub_pos_keep]
                            sv_center_sub = sv_center[sub_pos_keep]

                            # Compute per-feature correlations
                            rows_corr: List[Tuple[str, float, float, float]] = []
                            for j, fname in enumerate(cols):
                                x = sv_width_sub[:, j]
                                if np.std(x) > 1e-9 and np.std(under_sub[finite_rows_sub]) > 1e-9:
                                    rho = float(np.corrcoef(x, under_sub[finite_rows_sub])[0, 1])
                                else:
                                    rho = 0.0
                                mc = float(np.mean(sv_center_sub[:, j]))
                                mw = float(np.mean(sv_width_sub[:, j]))
                                rows_corr.append((fname, rho, mc, mw))
                            rows_corr.sort(key=lambda x: abs(x[1]), reverse=True)

                            print(f"  [ams-{label}] Features by corr(width SHAP, under-miss):")
                            for fname, rho_u, mc, mw in rows_corr[:top_k]:
                                print(f"    {fname}: corr_under_width={rho_u:+.3f}, mean_center={mc:+.3f}, mean_width={mw:+.3f}")

            # Compute separate feature correlations for bad vs good
            compute_ams_feature_correlations(bad, "bad")
            compute_ams_feature_correlations(good, "good")
        except Exception as _:
            pass

        results[pid] = {
            "pred_median_bad": mb,
            "pred_median_good": mg,
            "pred_width_bad": wb,
            "pred_width_good": wg,
            "q_low": res_q25,
            "q_high": res_q75,
            "center": res_center,
            "width": res_width,
            "stability": res_stability,
        }

        # Attach separated bottleneck hit results if available
        if nrows > 0 and ground_truth_dict:
            results[pid]["bot_hit_center_width"] = bot_hit_center_width
            results[pid]["bot_hit_q_low_high"] = bot_hit_q_low_high

        # Store the raw SHAP matrices for the bad/good subsets
        results[pid]["shap_values"] = {
            "bad_center": sv_center[bad_pos],
            "good_center": sv_center[good_pos],
            "bad_width": sv_width[bad_pos],
            "good_width": sv_width[good_pos],
            "bad_stability": sv_stability[bad_pos],
            "good_stability": sv_stability[good_pos],
            "features": cols  # Store the corresponding feature names
        }

        # Also store the underlying data subsets for context
        results[pid]["data"] = {
            "bad": bad,
            "good": good
        }

        results[pid]["first_sample"] = first_sample_dataframes
        # Store these dataframes in your main results dictionary
        results[pid]["summary_dataframes"] = summary_dataframes

        # Store I/O-bound summary dataframes
        results[pid]["io_bound_summary_dataframes"] = io_bound_summary_dataframes

    if return_data:
        results["q_pair"] = (q_low, q_high)
        return results
    return None


def shap_view_trajectory_report_quantiles(
    q_low_pipe,
    q_high_pipe,
    df: pd.DataFrame,
    feature_cols: List[str],
    view_col: str = "epoch",
    top_k: int = 8,
    sample_rows: int | None = 512,
    return_data: bool = False,
) -> Dict[str, Any] | None:
    """Compute per-view SHAP trajectories for q25/q75 and derived center/width.

    For each value in df[view_col]:
      - Compute Tree SHAP matrices for q25 and q75 (optionally sample rows per epoch)
      - Aggregate to layer-level via layer_key for normalized |SHAP| shares and signed means
      - Also compute center = (sv75+sv25)/2 and width = (sv75-sv25)/2 views

    Prints concise summaries and optionally returns a structured dict with per-epoch stats.
    """
    if df.empty or view_col not in df.columns:
        print(f"No '{view_col}' column or empty dataframe; skipping view-based SHAP trajectories.")
        return None

    try:
        import shap  # type: ignore
    except Exception as exc:
        print(f"SHAP not available ({exc}); skipping view-based SHAP trajectories.")
        return None

    try:
        imp25 = q_low_pipe.named_steps["simpleimputer"]
        gbm25 = q_low_pipe.named_steps["gradientboostingregressor"]
        imp75 = q_high_pipe.named_steps["simpleimputer"]
        gbm75 = q_high_pipe.named_steps["gradientboostingregressor"]
    except Exception:
        print("Pipelines not accessible for SHAP; skipping view-based trajectories.")
        return None

    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        print("No overlapping feature columns for view-based trajectories.")
        return None

    results: Dict[str, Any] = {"rows": [], "view_col": view_col}
    warned_mismatch = False

    def _align(shap_matrix: np.ndarray, columns: List[str]) -> Tuple[np.ndarray, List[str]]:
        if shap_matrix.shape[1] != len(columns):
            limit = min(shap_matrix.shape[1], len(columns))
            nonlocal warned_mismatch
            if not warned_mismatch:
                print(f"Warning: SHAP feature length mismatch (cols={len(columns)} vs contrib={shap_matrix.shape[1]}). Truncating to {limit}.")
                warned_mismatch = True
            return shap_matrix[:, :limit], columns[:limit]
        return shap_matrix, columns

    for view_value, g in df.groupby(view_col):
        X = g[cols]
        if sample_rows is not None and len(X) > sample_rows:
            X = X.sample(sample_rows, random_state=42)
        if len(X) == 0:
            continue

        Xt25 = imp25.transform(X)
        sv25 = shap.TreeExplainer(gbm25).shap_values(Xt25)
        if isinstance(sv25, list):
            sv25 = sv25[0]
        Xt75 = imp75.transform(X)
        sv75 = shap.TreeExplainer(gbm75).shap_values(Xt75)
        if isinstance(sv75, list):
            sv75 = sv75[0]

        sv25, _ = _align(sv25, cols)
        sv75, _ = _align(sv75, cols)
        limit = min(sv25.shape[1], sv75.shape[1])
        sv25 = sv25[:, :limit]
        sv75 = sv75[:, :limit]
        cols_use = cols[:limit]

        def _agg(sv: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
            mean_abs = np.abs(sv).mean(axis=0)
            mean_signed = sv.mean(axis=0)
            total = float(mean_abs.sum()) + 1e-12
            layer_abs: Dict[str, float] = {}
            layer_signed: Dict[str, float] = {}
            for j, fname in enumerate(cols_use):
                key = layer_key(fname)
                layer_abs[key] = layer_abs.get(key, 0.0) + float(mean_abs[j])
                layer_signed[key] = layer_signed.get(key, 0.0) + float(mean_signed[j])
            # Normalize abs to shares; signed is raw mean
            layer_abs_norm = {k: v / total for k, v in layer_abs.items()}
            return layer_abs_norm, layer_signed

        center = (sv75 + sv25) / 2.0
        width = (sv75 - sv25) / 2.0
        stability = sv25 - sv75

        layers_q25_abs, layers_q25_signed = _agg(sv25)
        layers_q75_abs, layers_q75_signed = _agg(sv75)
        layers_center_abs, layers_center_signed = _agg(center)
        layers_width_abs, layers_width_signed = _agg(width)
        layers_stability_abs, layers_stability_signed = _agg(stability)

        # Predicted median/width per view value (on the same sampled X)
        q25p = q_low_pipe.predict(X)
        q75p = q_high_pipe.predict(X)
        pred_median = float(((q25p + q75p) / 2.0).mean())
        pred_width = float((q75p - q25p).mean())

        results["rows"].append({
            "view": view_value,
            "q25_abs": layers_q25_abs,
            "q25_signed": layers_q25_signed,
            "q75_abs": layers_q75_abs,
            "q75_signed": layers_q75_signed,
            "center_abs": layers_center_abs,
            "center_signed": layers_center_signed,
            "width_abs": layers_width_abs,
            "width_signed": layers_width_signed,
            "stability_abs": layers_stability_abs,
            "stability_signed": layers_stability_signed,
            "pred_median": pred_median,
            "pred_width": pred_width,
        })

    # Brief print summary: top layers by mean share over epochs for center/width
    if results["rows"]:
        def _mean_top(key: str) -> List[Tuple[str, float]]:
            acc: Dict[str, float] = {}
            for e in results["rows"]:
                for k, v in e[key].items():
                    acc[k] = acc.get(k, 0.0) + float(v)
            # average over epochs
            n = max(len(results["rows"]), 1)
            items = [(k, v / n) for k, v in acc.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            return items

        top_center = _mean_top("center_abs")[:top_k]
        top_width = _mean_top("width_abs")[:top_k]
        print("\nView-based trajectory SHAP (mean over rows):")
        print("  Top layers (center |SHAP| share):")
        for k, v in top_center:
            print(f"    {k}: {v:.3f}")
        print("  Top layers (width |SHAP| share):")
        for k, v in top_width:
            print(f"    {k}: {v:.3f}")

    if return_data:
        return results
    return None


# ---------------------- Visualization Utilities ----------------------
def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _prep_view_layer_matrix(view_results: Dict[str, Any], key: str) -> Tuple[List[Any], List[str], Any]:
    views: List[Any] = []
    all_layers: Dict[str, int] = {}
    # Collect union of layers
    for e in view_results.get("rows", []):
        views.append(e.get("view"))
        for k in e.get(key, {}).keys():
            if k not in all_layers:
                all_layers[k] = len(all_layers)
    layer_names = sorted(all_layers.keys())
    if not layer_names or not views:
        return [], [], np.zeros((0, 0))
    mat = np.zeros((len(views), len(layer_names)), dtype=float)
    for i, e in enumerate(view_results.get("rows", [])):
        row = e.get(key, {})
        for k, v in row.items():
            j = layer_names.index(k)
            mat[i, j] = float(v)
    return views, layer_names, mat

def plot_ams_diagnostics(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    label: str,
    framework: str,
    target_col: str,
    out_dir: str = "results",
) -> None:
    """Generate AMS diagnostic plots and save with framework/target prefix.

    Produces:
    - Histogram of per-sample AMS
    - Histogram of normalized under/over components
    - AMS vs predicted width scatter
    - AMS vs true width scatter
    - Risk-Coverage curve using proxy risk = -predicted width
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        Tmin = np.minimum(y_true_int[:, 0], y_true_int[:, 1])
        Tmax = np.maximum(y_true_int[:, 0], y_true_int[:, 1])
        Pmin = np.minimum(y_pred_int[:, 0], y_pred_int[:, 1])
        Pmax = np.maximum(y_pred_int[:, 0], y_pred_int[:, 1])
        true_w = np.maximum(Tmax - Tmin, 1e-12)
        pred_w = np.maximum(Pmax - Pmin, 0.0)
        inter = np.maximum(0.0, np.minimum(Tmax, Pmax) - np.maximum(Tmin, Pmin))
        under_err = np.maximum(0.0, true_w - inter)
        over_err = np.maximum(0.0, pred_w - inter)
        ams_ps = (2.0 * under_err + over_err) / true_w
        under_ps = under_err / true_w
        over_ps = over_err / true_w
        lab = label.lower()
        pref = f"{framework}_{target_col}"
        # Severity histogram
        plt.figure(figsize=(6, 3))
        plt.hist(ams_ps, bins=40, color="C0", alpha=0.8)
        plt.title(f"{label}: AMS per-sample"); plt.xlabel("AMS"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{pref}_ams_hist_{lab}.png"), dpi=150); plt.close()
        # Under vs Over components
        plt.figure(figsize=(6, 3))
        plt.hist(under_ps, bins=40, color="C3", alpha=0.6, label="under/true_w")
        plt.hist(over_ps, bins=40, color="C2", alpha=0.6, label="over/true_w")
        plt.title(f"{label}: Under vs Over components"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{pref}_ams_components_{lab}.png"), dpi=150); plt.close()
        # AMS vs predicted width
        plt.figure(figsize=(5, 4))
        plt.scatter(pred_w, ams_ps, s=8, alpha=0.4)
        plt.xlabel("Pred width"); plt.ylabel("AMS"); plt.title(f"{label}: AMS vs predicted width")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{pref}_ams_vs_predwidth_{lab}.png"), dpi=150); plt.close()
        # AMS vs true width
        plt.figure(figsize=(5, 4))
        plt.scatter(true_w, ams_ps, s=8, alpha=0.4, color="C1")
        plt.xlabel("True width"); plt.ylabel("AMS"); plt.title(f"{label}: AMS vs true width")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{pref}_ams_vs_truewidth_{lab}.png"), dpi=150); plt.close()
        # Risk-coverage (proxy risk = -pred width)
        proxy_risk = -pred_w
        order = np.argsort(proxy_risk)[::-1]
        ams_sorted = ams_ps[order]
        n = len(ams_sorted); coverage = np.arange(1, n+1, dtype=float) / float(n)
        sel_risk = np.cumsum(ams_sorted) / np.arange(1, n+1, dtype=float)
        aurc = float(np.trapz(sel_risk, coverage))
        plt.figure(figsize=(5, 4))
        plt.plot(coverage, sel_risk, label=f"AURC={aurc:.4f}")
        plt.xlabel("Coverage"); plt.ylabel("Selective risk (AMS)")
        plt.title(f"{label}: Risk-Coverage (AMS)"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{pref}_ams_risk_coverage_{lab}.png"), dpi=150); plt.close()
    except Exception:
        pass


def plot_view_shap_heatmaps(view_results: Dict[str, Any], out_dir: str = "results", label: str = "Test", cmap: str = "viridis") -> None:
    _ensure_dir(out_dir)
    view_col = view_results.get("view_col", "view")
    views = [
        ("q25_abs", "Q25 |SHAP| share"),
        ("q75_abs", "Q75 |SHAP| share"),
        ("center_abs", "Center |SHAP| share"),
        ("width_abs", "Width |SHAP| share"),
        ("stability_abs", "Stability |SHAP| share"),
    ]
    for key, title in views:
        xs, layers, mat = _prep_view_layer_matrix(view_results, key)
        if mat.size == 0:
            continue
        plt.figure(figsize=(max(6, len(xs) * 0.25), max(3, len(layers) * 0.4)))
        ax = sns.heatmap(mat.T, cmap=cmap, cbar=True)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, rotation=90)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=90)
        ax.set_xlabel(view_col)
        ax.set_ylabel("Layer")
        plt.title(f"{label}: {title} by layer vs {view_col}")
        fname = os.path.join(out_dir, f"{view_col}_heatmap_{key}_{label.lower()}.png")
        plt.tight_layout()
        try:
            plt.savefig(fname, dpi=150)
            print(f"Saved heatmap: {fname}")
        except Exception as exc:
            print(f"Failed to save heatmap {fname}: {exc}")
        finally:
            plt.close()


def plot_view_layer_timeseries(view_results: Dict[str, Any], out_dir: str = "results", top_k: int = 6, label: str = "Test") -> None:
    _ensure_dir(out_dir)
    xs, layers, mat_center = _prep_view_layer_matrix(view_results, "center_abs")
    if mat_center.size == 0:
        return
    # Select top_k by mean share
    mean_share = mat_center.mean(axis=0)
    idx = np.argsort(-mean_share)[: min(top_k, len(layers))]
    sel_layers = [layers[i] for i in idx]
    sel_mat = mat_center[:, idx]

    # Retrieve predicted median/width
    pred_meds = [e.get("pred_median", np.nan) for e in view_results.get("rows", [])]
    pred_wids = [e.get("pred_width", np.nan) for e in view_results.get("rows", [])]
    view_col = view_results.get("view_col", "view")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(sel_layers):
        ax1.plot(xs, sel_mat[:, j], label=name, linewidth=2)
    ax1.set_xlabel(view_col)
    ax1.set_ylabel("Center |SHAP| share")
    ax1.set_title(f"{label}: Top-{len(sel_layers)} layer shares (center) vs {view_col}")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))

    # Secondary axis for median/width (scaled)
    ax2 = ax1.twinx()
    try:
        ax2.plot(xs, pred_meds, color='black', linestyle='--', alpha=0.7, label='pred_median')
        ax2.plot(xs, pred_wids, color='gray', linestyle=':', alpha=0.7, label='pred_width')
        ax2.set_ylabel("Pred median/width")
        # Change-point markers on width using diff z-score
        diffs = np.diff(np.asarray(pred_wids, dtype=float))
        if len(diffs) >= 3 and np.nanstd(diffs) > 0:
            z = (diffs - np.nanmean(diffs)) / (np.nanstd(diffs) + 1e-12)
            cp_idx = np.where(np.abs(z) > 2.0)[0]  # simple 2-sigma rule
            for ci in cp_idx:
                x = xs[ci + 1]
                ax1.axvline(x=x, color='red', alpha=0.2)
    except Exception:
        pass

    fname = os.path.join(out_dir, f"{view_col}_timeseries_center_{label.lower()}.png")
    plt.tight_layout()
    try:
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved time series: {fname}")
    except Exception as exc:
        print(f"Failed to save time series {fname}: {exc}")
    finally:
        plt.close()


def plot_view_pair_layer_deltas(
    q_low_pipe,
    q_high_pipe,
    holdout_df: pd.DataFrame,
    feature_cols: List[str],
    view_col: str = "epoch",
    top_k: int = 8,
    sample_rows: int | None = 512,
    out_dir: str = "results",
) -> None:
    """For each holdout pair, plot epoch-wise layer deltas (bad - good) for center/width.

    Uses normalized |SHAP| shares aggregated per layer. Marks sign flips by hatch overlay.
    """
    if holdout_df.empty or "holdout_name" not in holdout_df.columns or view_col not in holdout_df.columns:
        print("Holdout data lacks 'holdout_name' or view column; skipping pair delta plots.")
        return

    try:
        import shap  # type: ignore
    except Exception as exc:
        print(f"SHAP not available ({exc}); skipping pair delta plots.")
        return

    try:
        imp25 = q_low_pipe.named_steps["simpleimputer"]
        gbm25 = q_low_pipe.named_steps["gradientboostingregressor"]
        imp75 = q_high_pipe.named_steps["simpleimputer"]
        gbm75 = q_high_pipe.named_steps["gradientboostingregressor"]
    except Exception:
        print("Pipelines not accessible for SHAP; skipping pair delta plots.")
        return

    cols = [c for c in feature_cols if c in holdout_df.columns]
    if not cols:
        print("No overlapping feature columns for pair delta plots.")
        return

    _ensure_dir(out_dir)

    def _agg_layers(sv: np.ndarray, cols_use: List[str]) -> Dict[str, float]:
        mean_abs = np.abs(sv).mean(axis=0)
        total = float(mean_abs.sum()) + 1e-12
        layer_abs: Dict[str, float] = {}
        for j, fname in enumerate(cols_use):
            key = layer_key(fname)
            layer_abs[key] = layer_abs.get(key, 0.0) + float(mean_abs[j])
        return {k: v / total for k, v in layer_abs.items()}

    for pid, g in holdout_df.copy().assign(pair_id=lambda d: d["holdout_name"].str.rsplit("_", n=1).str[0]).groupby("pair_id"):
        bad = g[g["holdout_name"].str.endswith("_bad")]
        good = g[g["holdout_name"].str.endswith("_good")]
        if len(bad) == 0 or len(good) == 0:
            continue
        # View order
        x_values = sorted(set(g[view_col].tolist()))
        layers_union: Dict[str, None] = {}
        center_deltas: List[Dict[str, float]] = []
        width_deltas: List[Dict[str, float]] = []
        stability_deltas: List[Dict[str, float]] = []

        for xv in x_values:
            gb = bad[bad[view_col] == xv]
            gg = good[good[view_col] == xv]
            # Sampling
            if sample_rows is not None:
                if len(gb) > sample_rows:
                    gb = gb.sample(sample_rows, random_state=42)
                if len(gg) > sample_rows:
                    gg = gg.sample(sample_rows, random_state=42)
            def _sv(dfp: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
                if dfp.empty:
                    return np.zeros((0, 0)), np.zeros((0, 0)), []
                Xp = dfp[cols]
                Xt25 = imp25.transform(Xp)
                sv25 = shap.TreeExplainer(gbm25).shap_values(Xt25)
                if isinstance(sv25, list):
                    sv25 = sv25[0]
                Xt75 = imp75.transform(Xp)
                sv75 = shap.TreeExplainer(gbm75).shap_values(Xt75)
                if isinstance(sv75, list):
                    sv75 = sv75[0]
                limit = min(sv25.shape[1], sv75.shape[1], len(cols))
                return sv25[:, :limit], sv75[:, :limit], cols[:limit]
            sv25_b, sv75_b, cols_b = _sv(gb)
            sv25_g, sv75_g, cols_g = _sv(gg)
            limit = min(len(cols_b), len(cols_g))
            if limit == 0:
                continue
            cols_use = cols_b[:limit]
            sv25_b = sv25_b[:, :limit]
            sv75_b = sv75_b[:, :limit]
            sv25_g = sv25_g[:, :limit]
            sv75_g = sv75_g[:, :limit]

            center_b = (sv75_b + sv25_b) / 2.0
            width_b = (sv75_b - sv25_b) / 2.0
            stability_b = sv25_b - sv75_b
            center_g = (sv75_g + sv25_g) / 2.0
            width_g = (sv75_g - sv25_g) / 2.0
            stability_g = sv25_g - sv75_g

            lb = _agg_layers(center_b, cols_use)
            lg = _agg_layers(center_g, cols_use)
            wb = _agg_layers(width_b, cols_use)
            wg = _agg_layers(width_g, cols_use)
            sb = _agg_layers(stability_b, cols_use)
            sg = _agg_layers(stability_g, cols_use)
            # deltas
            keys = set(lb) | set(lg) | set(wb) | set(wg) | set(sb) | set(sg)
            cd = {k: lb.get(k, 0.0) - lg.get(k, 0.0) for k in keys}
            wd = {k: wb.get(k, 0.0) - wg.get(k, 0.0) for k in keys}
            sd = {k: sb.get(k, 0.0) - sg.get(k, 0.0) for k in keys}
            for k in keys:
                layers_union[k] = None
            center_deltas.append(cd)
            width_deltas.append(wd)
            stability_deltas.append(sd)

        if not center_deltas:
            continue

        layer_names = sorted(layers_union.keys())
        def _to_matrix(lst: List[Dict[str, float]]):
            M = np.zeros((len(lst), len(layer_names)), dtype=float)
            for i, d in enumerate(lst):
                for j, name in enumerate(layer_names):
                    M[i, j] = float(d.get(name, 0.0))
            return M
        M_center = _to_matrix(center_deltas)
        M_width = _to_matrix(width_deltas)
        M_stability = _to_matrix(stability_deltas)

        # Plot heatmaps of signed deltas
        for key, M in [("center", M_center), ("width", M_width), ("stability", M_stability)]:
            plt.figure(figsize=(max(6, len(x_values) * 0.25), max(3, len(layer_names) * 0.45)))
            vmax = np.max(np.abs(M)) + 1e-12
            ax = sns.heatmap(M.T, cmap="coolwarm", center=0.0, vmin=-vmax, vmax=vmax, cbar=True)
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names, rotation=90)
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_values[:len(x_values)], rotation=90)
            ax.set_xlabel(view_col)
            ax.set_ylabel("Layer")
            plt.title(f"Pair {pid}: {key} layer deltas (bad - good) by layer vs {view_col}")

            fname = os.path.join(out_dir, f"pair_{pid.replace('/', '-')}_{view_col}_deltas_{key}.png")
            plt.tight_layout()
            try:
                plt.savefig(fname, dpi=150)
                print(f"Saved pair delta heatmap: {fname}")
            except Exception as exc:
                print(f"Failed to save pair delta heatmap {fname}: {exc}")
            finally:
                plt.close()


