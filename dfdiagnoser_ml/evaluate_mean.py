import numpy as np
import pandas as pd
from math import sqrt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List

from dfdiagnoser_ml.common import layer_key, GROUND_TRUTH_BOTTLENECKS


def evaluate_on_holdout(pipeline, feature_cols: List[str], holdout_df: pd.DataFrame, target_col: str = "epoch_time_max") -> dict:
    """Evaluate fitted pipeline on the held-out configs and report pairwise ordering and deltas."""
    common = [c for c in feature_cols if c in holdout_df.columns]
    if not common:
        print("Holdout eval: no overlapping feature columns; skipping.")
        return
    X_hold = holdout_df[common]
    y_hold = holdout_df[target_col]
    y_pred = pipeline.predict(X_hold)

    mae = mean_absolute_error(y_hold, y_pred)
    rmse = sqrt(mean_squared_error(y_hold, y_pred))
    r2 = r2_score(y_hold, y_pred)
    print(f"Holdout -> MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    out = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    # Pairwise evaluation for before/after
    if "holdout_name" in holdout_df.columns:
        df_eval = holdout_df.copy()
        df_eval["y_pred"] = y_pred
        df_eval["pair_id"] = df_eval["holdout_name"].str.rsplit("_", n=1).str[0]
        ordering_ok = []
        delta_abs_err = []
        # Determine expected ordering based on target_col rules
        def _is_correct_order(pred_bad: float, pred_good: float) -> bool:
            # Default rule: higher is worse (bad > good)
            if str(target_col).endswith("_time_frac_epoch"):
                if str(target_col).startswith("compute_"):
                    # For compute utilization fraction, good should be higher than bad
                    return pred_bad < pred_good
                else:
                    # Unsupported *_time_frac_epoch target prefix
                    raise ValueError(f"Unsupported ordering rule for target_col='{target_col}'")
            return pred_bad > pred_good
        for pid, g in df_eval.groupby("pair_id"):
            bad = g[g["holdout_name"].str.endswith("_bad")]
            good = g[g["holdout_name"].str.endswith("_good")]
            if len(bad) == 0 or len(good) == 0:
                continue
            pred_bad = float(bad["y_pred"].mean())
            pred_good = float(good["y_pred"].mean())
            obs_bad = float(bad[target_col].mean())
            obs_good = float(good[target_col].mean())
            ordering_ok.append(1 if _is_correct_order(pred_bad, pred_good) else 0)
            delta_abs_err.append(abs((pred_bad - pred_good) - (obs_bad - obs_good)))
        if ordering_ok:
            oa = float(np.mean(ordering_ok))
            da = float(np.mean(delta_abs_err))
            print(f"Holdout pair ordering accuracy: {oa:.3f} over {len(ordering_ok)} pairs")
            print(f"Holdout pair delta abs error (mean): {da:.3f}")
            out.update({"pair_ordering_acc": oa, "pair_delta_abs_err": da, "pair_count": int(len(ordering_ok))})
    return out


def shap_layer_report(pipeline, df: pd.DataFrame, feature_cols: List[str], sample_rows: int = 1024) -> dict:
    """Compute SHAP on a sample and aggregate by coarse layer prefix to sanity-check attributions."""
    try:
        import shap  # type: ignore
    except Exception as exc:
        print(f"SHAP not available ({exc}); skipping SHAP report.")
        return {"available": False}

    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) == 0:
        print("No overlapping feature columns for SHAP.")
        return {"available": False}
    X = df[cols]
    if len(X) == 0:
        print("Empty dataframe for SHAP.")
        return {"available": False}
    Xs = X.sample(min(len(X), sample_rows), random_state=42)

    try:
        imputer = pipeline.named_steps["simpleimputer"]
        model = pipeline.named_steps["randomforestregressor"]
    except Exception:
        print("Pipeline steps not accessible for SHAP; skipping.")
        return {"available": False}

    Xt = imputer.transform(Xs)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[0]
    contrib = np.abs(sv).mean(axis=0)
    if contrib.ndim == 0:
        contrib = np.array([contrib])

    if len(cols) != contrib.shape[0]:
        print(f"Warning: SHAP feature length mismatch (cols={len(cols)} vs contrib={contrib.shape[0]}). Truncating to min.")
        limit = min(len(cols), contrib.shape[0])
        cols = cols[:limit]
        contrib = contrib[:limit]

    group = {}
    for idx, feat in enumerate(cols):
        key = layer_key(feat)
        group[key] = group.get(key, 0.0) + float(contrib[idx])
    total = sum(group.values()) + 1e-12
    top = sorted(((k, v / total) for k, v in group.items()), key=lambda x: x[1], reverse=True)[:10]
    print("Top SHAP layer groups (fraction of |contrib|):")
    for k, frac in top:
        print(f"  {k}: {frac:.3f}")

    # Also print top individual features
    feat_norm = (contrib / total)
    feat_top = sorted(zip(cols, feat_norm), key=lambda x: x[1], reverse=True)[:10]
    print("Top SHAP features (fraction of |contrib|):")
    for feat, frac in feat_top:
        print(f"  {feat}: {frac:.3f}")
    return {
        "available": True,
        "layers_group": group,
        "top_layers": top[:3],
        "top_features": feat_top[:3],
    }


def run_leak_checks(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], target_col: str = "epoch_time_max", groups_col: str = "run_id") -> None:
    """Simple leakage probes: drop configs and permute target to ensure R2 collapses."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    # X_test not used directly in leakage probes
    y_test = test_df[target_col]

    # Config-drop test
    no_cfg_cols = [c for c in feature_cols if not str(c).startswith("config_")]
    if not no_cfg_cols:
        print("Leak check (drop config_*) skipped: no non-config features present.")
    else:
        pipe2 = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=5, random_state=123, n_jobs=-1))
        pipe2.fit(train_df[no_cfg_cols], y_train)
        y_pred = pipe2.predict(test_df[no_cfg_cols])
        r2_nc = r2_score(y_test, y_pred)
        print(f"Leak check (drop config_*): Test R2={r2_nc:.3f} with {len(no_cfg_cols)} features")

    # Target permutation CV
    groups = train_df[groups_col]
    uniq = groups.nunique()
    if uniq >= 2:
        cv = GroupKFold(n_splits=min(5, uniq))
        rng = np.random.RandomState(0)
        y_perm = y_train.copy().to_numpy()
        rng.shuffle(y_perm)
        pipe3 = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=321, n_jobs=-1))
        scores = cross_validate(pipe3, X_train, y_perm, cv=cv, groups=groups, scoring={"r2": "r2"}, n_jobs=-1)
        print(f"Leak check (target permutation): CV R2 mean={scores['test_r2'].mean():.3f}")


def shap_holdout_pair_report(
    pipeline,
    holdout_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "epoch_time_max",
    top_k: int = 8,
    k_list: List[int] | None = None,
) -> dict:
    """For each holdout pair (bad/good), compute SHAP and show layer-level attributions and deltas."""
    if holdout_df.empty or "holdout_name" not in holdout_df.columns:
        print("No holdout pairs to report.")
        return {"available": False}
    try:
        import shap  # type: ignore
    except Exception as exc:
        print(f"SHAP not available ({exc}); skipping holdout pair report.")
        return {"available": False}

    try:
        imputer = pipeline.named_steps["simpleimputer"]
        model = pipeline.named_steps["randomforestregressor"]
    except Exception:
        print("Pipeline steps not accessible for SHAP; skipping holdout pair report.")
        return
    
    holdout_df = holdout_df.copy()
    holdout_df["pair_id"] = holdout_df["holdout_name"].str.rsplit("_", n=1).str[0]
    summaries = []
    hit_rate_results = {}
    k_list = sorted(set(k_list or [1, 3, 5, 10]))
    for pid, g in holdout_df.groupby("pair_id"):
        bad = g[g["holdout_name"].str.endswith("_bad")]
        good = g[g["holdout_name"].str.endswith("_good")]
        if len(bad) == 0 or len(good) == 0:
            continue

        cols = [c for c in feature_cols if c in g.columns]
        X_pair = g[cols]
        Xt = imputer.transform(X_pair)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[0]
        
        # --- Start of Bottleneck Hit Rate Logic (Per-Row) ---
        target_is_higher_is_better = "frac" in target_col and "compute" in target_col
        pos_map = {idx: i for i, idx in enumerate(g.index)}
        bad_pos = [pos_map[i] for i in bad.index if i in pos_map]

        feature_hit_count = {k: 0 for k in k_list}
        layer_hit_count = {k: 0 for k in k_list}
        nrows = len(bad_pos)

        if nrows > 0:
            bad_sv = sv[bad_pos]
            ground_truth_dict = GROUND_TRUTH_BOTTLENECKS.get(pid)
            if ground_truth_dict:
                ground_truth_prefixes = ground_truth_dict.get("feature_prefixes", [])
                ground_truth_layers = ground_truth_dict.get("layers", [])
                
                # Iterate over each row (epoch) in the bad configuration
                for i in range(nrows):
                    row_sv = bad_sv[i, :]
                    feature_shap_pairs = list(zip(cols, row_sv))
                    feature_shap_pairs.sort(key=lambda item: item[1], reverse=not target_is_higher_is_better)

                    for k in k_list:
                        top_k_features = [feature for feature, _ in feature_shap_pairs[:k]]
                        # Feature Prefix Hit for this row
                        if any(gt_prefix in top_feature for gt_prefix in ground_truth_prefixes for top_feature in top_k_features):
                            feature_hit_count[k] += 1
                        # Layer Hit for this row
                        top_k_layers = {layer_key(f) for f in top_k_features}
                        if any(gt_layer in top_layer for gt_layer in ground_truth_layers for top_layer in top_k_layers):
                            layer_hit_count[k] += 1

                hit_rate_results[pid] = {
                    "by_k": {
                        k: {
                            "feature_hit_count": feature_hit_count[k],
                            "layer_hit_count": layer_hit_count[k],
                            "nrows": nrows,
                        }
                        for k in k_list
                    }
                }

                if 3 in k_list:
                    print(f"  [Mean Model Bottleneck Hit Rate Check for '{pid}']")
                    print(f"    Feature Hits: {feature_hit_count[3]}/{nrows} ({feature_hit_count[3]/nrows if nrows > 0 else 0:.2f})")
                    print(f"    Layer Hits:   {layer_hit_count[3]}/{nrows} ({layer_hit_count[3]/nrows if nrows > 0 else 0:.2f})")
        # --- End of Bottleneck Hit Rate Logic ---

        contrib = np.abs(sv)
        if contrib.shape[1] != len(cols):
            limit = min(contrib.shape[1], len(cols))
            contrib = contrib[:, :limit]
            cols = cols[:limit]

        idx_bad = bad.index
        idx_good = good.index
        pos_map = {idx: i for i, idx in enumerate(g.index)}
        bad_pos = [pos_map[i] for i in idx_bad if i in pos_map]
        good_pos = [pos_map[i] for i in idx_good if i in pos_map]
        if not bad_pos or not good_pos:
            continue

        def agg_stats(pos_list):
            mean_contrib = contrib[pos_list].mean(axis=0)
            total = float(mean_contrib.sum()) + 1e-12
            layer_sums = {}
            for j, name in enumerate(cols):
                key = layer_key(name)
                layer_sums[key] = layer_sums.get(key, 0.0) + float(mean_contrib[j])
            layer_norm = {k: v / total for k, v in layer_sums.items()}
            feat_norm = (mean_contrib / total)
            return layer_norm, feat_norm

        bad_layers, bad_feat_norm = agg_stats(bad_pos)
        good_layers, good_feat_norm = agg_stats(good_pos)

        # Compact layer table: [layer, bad_share, good_share, delta]
        all_keys = sorted(set(bad_layers) | set(good_layers))
        layer_rows = []
        for k in all_keys:
            b = bad_layers.get(k, 0.0)
            g = good_layers.get(k, 0.0)
            d = b - g
            layer_rows.append((k, b, g, d))
        layer_rows.sort(key=lambda x: abs(x[3]), reverse=True)

        y_bad_pred = float(pipeline.predict(bad[cols]).mean())
        y_good_pred = float(pipeline.predict(good[cols]).mean())
        y_bad = float(bad[target_col].mean())
        y_good = float(good[target_col].mean())

        print(f"\nHoldout pair: {pid}")
        print(f"  Observed means: bad={y_bad:.3f}, good={y_good:.3f}, delta={y_bad - y_good:.3f}")
        print(f"  Predicted means: bad={y_bad_pred:.3f}, good={y_good_pred:.3f}, delta={y_bad_pred - y_good_pred:.3f}")
        print("  Layers (bad_share, good_share, delta):")
        for k, b, g, d in layer_rows[:top_k]:
            print(f"    {k}: bad={b:.3f}, good={g:.3f}, delta={d:+.3f}")

        # Also print top features by their delta impact
        feat_rows = []
        fb = bad_feat_norm.tolist()
        fg = good_feat_norm.tolist()
        for idx_f, fname in enumerate(cols):
            vb = float(fb[idx_f])
            vg = float(fg[idx_f])
            delta = vb - vg
            feat_rows.append((fname, vb, vg, delta))
        feat_rows.sort(key=lambda x: abs(x[3]), reverse=True)

        print("  Features (bad_share, good_share, delta):")
        for fname, vb, vg, fdelta in feat_rows[:top_k]:
            print(f"    {fname}: bad={vb:.3f}, good={vg:.3f}, delta={fdelta:+.3f}")
        summaries.append({
            "pair_id": pid,
            "obs_delta": float(y_bad - y_good),
            "pred_delta": float(y_bad_pred - y_good_pred),
            "top_layers": layer_rows[:3],
            "top_features": feat_rows[:3],
        })
    return {"available": True, "pairs": summaries, "pair_count": len(summaries), "bottleneck_hit_results": hit_rate_results}
