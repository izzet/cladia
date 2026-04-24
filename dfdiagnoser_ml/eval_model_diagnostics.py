import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import warnings
import traceback

from dfdiagnoser_ml.common import GLOBALS_DIR, get_quantiles, layer_key, add_special_features
from dfdiagnoser_ml.metrics import (
    compute_asymmetric_miss_score,
    compute_interval_quality_score,
    compute_prediction_interval_coverage_probability,
    compute_winkler_score,
    lenient_overlap_accuracy,
)


# Silence pandas SettingWithCopyWarning for attribution column writes
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)  # type: ignore[attr-defined]
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)  # type: ignore[attr-defined]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _load_artifacts(view_type: str, framework: str, target_col: str, feature_groups: List[int] | None, q_method: str, q_low: int, q_high: int, posix_only: bool = False) -> Tuple:
    models_dir = f"{GLOBALS_DIR}/{view_type}/models"

    path_infix, posix_suffix, q_suffix, fg_suffix = _dir_parts(view_type, framework, target_col, feature_groups, q_method, q_low, q_high, posix_only)

    print(f"[E4] Loading artifacts from {models_dir} (framework={framework}, target={target_col}, fg={fg_suffix or 'none'}, q_method={q_method}, q_low={q_low}, q_high={q_high})")
    # Mean pipeline + features 
    mean_pipe_path = f"{models_dir}/mean_pipeline_{path_infix}{posix_suffix}.joblib"
    mean_feats_path = f"{models_dir}/mean_features_{path_infix}{posix_suffix}.joblib"
    mean_pipe = joblib.load(mean_pipe_path) 
    mean_feats: List[str] = joblib.load(mean_feats_path)

    # Quantile pipelines + features 
    q_low_path = f"{models_dir}/quantile_pipeline_low_{path_infix}_{q_suffix}{posix_suffix}.joblib"
    q_high_path = f"{models_dir}/quantile_pipeline_high_{path_infix}_{q_suffix}{posix_suffix}.joblib"
    q_feats_path = f"{models_dir}/quantile_features_{path_infix}_{q_suffix}{posix_suffix}.joblib"
    q_low_pipe = joblib.load(q_low_path)
    q_high_pipe = joblib.load(q_high_path)
    q_feats: List[str] = joblib.load(q_feats_path)

    print(f"[E4] mean_feats={len(mean_feats)}; q_feats={len(q_feats)}")
    return mean_pipe, mean_feats, q_low_pipe, q_high_pipe, q_feats


def _predict_mean(mean_pipe, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cols = [c for c in feature_cols if c in df.columns]
    print(f"[E4] Mean features overlap: {len(cols)}/{len(feature_cols)} present in DF")
    X = df[cols]
    y_true = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
    y_pred = mean_pipe.predict(X)
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    return y_pred, {"abs_err": abs_err, "sq_err": sq_err, "cols": np.array(cols, dtype=object)}  # type: ignore


def _tree_shap_matrix(pipeline, X: pd.DataFrame) -> np.ndarray | None:
    try:
        import shap  # type: ignore
    except Exception:
        return None
    try:
        imputer = pipeline.named_steps["simpleimputer"]
    except Exception:
        return None
    # Try common model step names
    model = None
    for name in ("randomforestregressor", "gradientboostingregressor"):
        if name in pipeline.named_steps:
            model = pipeline.named_steps[name]
            break
    if model is None:
        return None
    # Coerce X to purely numeric numpy array to avoid dtype pitfalls with pandas extension dtypes
    try:
        X_num = X.apply(pd.to_numeric, errors="coerce") if isinstance(X, pd.DataFrame) else X
        if isinstance(X_num, pd.DataFrame):
            X_arr = X_num.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X_num, dtype=float)
        Xt = imputer.transform(X_arr)
    except Exception as exc:
        try:
            print("[E4][SHAP] imputer.transform failed. Debug info:")
            print(f"  X type={type(X)}, shape={(getattr(X, 'shape', None))}")
            if isinstance(X, pd.DataFrame):
                print(f"  X columns={len(X.columns)}; first5={list(X.columns[:5])}")
                na_counts = pd.isna(X).sum().sum()
                print(f"  X total NaNs={int(na_counts)}")
                dtypes = X.dtypes.astype(str).value_counts().to_dict()
                print(f"  X dtypes distribution={dtypes}")
            try:
                Xn = X.apply(pd.to_numeric, errors="coerce") if isinstance(X, pd.DataFrame) else None
                if isinstance(Xn, pd.DataFrame):
                    print(f"  After numeric coercion: NaNs={int(pd.isna(Xn).sum().sum())}; dtypes={Xn.dtypes.astype(str).value_counts().to_dict()}")
            except Exception:
                pass
            print(f"  imputer type={type(imputer)}")
            nfi = getattr(imputer, 'n_features_in_', None)
            stats = getattr(imputer, 'statistics_', None)
            print(f"  imputer.n_features_in_={nfi}; statistics_shape={(None if stats is None else getattr(stats, 'shape', None))}")
            print(f"  model type={type(model)}")
            print(f"  exception={exc}")
            traceback.print_exc()
        except Exception:
            pass
        return None
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[0]
        return sv
    except Exception as exc:
        try:
            print("[E4][SHAP] TreeExplainer failed. Debug info:")
            print(f"  Xt shape={(getattr(Xt, 'shape', None))}")
            print(f"  model type={type(model)}")
            print(f"  exception={exc}")
            traceback.print_exc()
        except Exception:
            pass
        return None


def _topk_from_vector(values: np.ndarray, names: List[str], top_k: int) -> List[Tuple[str, float]]:
    if values.ndim != 1:
        values = values.ravel()
    idx = np.argsort(-np.abs(values))[: min(top_k, len(values))]
    return [(names[i], float(values[i])) for i in idx]


def _topk_layers_from_vector(values: np.ndarray, names: List[str], top_k: int) -> List[Tuple[str, float]]:
    acc: Dict[str, float] = {}
    for j, v in enumerate(values):
        acc[layer_key(names[j])] = acc.get(layer_key(names[j]), 0.0) + float(abs(v))
    items = sorted(acc.items(), key=lambda x: x[1], reverse=True)
    return items[: min(top_k, len(items))]


def _attach_topk(df_out: pd.DataFrame, row_index: int, prefix: str, values_row: np.ndarray, feature_names: List[str], top_k: int) -> None:
    feats = _topk_from_vector(values_row, feature_names, top_k)
    lays = _topk_layers_from_vector(values_row, feature_names, top_k)
    for i, (name, val) in enumerate(feats, start=1):
        df_out.loc[row_index, f"top_{prefix}_feature_{i}"] = name
        df_out.loc[row_index, f"top_{prefix}_feature_{i}_value"] = float(val)
    for i, (name, val) in enumerate(lays, start=1):
        df_out.loc[row_index, f"top_{prefix}_layer_{i}"] = name
        df_out.loc[row_index, f"top_{prefix}_layer_{i}_value"] = float(val)


def _predict_quantiles(q_low_pipe, q_high_pipe, df: pd.DataFrame, feature_cols: List[str], target_col: str, q_method: str, q_pair: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols]
    q_low_pred = q_low_pipe.predict(X)
    q_high_pred = q_high_pipe.predict(X)
    q_min = np.minimum(q_low_pred, q_high_pred)
    q_max = np.maximum(q_low_pred, q_high_pred)
    center = (q_min + q_max) / 2.0
    width = q_max - q_min
    # True interval
    y_low_true, y_high_true = get_quantiles(df, target_col, q_method=q_method, q_pair=q_pair)
    y_true_int = np.stack([y_low_true, y_high_true], axis=1)
    y_pred_int = np.stack([q_min, q_max], axis=1)

    # R2 calculations
    from sklearn.metrics import r2_score
    try:
        true_center = (y_low_true + y_high_true) / 2.0
        true_width = y_high_true - y_low_true
        pred_center = center
        pred_width = width
        r2_center = r2_score(true_center, pred_center)
        r2_width = r2_score(true_width, pred_width)
    except Exception:
        r2_center = float("nan")
        r2_width = float("nan")

    # Metrics (dataset-level scalars)
    overlap = lenient_overlap_accuracy(y_true_int, y_pred_int)
    winkler = compute_winkler_score(y_pred_int, center, alpha=0.5, include_endpoints=True)
    iqs = compute_interval_quality_score(y_true_int, y_pred_int, lambda_weight=1.0, tolerance=0.0, width_normalization="dataset_mean")
    ams = compute_asymmetric_miss_score(y_true_int, y_pred_int, alpha=2.0)
    picp = compute_prediction_interval_coverage_probability(y_pred_int, center[:, None], include_endpoints=True)
    return (
        q_min,
        q_max,
        {
            "center": center,
            "width": width,
            "y_true_int": y_true_int,
            "y_pred_int": y_pred_int,
            "overlap": np.array([overlap]),
            "winkler": np.array([winkler]),
            "iqs": np.array([iqs]),
            "ams_alpha2": np.array([ams]),
            "picp": np.array([picp]),
            "r2_center": np.array([r2_center]),
            "r2_width": np.array([r2_width]),
            "cols": np.array(cols, dtype=object),  # type: ignore
        },
    )


def evaluate_holdout_per_epoch(
    framework: str,
    view_type: str = "epoch",
    target_col: str = "compute_time_frac_epoch",
    dataset_name: str = "ml_data_{framework}_holdout_full.parquet",
    output_name: str = "ml_data_{framework}_holdout_full_{infix}_evaluated.parquet",
    top_k: int = 8,
    q_method: str = "delta",
    q_low: int = 25,
    q_high: int = 75,
    feature_groups: List[int] | None = None,
    posix_only: bool = False,
) -> str:
    """Load saved models and evaluate per-epoch predictions + SHAP top-k attributions.

    Returns path to saved parquet with enriched columns.
    """
    mean_pipe, mean_feats, q_low_pipe, q_high_pipe, q_feats = _load_artifacts(view_type, framework, target_col, feature_groups, q_method, q_low, q_high)

    dataset_dir = f"{GLOBALS_DIR}/{view_type}/datasets"    
    in_path = f"{dataset_dir}/" + dataset_name.format(framework=framework)
    df = pd.read_parquet(in_path)
    if feature_groups is not None and 3 in feature_groups:
        df = add_special_features(df)
    print(f"[E4] Loaded dataset: shape={df.shape}; columns={len(df.columns)}")

    # Predictions
    y_pred_mean, mean_info = _predict_mean(mean_pipe, df, mean_feats, target_col)
    q_pair = (q_low, q_high)
    q_low_pred, q_high_pred, qinfo = _predict_quantiles(q_low_pipe, q_high_pipe, df, q_feats, target_col, q_method, q_pair)

    # Compose output DataFrame (copy to avoid mutating original)
    out = df.copy()
    out["y_pred_mean"] = y_pred_mean
    out["mean_abs_err"] = mean_info["abs_err"]
    out["mean_sq_err"] = mean_info["sq_err"]

    out[f"q{q_low}_pred"] = q_low_pred
    out[f"q{q_high}_pred"] = q_high_pred
    out["pred_center"] = qinfo["center"]
    out["pred_width"] = qinfo["width"]

    # Per-sample quantile diagnostics
    try:
        # Per-sample AMS
        from dfdiagnoser_ml.metrics import compute_asymmetric_miss_score as _ams

        out["ams_ps_alpha2"] = _ams(qinfo["y_true_int"], qinfo["y_pred_int"], alpha=2.0, return_per_sample=True)
    except Exception:
        pass

    # Global metrics duplicated for convenience
    out["quant_overlap_strict"] = float(qinfo["overlap"][0])
    out["quant_winkler_0p5"] = float(qinfo["winkler"][0])
    out["quant_iqs"] = float(qinfo["iqs"][0])
    out["quant_ams_alpha2"] = float(qinfo["ams_alpha2"][0])
    out["quant_picp"] = float(qinfo["picp"][0])
    out["quant_r2_center"] = float(qinfo["r2_center"][0])
    out["quant_r2_width"] = float(qinfo["r2_width"][0])

    # SHAP top-k for mean
    if isinstance(mean_feats, list) and len(mean_feats) > 0:
        mean_cols = list(mean_feats)
        X_mean = out.reindex(columns=mean_cols, fill_value=np.nan)
        print(f"[E4] X_mean shape={X_mean.shape}; NaNs={int(pd.isna(X_mean).sum().sum())}")
        sv_mean = _tree_shap_matrix(mean_pipe, X_mean)
        if sv_mean is not None and sv_mean.shape[1] == len(mean_cols):
            for i in range(len(out)):
                _attach_topk(out, out.index[i], "mean", sv_mean[i], mean_cols, top_k)  # type: ignore
        else:
            print("[E4] mean SHAP unavailable or width mismatch; skipping top_k for mean.")

    # SHAP top-k for quantiles (q_low, q_high) and derived center/width
    if isinstance(q_feats, list) and len(q_feats) > 0:
        q_cols = list(q_feats)
        X_q = out.reindex(columns=q_cols, fill_value=np.nan)
        print(f"[E4] X_q shape={X_q.shape}; NaNs={int(pd.isna(X_q).sum().sum())}")
        sv_q_low = _tree_shap_matrix(q_low_pipe, X_q)
        sv_q_high = _tree_shap_matrix(q_high_pipe, X_q)
        if sv_q_low is not None and sv_q_high is not None and sv_q_low.shape[1] == len(q_cols) and sv_q_high.shape[1] == len(q_cols):
            for i in range(len(out)):
                row_low = sv_q_low[i]
                row_high = sv_q_high[i]
                row_idx = out.index[i]
                _attach_topk(out, row_idx, f"q{q_low}", row_low, q_cols, top_k)  # type: ignore
                _attach_topk(out, row_idx, f"q{q_high}", row_high, q_cols, top_k)  # type: ignore
                _attach_topk(out, row_idx, "center", (row_high + row_low) / 2.0, q_cols, top_k)  # type: ignore
                _attach_topk(out, row_idx, "width", (row_high - row_low) / 2.0, q_cols, top_k)  # type: ignore
        else:
            print(f"[E4] quantile SHAP unavailable or width mismatch; skipping top_k for q{q_low}/q{q_high}/center/width.")

    # Save
    eval_dir = f"{dataset_dir}/evaluated"
    _ensure_dir(eval_dir)
    # Derive output filename from input dataset filename
    path_infix, posix_suffix, q_suffix, _ = _dir_parts(view_type, framework, target_col, feature_groups, q_method, q_low, q_high, posix_only)
    infix = f"{path_infix}_{q_suffix}{posix_suffix}"
    out_path = f"{eval_dir}/" + output_name.format(framework=framework, infix=infix)
    out.to_parquet(out_path, index=False)
    print(f"Saved evaluated dataframe: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    quantile_profiles = {
        "iqr": (25, 75),
        "tail": (90, 95),
    }

    parser = argparse.ArgumentParser(description="Evaluate saved models and write diagnostic outputs")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], required=True)
    parser.add_argument("--view_type", type=str, default="epoch")
    parser.add_argument("--target_col", type=str, default="compute_time_frac_epoch")
    parser.add_argument("--dataset_name", type=str, default="ml_data_{framework}_holdout_full.parquet")
    parser.add_argument("--output_name", type=str, default="ml_data_{framework}_holdout_full_{infix}_evaluated.parquet")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--quantile_profile", type=str, choices=sorted(quantile_profiles), default="iqr")
    parser.add_argument("--q_method", type=str, choices=["mc", "delta"], default="mc", help=argparse.SUPPRESS)
    parser.add_argument("--q_low", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--q_high", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--feature_groups", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--posix_only", action="store_true", help="Use only POSIX features")
    args = parser.parse_args()
    q_low, q_high = quantile_profiles[args.quantile_profile]
    if args.q_low is not None:
        q_low = args.q_low
    if args.q_high is not None:
        q_high = args.q_high

    evaluate_holdout_per_epoch(
        framework=args.framework,
        view_type=args.view_type,
        target_col=args.target_col,
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        top_k=args.top_k,
        q_method=args.q_method,
        q_low=q_low,
        q_high=q_high,
        feature_groups=args.feature_groups,
        posix_only=args.posix_only,
    )


