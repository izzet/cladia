import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from typing import List, Optional, Tuple
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

from dfdiagnoser_ml.common import select_epoch_features, prune_empty_features, get_quantiles
from dfdiagnoser_ml.metrics import (
    compute_asymmetric_miss_score,
    compute_winkler_score,
    compute_coverage_width_criterion,
    median_absolute_percentage_error as mdape_metric,
)


def train_quantile_models(X: pd.DataFrame, y_low: np.ndarray, y_high: np.ndarray,
                         alpha_low: float, alpha_high: float, groups: pd.Series,
                         model_type: str = 'gb', cv_folds: int = 5):
    """
    Train quantile regression models with cross validation support.

    Parameters:
    - model_type: 'gb' (GradientBoosting), 'catboost', or 'lightgbm'
    - cv_folds: Number of CV folds (default 5)
    """
    # Create models based on type
    if model_type == 'gb':
        model_low = GradientBoostingRegressor(loss="quantile", alpha=alpha_low,
                                            random_state=42, n_estimators=600, max_depth=3)
        model_high = GradientBoostingRegressor(loss="quantile", alpha=alpha_high,
                                             random_state=42, n_estimators=600, max_depth=3)
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        pipe_low = make_pipeline(imputer, model_low)
        pipe_high = make_pipeline(imputer, model_high)
        return_models = False

    elif model_type == 'catboost':
        if CatBoostRegressor is None:
            raise ImportError(
                "catboost is required for model_type='catboost' but is not installed."
            )
        model_low = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha_low}",
            random_seed=42,
            iterations=600,
            depth=6,
            learning_rate=0.05,
            verbose=False
        )
        model_high = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha_high}",
            random_seed=42,
            iterations=600,
            depth=6,
            learning_rate=0.05,
            verbose=False
        )
        # CatBoost handles NaNs natively, so we don't need a pipeline with imputer
        pipe_low = model_low
        pipe_high = model_high
        return_models = True

    elif model_type == 'lightgbm':
        zero_as_missing = True
        model_low = LGBMRegressor(objective="quantile", alpha=alpha_low, random_state=42,
                                n_estimators=600, max_depth=3, verbosity=-1, zero_as_missing=zero_as_missing)
        model_high = LGBMRegressor(objective="quantile", alpha=alpha_high, random_state=42,
                                 n_estimators=600, max_depth=3, verbosity=-1, zero_as_missing=zero_as_missing)
        # LightGBM handles NaNs natively, so we don't need a pipeline with imputer
        pipe_low = model_low
        pipe_high = model_high
        return_models = True

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'gb', 'catboost', 'lightgbm'")

    # Perform cross validation
    mask = np.isfinite(y_low) & np.isfinite(y_high)
    X_fit = X.loc[mask] if hasattr(X, 'loc') else X[mask]
    groups_cv = groups.loc[mask] if hasattr(groups, 'loc') else groups[mask]

    uniq_groups = pd.Series(groups_cv).nunique()
    if uniq_groups >= 2:
        cv = GroupKFold(n_splits=min(cv_folds, uniq_groups))
        mdape_scorer = make_scorer(mdape_metric, greater_is_better=False)
        scoring = {
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mape": "neg_mean_absolute_percentage_error",
            "mdea": "neg_median_absolute_error",
            "mdape": mdape_scorer,
        }
        scores_low = cross_validate(pipe_low, X_fit, y_low[mask], cv=cv, groups=groups_cv,
                        scoring=scoring, n_jobs=-1)
        scores_high = cross_validate(pipe_high, X_fit, y_high[mask], cv=cv, groups=groups_cv,
                        scoring=scoring, n_jobs=-1)
        
        def format_scores(scores, prefix):
            return (
                f"{prefix} MAE={-scores['test_mae'].mean():.3f}, "
                f"{prefix} R2={scores['test_r2'].mean():.3f}, "
                f"{prefix} RMSE={-scores['test_rmse'].mean():.3f}, "
                f"{prefix} MAPE={-scores['test_mape'].mean():.3%}, "
                f"{prefix} MDE={-scores['test_mdea'].mean():.3f}, "
                f"{prefix} MDAPE={-scores['test_mdape'].mean():.3%}"
            )

        print(f"{model_type.upper()} Train CV -> {format_scores(scores_low, 'Low')}")
        print(f"{model_type.upper()} Train CV -> {format_scores(scores_high, 'High')}")
    else:
        print(f"{model_type.upper()} Train CV skipped (insufficient groups)")

    # Fit the models
    if return_models:
        model_low.fit(X_fit, y_low[mask])
        model_high.fit(X_fit, y_high[mask])
        return model_low, model_high
    else:
        pipe_low.fit(X_fit, y_low[mask])
        pipe_high.fit(X_fit, y_high[mask])
        return pipe_low, pipe_high


def _apply_calibration(y_pred: np.ndarray, scale: float, shift: float) -> np.ndarray:
    c = (y_pred[:, 0] + y_pred[:, 1]) / 2.0
    w = (y_pred[:, 1] - y_pred[:, 0]) * max(scale, 0.0)
    L = c - w/2.0 + shift
    U = c + w/2.0 + shift
    return np.column_stack([np.minimum(L, U), np.maximum(L, U)])


def _tune_calibration(y_true: np.ndarray, y_pred: np.ndarray, metric: str, y_point: np.ndarray | None):
    scales = [0.8, 1.0, 1.2]
    tw = (y_true[:, 1] - y_true[:, 0]).mean()
    shifts = [-0.1*tw, 0.0, 0.1*tw]
    val_mask = (np.arange(len(y_true)) % 2) == 0
    best = (None, np.inf)
    for s in scales:
        for d in shifts:
            P = _apply_calibration(y_pred, s, d)
            if metric == 'AMS':
                val = compute_asymmetric_miss_score(y_true[val_mask], P[val_mask], alpha=2.0)
            elif metric == 'WINKLER':
                val = compute_winkler_score(P[val_mask], y_point[val_mask], alpha=0.5, include_endpoints=True)
            elif metric == 'CWC':
                val = compute_coverage_width_criterion(P[val_mask], y_point[val_mask], alpha=0.5, lam=10.0, eta=50.0, include_endpoints=True)
            else:
                raise ValueError('Unknown metric')
            if val < best[1]:
                best = ((s, d), val)
    return best[0]


def run_quantile_training_and_calibration(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "epoch_time_max",
    feature_groups: Optional[List[int]] = None,
    posix_only: bool = False,
    q_method: str = "delta",
    q_low: int = 25,
    q_high: int = 75,
    model_type: str = 'lightgbm',
    cv_folds: int = 5,
):
    feature_cols = select_epoch_features(train_df, target_col=target_col, posix_only=posix_only, feature_groups=feature_groups)
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    feature_cols = prune_empty_features(train_df, test_df, feature_cols)
    print(f"Selected {len(feature_cols)} features for quantiles")

    y_low_train, y_high_train = get_quantiles(train_df, target_col, q_method=q_method, q_pair=(q_low, q_high))
    alpha_low = float(q_low) / 100.0
    alpha_high = float(q_high) / 100.0
    groups = train_df.get("workload_config_id", pd.Series(np.arange(len(train_df))))
    q_low_pipe, q_high_pipe = train_quantile_models(train_df[feature_cols], y_low_train, y_high_train,
                                alpha_low, alpha_high, groups, model_type=model_type, cv_folds=cv_folds)

    # Calculate and print metrics for train and test sets
    from dfdiagnoser_ml.evaluate_quantiles import _calculate_quantile_metrics, print_quantile_metrics
    train_metrics = _calculate_quantile_metrics(
        train_df, q_low_pipe, q_high_pipe, feature_cols, target_col, q_method, (q_low, q_high)
    )
    print_quantile_metrics(train_metrics, "Train")
    test_metrics = _calculate_quantile_metrics(
        test_df, q_low_pipe, q_high_pipe, feature_cols, target_col, q_method, (q_low, q_high)
    )
    print_quantile_metrics(test_metrics, "Test")

    # Combine metrics into a single dictionary with prefixes
    metrics = {f"q_{k}_train": v for k, v in train_metrics.items()}
    metrics.update({f"q_{k}_test": v for k, v in test_metrics.items()})
    
    # Perform calibration on the test set and add to metrics
    y_low_true_test, y_high_true_test = get_quantiles(test_df, target_col, q_method=q_method, q_pair=(q_low, q_high))
    mask_test = np.isfinite(y_low_true_test) & np.isfinite(y_high_true_test)
    if np.any(mask_test):
        qlow_pred_test = q_low_pipe.predict(test_df[feature_cols][mask_test])
        qhigh_pred_test = q_high_pipe.predict(test_df[feature_cols][mask_test])
        y_true_test = np.column_stack([y_low_true_test[mask_test], y_high_true_test[mask_test]])
        y_pred_test = np.column_stack([np.minimum(qlow_pred_test, qhigh_pred_test), np.maximum(qlow_pred_test, qhigh_pred_test)])
        y_point_test = (y_true_test[:, 0] + y_true_test[:, 1]) / 2.0

        cal_ams = _tune_calibration(y_true_test, y_pred_test, 'AMS', y_point=None)
        cal_win = _tune_calibration(y_true_test, y_pred_test, 'WINKLER', y_point=y_point_test)
        cal_cwc = _tune_calibration(y_true_test, y_pred_test, 'CWC', y_point=y_point_test)
        
        P_ams = _apply_calibration(y_pred_test, cal_ams[0], cal_ams[1])
        P_win = _apply_calibration(y_pred_test, cal_win[0], cal_win[1])
        P_cwc = _apply_calibration(y_pred_test, cal_cwc[0], cal_cwc[1])
        
        metrics["q_calib_ams_test"] = compute_asymmetric_miss_score(y_true_test, P_ams, alpha=2.0)
        metrics["q_calib_winkler_test"] = compute_asymmetric_miss_score(y_true_test, P_win, alpha=2.0)
        metrics["q_calib_cwc_test"] = compute_asymmetric_miss_score(y_true_test, P_cwc, alpha=2.0)
        
        print(f"Quantile calib summary (Test): AMS[{metrics['q_calib_ams_test']:.3f}] vs "
              f"Winkler[{metrics['q_calib_winkler_test']:.3f}] vs CWC[{metrics['q_calib_cwc_test']:.3f}]")

    return q_low_pipe, q_high_pipe, feature_cols, metrics

