import numpy as np
import pandas as pd
from math import sqrt
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Optional

from dfdiagnoser_ml.common import select_epoch_features, prune_empty_features, drop_nonfinite_target
from dfdiagnoser_ml.metrics import median_absolute_percentage_error as mdape_metric


def run_mean_training(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str = "epoch_time_max", 
    feature_groups: Optional[List[int]] = None,
    posix_only: bool = False,
):
    feature_cols = select_epoch_features(train_df, target_col=target_col, posix_only=posix_only, feature_groups=feature_groups)
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    feature_cols = prune_empty_features(train_df, test_df, feature_cols)
    print(f"Selected {len(feature_cols)} features for mean model")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    groups = train_df.get("workload_config_id", pd.Series(np.arange(len(train_df))))
    # groups = train_df.get("workload_name", pd.Series(np.arange(len(train_df))))

    pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1),
    )

    uniq = pd.Series(groups).nunique()
    if uniq >= 2:
        cv = GroupKFold(n_splits=min(5, uniq))
        mdape_scorer = make_scorer(mdape_metric, greater_is_better=False)
        scoring = {
            "mae": "neg_mean_absolute_error", 
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mape": "neg_mean_absolute_percentage_error",
            "mdea": "neg_median_absolute_error",
            "mdape": mdape_scorer,
        }
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
        print(f"Mean model Train CV -> MAE={-scores['test_mae'].mean():.3f}, R2={scores['test_r2'].mean():.3f}, "
              f"RMSE={-scores['test_rmse'].mean():.3f}, MAPE={-scores['test_mape'].mean():.3%}, "
              f"MDE={-scores['test_mdea'].mean():.3f}, MDAPE={-scores['test_mdape'].mean():.3%}")
    else:
        print("Mean model CV skipped (insufficient groups)")

    pipe.fit(X_train, y_train)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    metrics = {
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "rmse_train": sqrt(mean_squared_error(y_train, y_train_pred)),
        "r2_train": r2_score(y_train, y_train_pred),
        "mdea_train": median_absolute_error(y_train, y_train_pred),
        "mape_train": mean_absolute_percentage_error(y_train, y_train_pred),
        "mdape_train": mdape_metric(y_train, y_train_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "rmse_test": sqrt(mean_squared_error(y_test, y_test_pred)),
        "r2_test": r2_score(y_test, y_test_pred),
        "mdea_test": median_absolute_error(y_test, y_test_pred),
        "mape_test": mean_absolute_percentage_error(y_test, y_test_pred),
        "mdape_test": mdape_metric(y_test, y_test_pred),
    }

    print(f"Mean model Test -> MAE={metrics['mae_test']:.3f}, RMSE={metrics['rmse_test']:.3f}, R2={metrics['r2_test']:.3f}, MDE={metrics['mdea_test']:.3f}, MAPE={metrics['mape_test']:.3%}, MDAPE={metrics['mdape_test']:.3%}")
    return pipe, feature_cols, metrics


