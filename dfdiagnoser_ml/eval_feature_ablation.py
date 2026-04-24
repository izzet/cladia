import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from collections import Counter
from typing import List, Tuple
import os

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from dfdiagnoser_ml.common import get_feature_group, get_quantiles, layer_key, select_epoch_features, prune_empty_features



def analyze_harmful_features_pca(
    df: pd.DataFrame, top_features: List[str], target_col: str, framework_name: str, quantile_label: str, posix_only: bool
):
    """Performs PCA on a set of features and plots the results against the target column."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[PCA] Skipping PCA plot generation: matplotlib not installed. Please run `pip install matplotlib`.")
        return

    unique_features = sorted(list(set(top_features)))
    label = quantile_label.upper()
    print(f"\n--- Performing PCA on {len(unique_features)} unique top harmful features for {label} ---")
    if len(unique_features) < 2:
        print(f"[PCA] Skipping for {label}: at least 2 features are required for PCA.")
        return

    pca_df = df[unique_features + [target_col]].dropna(subset=[target_col])
    X = pca_df[unique_features]
    y = pca_df[target_col]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5)),
        ]
    )
    principal_components = pipeline.fit_transform(X)
    
    pca = pipeline.named_steps["pca"]
    loadings = pca.components_
    explained_variance_ratios = pca.explained_variance_ratio_

    print(f"\n[PCA] Top 5 feature contributions for each of the 5 principal components:")
    for i, component in enumerate(loadings):
        print(f"\n--- Principal Component {i+1} (Explains {explained_variance_ratios[i]:.2%} of variance) ---")
        feature_loadings = pd.Series(component, index=unique_features)
        top_features = feature_loadings.abs().nlargest(5)
        print(feature_loadings[top_features.index].to_string())

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label=target_col)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    total_explained_variance = np.sum(pca.explained_variance_ratio_)
    plt.title(
        f"PCA of Top Harmful {label} Features for {framework_name.capitalize()} (PC1 vs PC2)\n"
        f"Total Explained Variance (5 components): {total_explained_variance:.2%}"
    )
    plt.grid(True)

    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    posix_suffix = "_posix" if posix_only else ""
    output_path = f"{output_dir}/pca_harmful_features_{framework_name}_{quantile_label}{posix_suffix}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PCA plot to {output_path}")

def analyze_pc1_impact(
    df: pd.DataFrame, top_features: List[str], target_col: str, framework_name: str, quantile_label: str, posix_only: bool
):
    """Analyzes the first principal component's feature contributions and its relationship to the target."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    unique_features = sorted(list(set(top_features)))
    label = quantile_label.upper()
    print(f"\n--- PC1 Impact Analysis for {label} Features ---")
    if len(unique_features) < 2:
        return

    pca_df = df[unique_features + [target_col]].dropna(subset=[target_col])
    X = pca_df[unique_features]
    y = pca_df[target_col]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5)),
        ]
    )
    principal_components = pipeline.fit_transform(X)

    pc1_loadings = pipeline.named_steps["pca"].components_[0]
    pc1_component_values = principal_components[:, 0]

    impact_df = pd.DataFrame({"feature": unique_features, "pc1_loading": pc1_loadings})
    impact_df["correlation_with_target"] = [X[f].corr(y) for f in unique_features]
    impact_df = impact_df.reindex(impact_df.pc1_loading.abs().sort_values(ascending=False).index).head(10)

    print("\n[PC1] Top 10 feature contributions to Principal Component 1:")
    print(impact_df.to_string(index=False))

    plt.figure(figsize=(12, 10))
    plt.scatter(pc1_component_values, y, alpha=0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel(target_col)
    plt.title(f"PC1 vs. Target for Top Harmful {label} Features ({framework_name.capitalize()})")
    plt.grid(True)

    output_dir = "visualizations"
    posix_suffix = "_posix" if posix_only else ""
    output_path = f"{output_dir}/pc1_vs_target_{framework_name}_{quantile_label}{posix_suffix}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PC1 vs. Target plot to {output_path}")


# --- New methodology functions ---

def evaluate_feature_set_performance(df: pd.DataFrame, target_col: str, posix_only: bool):
    """
    Evaluates the performance of a feature set by comparing a baseline model
    with a model using PCA-transformed features.
    """
    feature_set_name = "POSIX-only Features" if posix_only else "All Features"
    print(f"\n--- Evaluating Predictive Performance for: {feature_set_name} ---")

    # 1. Select features and prepare data
    feature_cols = select_epoch_features(df, target_col, posix_only=posix_only)
    feature_cols = prune_empty_features(df, df, feature_cols)
    
    # Use get_quantiles to create the correct target for the quantile regressor
    clean_df = df.dropna(subset=[target_col]).copy()
    y_q25, _ = get_quantiles(clean_df, target_col, q_method='mc', q_pair=(25, 75))
    
    # Align X and y, dropping NaNs from the target
    finite_y_mask = np.isfinite(y_q25)
    y = y_q25[finite_y_mask]
    X_df = clean_df[finite_y_mask]
    X = X_df[feature_cols]

    # Impute NaNs before splitting
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Evaluate a baseline model (no PCA)
    print("\n--- Baseline Model (Ridge Regression) ---")
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # ('ridge', Ridge(random_state=42)),
        ('gb', GradientBoostingRegressor(loss="quantile", alpha=0.25,
                                            random_state=42, n_estimators=600, max_depth=3))
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_baseline = baseline_pipeline.predict(X_test)
    r2_baseline = r2_score(y_test, y_pred_baseline)
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)
    print(f"Baseline Model Test Set R-squared: {r2_baseline:.4f}")
    print(f"Baseline Model Test Set MSE: {mse_baseline:.4f}")

    # 3. Evaluate model with PCA, using GridSearchCV to find the best # of components
    # that explain at least 90% of the variance.
    print("\n--- PCA-based Model (Gradient Boosting Regressor) ---")

    # First, determine the minimum number of components to explain 90% of the variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    min_components_for_90_variance = np.argmax(cumulative_variance >= 0.90) + 1
    
    print(f"Minimum components to explain 90% variance: {min_components_for_90_variance}")

    # Now, build the pipeline and search space for GridSearchCV
    pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('gb', GradientBoostingRegressor(loss="quantile", alpha=0.25,
                                            random_state=42, n_estimators=600, max_depth=3))
    ])

    # The search space starts from the minimum components required for 90% variance
    param_grid = {'pca__n_components': range(min_components_for_90_variance, X_train.shape[1] + 1)}

    grid_search = GridSearchCV(pca_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best number of components found: {grid_search.best_params_['pca__n_components']}")
    print(f"Best Cross-Validation R-squared: {grid_search.best_score_:.4f}")

    best_pca_model = grid_search.best_estimator_
    y_pred_pca = best_pca_model.predict(X_test)
    r2_pca = r2_score(y_test, y_pred_pca)
    mse_pca = mean_squared_error(y_test, y_pred_pca)

    # Get explained variance and unique features from the best model
    pca_transformer = best_pca_model.named_steps['pca']
    explained_variance = pca_transformer.explained_variance_ratio_
    total_explained_variance = sum(explained_variance)

    components = pca_transformer.components_
    all_top_contributing_features = set()
    for component in components:
        feature_loadings = pd.Series(component, index=feature_cols)
        top_features = feature_loadings.abs().nlargest(5)
        all_top_contributing_features.update(top_features.index.tolist())
    num_unique_features = len(all_top_contributing_features)

    print(f"\nVariance explained by each component (up to 99% cumulative):")
    cumulative_variance_report = 0.0
    for i, var in enumerate(explained_variance):
        cumulative_variance_report += var
        print(f"  - PC {i+1}: {var:.2%} (Cumulative: {cumulative_variance_report:.2%})")
        if cumulative_variance_report >= 0.99:
            print("  (Stopping report at >99% cumulative variance)")
            break
    
    print(f"PCA-based Model Test Set R-squared: {r2_pca:.4f}")
    print(f"PCA-based Model Test Set MSE: {mse_pca:.4f}")
    print(f"Total Variance Explained by PCA: {total_explained_variance:.2%}")
    print(f"Unique Features in Top Components: {num_unique_features}")

    # Print feature breakdown for the selected components
    print("\n--- Top Features in Selected Principal Components ---")
    for i, component in enumerate(components):
        print(f"\n--- Principal Component {i+1} ---")
        feature_loadings = pd.Series(component, index=feature_cols)
        top_features = feature_loadings.abs().nlargest(5)
        print(feature_loadings[top_features.index].to_string())
    
    return all_top_contributing_features


def compare_feature_lists(features_all: set, features_posix: set):
    """Compares and prints the differences between two feature lists."""
    print("\n--- Comparison of Top Contributing Features from PCA Models ---")
    
    all_only = sorted(list(features_all - features_posix))
    posix_only = sorted(list(features_posix - features_all)) # Should be empty in theory
    in_both = sorted(list(features_all & features_posix))

    print(f"\nTop features found ONLY in the 'All Features' model ({len(all_only)}):")
    for feature in all_only:
        print(f"  - {feature}")

    if posix_only:
        print(f"\nTop features found ONLY in the 'POSIX-only' model ({len(posix_only)}):")
        for feature in posix_only:
            print(f"  - {feature}")

    print(f"\nTop features found in BOTH models ({len(in_both)}):")
    for feature in in_both:
        print(f"  - {feature}")


def discover_robust_harmful_features(
    full_train_df: pd.DataFrame,
    target_col: str,
    quantile_to_analyze: int,
    n_splits: int = 5,
    top_k: int = 10,
    posix_only: bool = False,
) -> List[str]:
    """
    Performs GroupKFold cross-validation to find a robust list of top-k harmful features.
    This function's ONLY purpose is to identify a stable set of features for later analysis.
    """
    print(f"\n--- Discovering Robust Harmful Features for Q{quantile_to_analyze} ---")
    
    clean_df = full_train_df.dropna(subset=[target_col, 'run_id']).copy()
    alpha = quantile_to_analyze / 100.0
    all_top_features_for_quantile: List[str] = []

    groups = clean_df['run_id']
    gkf = GroupKFold(n_splits=n_splits)

    print(f"Performing {n_splits}-fold cross-validation for feature discovery...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(clean_df, groups=groups)):
        train_fold_df = clean_df.iloc[train_idx]
        val_fold_df = clean_df.iloc[val_idx]

        feature_cols = select_epoch_features(train_fold_df, target_col, posix_only=posix_only)
        feature_cols = prune_empty_features(train_fold_df, val_fold_df, feature_cols)
        
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_train = imputer.fit_transform(train_fold_df[feature_cols])
        X_val = imputer.transform(val_fold_df[feature_cols])

        y_train_quantile, _ = get_quantiles(train_fold_df, target_col, q_method='mc', q_pair=(quantile_to_analyze, 99))
        
        y_is_finite = np.isfinite(y_train_quantile)
        model = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, n_estimators=200, max_depth=4, random_state=42
        )
        model.fit(X_train[y_is_finite], y_train_quantile[y_is_finite])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        mean_shap = np.mean(shap_values, axis=0)
        
        shap_summary = pd.DataFrame({'feature': feature_cols, 'mean_shap': mean_shap})
        top_harmful = shap_summary.sort_values('mean_shap', ascending=True).head(top_k)
        
        all_top_features_for_quantile.extend(top_harmful['feature'].tolist())

    feature_frequency = Counter(all_top_features_for_quantile)
    robust_features = [f for f, count in feature_frequency.items() if count >= 2]
    print(f"Discovered {len(robust_features)} robust harmful features (appeared in >= 2 folds).")
    
    return robust_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze feature-set ablations and PCA structure.")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], required=True, help="Framework to analyze.")
    parser.add_argument("--in_path", type=str, help="Path to the training dataset parquet file.")
    parser.add_argument("--target_col", type=str, default="compute_time_frac_epoch", help="Target column for analysis.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for GroupKFold.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top harmful features to identify per fold.")
    parser.add_argument("--q_low", type=int, default=25, help="Lower quantile for analysis.")
    parser.add_argument("--q_high", type=int, default=75, help="Higher quantile for analysis.")
    parser.add_argument("--posix_only", action="store_true", help="Only analyze POSIX features.")
    args = parser.parse_args()

    # --- Main Workflow ---
    
    # 1. Load data
    if args.in_path:
        input_path = args.in_path
    else:
        from dfdiagnoser_ml.common import GLOBALS_DIR
        input_path = f"{GLOBALS_DIR}/epoch/datasets/ml_data_{args.framework}_train_full.parquet"
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)

    # 2. Discover robust harmful features for each quantile using cross-validation
    # robust_harmful_features_q_low = discover_robust_harmful_features(
    #     full_train_df=df,
    #     target_col=args.target_col,
    #     quantile_to_analyze=args.q_low,
    #     n_splits=args.n_splits,
    #     top_k=args.top_k,
    #     posix_only=args.posix_only,
    # )
    
    # robust_harmful_features_q_high = discover_robust_harmful_features(
    #     full_train_df=df,
    #     target_col=args.target_col,
    #     quantile_to_analyze=args.q_high,
    #     n_splits=args.n_splits,
    #     top_k=args.top_k,
    #     posix_only=args.posix_only,
    # )

    # # Combine for an overall "harmful" feature set
    # combined_robust_features = sorted(list(set(robust_harmful_features_q_low + robust_harmful_features_q_high)))
    
    # 3. Evaluate and compare predictive performance of feature sets
    top_features_all = evaluate_feature_set_performance(df, args.target_col, posix_only=False) # All features
    top_features_posix = evaluate_feature_set_performance(df, args.target_col, posix_only=True)  # POSIX-only features
    
    # 4. Compare the top contributing features discovered by the PCA in each model
    compare_feature_lists(top_features_all, top_features_posix)
