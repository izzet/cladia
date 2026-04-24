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

from dfdiagnoser_ml.common import get_quantiles, layer_key, select_epoch_features, prune_empty_features


def get_feature_group(feature: str) -> str:
    """Groups features based on layer and primary metric, with special handling for u_/o_ prefixes."""
    lk = layer_key(feature)

    if feature.startswith(("u_", "o_")):
        # e.g., u_reader_posix_lustre_time_frac_self -> u_reader_posix_lustre
        prefix = feature.split("_")[0]
        return f"{prefix}_{lk}"
    
    if feature.startswith(lk + "_"):
        remainder = feature.replace(lk + "_", "", 1)
        primary_metric = remainder.split("_")[0]
        return f"{lk}_{primary_metric}"

    # Fallback for simple features or where the layer key is the whole prefix
    return feature


def analyze_harmful_features_pca(
    df: pd.DataFrame,
    top_features: List[str],
    target_col: str,
    framework_name: str,
    quantile: int,
    posix_only: bool,
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
    print(f"\n--- Performing PCA on {len(unique_features)} unique top harmful features for Q{quantile} ---")
    if len(unique_features) < 2:
        print("[PCA] Skipping: at least 2 features are required for PCA.")
        return

    # Prepare data
    pca_df = df[unique_features + [target_col]].dropna(subset=[target_col])
    X = pca_df[unique_features]
    y = pca_df[target_col]

    # Pipeline for preprocessing and PCA
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5)),
        ]
    )

    # Fit and transform
    principal_components = pipeline.fit_transform(X)

    # Print feature loadings for each component
    pca = pipeline.named_steps["pca"]
    loadings = pca.components_
    explained_variance_ratios = pca.explained_variance_ratio_

    print(f"\n[PCA] Top 5 feature contributions for each of the 5 principal components:")
    for i, component in enumerate(loadings):
        print(f"\n--- Principal Component {i+1} (Explains {explained_variance_ratios[i]:.2%} of variance) ---")
        feature_loadings = pd.Series(component, index=unique_features)
        top_features = feature_loadings.abs().nlargest(5)
        print(feature_loadings[top_features.index].to_string())

    # Plotting (PC1 vs PC2)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label=target_col)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    total_explained_variance = np.sum(pca.explained_variance_ratio_)
    plt.title(
        f"PCA of Top Harmful Q{quantile} Features for {framework_name.capitalize()} (PC1 vs PC2)\n"
        f"Total Explained Variance (5 components): {total_explained_variance:.2%}"
    )
    plt.grid(True)

    # Save plot
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    posix_suffix = "_posix" if posix_only else ""
    output_path = f"{output_dir}/pca_harmful_features_{framework_name}_q{quantile}{posix_suffix}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PCA plot to {output_path}")


def analyze_pc1_impact(
    df: pd.DataFrame,
    top_features: List[str],
    target_col: str,
    framework_name: str,
    quantile: int,
    posix_only: bool,
):
    """Analyzes the first principal component's feature contributions and its relationship to the target."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # The other function already warned

    unique_features = sorted(list(set(top_features)))
    print(f"\n--- PC1 Impact Analysis for Q{quantile} Features ---")
    if len(unique_features) < 2:
        return

    # Prepare data
    pca_df = df[unique_features + [target_col]].dropna(subset=[target_col])
    X = pca_df[unique_features]
    y = pca_df[target_col]

    # Pipeline for preprocessing and PCA
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5)),
        ]
    )
    principal_components = pipeline.fit_transform(X)

    # Analyze PC1
    pc1_loadings = pipeline.named_steps["pca"].components_[0]
    pc1_component_values = principal_components[:, 0]

    # Create a summary dataframe
    impact_df = pd.DataFrame({
        "feature": unique_features,
        "pc1_loading": pc1_loadings,
    })
    impact_df["correlation_with_target"] = [X[f].corr(y) for f in unique_features]
    impact_df = impact_df.reindex(impact_df.pc1_loading.abs().sort_values(ascending=False).index).head(10)

    print("\n[PC1] Top 10 feature contributions to Principal Component 1:")
    print(impact_df.to_string(index=False))

    # Plot PC1 vs. Target
    plt.figure(figsize=(12, 10))
    plt.scatter(pc1_component_values, y, alpha=0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel(target_col)
    plt.title(f"PC1 vs. Target for Top Harmful Q{quantile} Features ({framework_name.capitalize()})")
    plt.grid(True)

    # Save plot
    output_dir = "visualizations"
    posix_suffix = "_posix" if posix_only else ""
    output_path = f"{output_dir}/pc1_vs_target_{framework_name}_q{quantile}{posix_suffix}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PC1 vs. Target plot to {output_path}")


def analyze_bottleneck_frequency_quantiles(
    framework_name: str,
    full_train_df: pd.DataFrame,
    target_col: str = "compute_time_frac_epoch",
    n_splits: int = 5,
    top_k: int = 5,
    posix_only: bool = False,
):
    """
    Performs GroupKFold cross-validation to find the most frequent top-k harmful features and layers
    for 25th and 75th quantiles.
    """
    print(f"\n--- Starting Bottleneck Frequency Analysis for: {framework_name} (Q25/Q75) ---")

    # Drop rows where the target is not finite
    clean_df = full_train_df.dropna(subset=[target_col, 'run_id']).copy()
    if len(clean_df) != len(full_train_df):
        print(f"Dropped {len(full_train_df) - len(clean_df)} rows with missing target or run_id.")

    all_top_features = []
    all_top_features_q25 = []
    all_top_features_q75 = []
    all_top_layers = []
    all_top_feature_groups = []
    all_top_feature_groups_q25 = []
    all_top_feature_groups_q75 = []

    groups = clean_df['run_id']
    gkf = GroupKFold(n_splits=n_splits)

    print(f"Performing {n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(clean_df, groups=groups)):
        print(f"  Processing Fold {fold + 1}/{n_splits}...")

        train_fold_df = clean_df.iloc[train_idx]
        val_fold_df = clean_df.iloc[val_idx]

        feature_cols = select_epoch_features(train_fold_df, target_col, posix_only=posix_only)
        feature_cols = prune_empty_features(train_fold_df, val_fold_df, feature_cols)
        
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_train = imputer.fit_transform(train_fold_df[feature_cols])
        X_val = imputer.transform(val_fold_df[feature_cols])

        # Get true quantiles for training
        y_q25_train, y_q75_train = get_quantiles(train_fold_df, target_col, q_method='mc', q_pair=(25, 75))

        # --- Q25 model ---
        y_q25_is_finite = np.isfinite(y_q25_train)
        X_train_q25 = X_train[y_q25_is_finite]
        y_q25_train_clean = y_q25_train[y_q25_is_finite]
        
        model_q25 = GradientBoostingRegressor(
            loss="quantile", alpha=0.25, n_estimators=200, max_depth=4, random_state=42
        )
        model_q25.fit(X_train_q25, y_q25_train_clean)

        # --- Q75 model ---
        y_q75_is_finite = np.isfinite(y_q75_train)
        X_train_q75 = X_train[y_q75_is_finite]
        y_q75_train_clean = y_q75_train[y_q75_is_finite]
        
        model_q75 = GradientBoostingRegressor(
            loss="quantile", alpha=0.75, n_estimators=200, max_depth=4, random_state=42
        )
        model_q75.fit(X_train_q75, y_q75_train_clean)

        # SHAP for Q25
        explainer_q25 = shap.TreeExplainer(model_q25)
        shap_values_q25 = explainer_q25.shap_values(X_val)
        mean_shap_q25 = np.mean(shap_values_q25, axis=0)
        
        shap_summary_q25 = pd.DataFrame({'feature': feature_cols, 'mean_shap': mean_shap_q25})
        top_harmful_q25 = shap_summary_q25.sort_values('mean_shap', ascending=True).head(top_k)
        
        
        fold_top_features_q25 = top_harmful_q25['feature'].tolist()
        all_top_features.extend(fold_top_features_q25)
        all_top_features_q25.extend(fold_top_features_q25)
        all_top_layers.extend([layer_key(f) for f in fold_top_features_q25])
        
        fold_top_groups_q25 = [get_feature_group(f) for f in fold_top_features_q25]
        all_top_feature_groups.extend(fold_top_groups_q25)
        all_top_feature_groups_q25.extend(fold_top_groups_q25)
        
        print(f"    Top {top_k} harmful features for Q25: {fold_top_features_q25}")

        # SHAP for Q75
        explainer_q75 = shap.TreeExplainer(model_q75)
        shap_values_q75 = explainer_q75.shap_values(X_val)
        mean_shap_q75 = np.mean(shap_values_q75, axis=0)

        shap_summary_q75 = pd.DataFrame({'feature': feature_cols, 'mean_shap': mean_shap_q75})
        top_harmful_q75 = shap_summary_q75.sort_values('mean_shap', ascending=True).head(top_k)

        fold_top_features_q75 = top_harmful_q75['feature'].tolist()
        all_top_features.extend(fold_top_features_q75)
        all_top_features_q75.extend(fold_top_features_q75)
        all_top_layers.extend([layer_key(f) for f in fold_top_features_q75])

        fold_top_groups_q75 = [get_feature_group(f) for f in fold_top_features_q75]
        all_top_feature_groups.extend(fold_top_groups_q75)
        all_top_feature_groups_q75.extend(fold_top_groups_q75)

        print(f"    Top {top_k} harmful features for Q75: {fold_top_features_q75}")


    print("\n--- Aggregated Results ---")
    feature_frequency = Counter(all_top_features)
    layer_frequency = Counter(all_top_layers)
    feature_frequency_q25 = Counter(all_top_features_q25)
    feature_frequency_q75 = Counter(all_top_features_q75)
    feature_frequency_groups = Counter(all_top_feature_groups)
    feature_frequency_groups_q25 = Counter(all_top_feature_groups_q25)
    feature_frequency_groups_q75 = Counter(all_top_feature_groups_q75)

    # Table 1: Layer Frequency
    total_slots = n_splits * top_k * 2
    print(f"\n1. Top Most Frequent Harmful Layers (count out of {total_slots} total top-{top_k} slots):")
    for layer, count in layer_frequency.most_common():
        print(f"  - {layer}: {count}/{total_slots} ({count / total_slots:.1%})")

    # Table 2: All Feature Group Frequency
    print(f"\n2. All Feature Groups: Top Most Frequent Harmful Groups (count out of {total_slots} total top-{top_k} slots):")
    for group, count in feature_frequency_groups.most_common():
        print(f"  - {group}: {count}/{total_slots} ({count / total_slots:.1%})")

    # Table 3: Q25 Feature Group Frequency
    quantile_total_slots = n_splits * top_k
    print(f"\n3. Q25 Feature Groups: Top Most Frequent Harmful Groups (count out of {quantile_total_slots} total top-{top_k} slots):")
    for group, count in feature_frequency_groups_q25.most_common():
        print(f"  - {group}: {count}/{quantile_total_slots} ({count / quantile_total_slots:.1%})")

    # Table 4: Q75 Feature Group Frequency
    print(f"\n4. Q75 Feature Groups: Top Most Frequent Harmful Groups (count out of {quantile_total_slots} total top-{top_k} slots):")
    for group, count in feature_frequency_groups_q75.most_common():
        print(f"  - {group}: {count}/{quantile_total_slots} ({count / quantile_total_slots:.1%})")

    # Table 5: All Feature Frequency
    num_models = n_splits * 2
    print(f"\n5. All Features: Top Most Frequent Harmful Features (appeared in top-{top_k} list of X / {num_models} models):")
    for feature, count in feature_frequency.most_common():
        print(f"  - {feature}: {count}/{num_models}")

    # Table 6: Q25 Feature Frequency
    num_quantile_models = n_splits
    print(f"\n6. Q25 Features: Top Most Frequent Harmful Features (appeared in top-{top_k} list of X / {num_quantile_models} models):")
    for feature, count in feature_frequency_q25.most_common():
        print(f"  - {feature}: {count}/{num_quantile_models}")
    
    # Table 7: Q75 Feature Frequency
    print(f"\n7. Q75 Features: Top Most Frequent Harmful Features (appeared in top-{top_k} list of X / {num_quantile_models} models):")
    for feature, count in feature_frequency_q75.most_common():
        print(f"  - {feature}: {count}/{num_quantile_models}")
        
    # Run PCA analysis on the collected harmful features
    analyze_harmful_features_pca(clean_df, all_top_features_q25, target_col, framework_name, 25, posix_only)
    analyze_harmful_features_pca(clean_df, all_top_features_q75, target_col, framework_name, 75, posix_only)
    analyze_harmful_features_pca(clean_df, all_top_features, target_col, framework_name, 100, posix_only)
    
    # Run PC1 impact analysis
    analyze_pc1_impact(clean_df, all_top_features_q25, target_col, framework_name, 25, posix_only)
    analyze_pc1_impact(clean_df, all_top_features_q75, target_col, framework_name, 75, posix_only)
    analyze_pc1_impact(clean_df, all_top_features, target_col, framework_name, 100, posix_only)

    return feature_frequency, layer_frequency, feature_frequency_q25, feature_frequency_q75


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bottleneck attribution frequency for quantile models.")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], required=True, help="Framework to analyze.")
    parser.add_argument("--in_path", type=str, help="Path to the training dataset parquet file.")
    parser.add_argument("--target_col", type=str, default="compute_time_frac_epoch", help="Target column for analysis.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for GroupKFold.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top harmful features to identify per fold.")
    parser.add_argument("--posix_only", action="store_true", help="Only analyze POSIX features.")
    args = parser.parse_args()

    # Build path if not provided
    if args.in_path:
        input_path = args.in_path
    else:
        from dfdiagnoser_ml.common import GLOBALS_DIR
        input_path = f"{GLOBALS_DIR}/epoch/datasets/ml_data_{args.framework}_train_full.parquet"
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    analyze_bottleneck_frequency_quantiles(
        framework_name=args.framework,
        full_train_df=df,
        target_col=args.target_col,
        n_splits=args.n_splits,
        top_k=args.top_k,
        posix_only=args.posix_only,
    )
