import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np


def plot_shap_summary_for_holdout_pairs(shap_results: dict, out_dir: str = "results"):
    """
    Generates and saves SHAP summary plots for each holdout pair.

    Args:
        shap_results: The dictionary returned by shap_holdout_pair_report_quantiles.
        out_dir: Directory to save the plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    for pair_id, data in shap_results.items():
        if "shap_values" not in data:
            continue

        sv = data["shap_values"]
        feature_names = sv["features"]
        df_subsets = data["data"]

        # Define the 8 plots to generate
        plot_configs = {
            "bad_center": (sv["bad_center"], df_subsets["bad"][feature_names]),
            "good_center": (sv["good_center"], df_subsets["good"][feature_names]),
            "bad_width": (sv["bad_width"], df_subsets["bad"][feature_names]),
            "good_width": (sv["good_width"], df_subsets["good"][feature_names]),
        }

        for name, (shap_matrix, feature_matrix) in plot_configs.items():
            if shap_matrix.shape[0] == 0:  # Skip if no data
                continue

            plt.figure()
            shap.summary_plot(
                shap_matrix,
                feature_matrix,
                show=False,
                plot_type="bar",  # Use 'dot' for more detail, 'bar' for importance
            )
            plt.title(f"SHAP Summary: {pair_id} - {name}")
            plt.tight_layout()

            # Sanitize filename
            safe_pair_id = pair_id.replace("/", "_")
            fig_path = os.path.join(out_dir, f"shap_{safe_pair_id}_{name}.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"Saved SHAP plot: {fig_path}")


def plot_prediction_intervals(
    evaluated_parquet_path: str,
    output_plot_path: str,
    target_col: str = "epoch_time_max",
    mean_pred_col: str = "y_pred_mean",
    q_low_pred_col: str = "q25_pred",
    q_high_pred_col: str = "q75_pred",
    outlier_keep_fraction: float = 1.0,
    output_csv_path: str | None = None,
    scale_y_to_percent: bool = False,
    y_axis_label: str = "Target Value",
):
    """
    Generates and saves a plot showing model prediction intervals against true values.

    Args:
        evaluated_parquet_path: Path to the input parquet file with evaluated predictions.
        output_plot_path: Path to save the output plot (format inferred from extension, e.g., .png, .pdf).
        target_col: Name of the column with the true target values.
        mean_pred_col: Name of the column with the mean predictions.
        q_low_pred_col: Name of the column with the lower quantile predictions.
        q_high_pred_col: Name of the column with the upper quantile predictions.
        outlier_keep_fraction: Fraction of out-of-range samples to keep. 1.0 keeps all.
        output_csv_path: Optional path to save the plotted data to a CSV file.
        scale_y_to_percent: If True, multiplies y-axis values by 100.
        y_axis_label: The label for the y-axis.
    """
    if not os.path.exists(evaluated_parquet_path):
        print(f"Error: Input file not found at {evaluated_parquet_path}")
        return

    df = pd.read_parquet(evaluated_parquet_path)

    # Verify necessary columns exist
    required_cols = [target_col, mean_pred_col, q_low_pred_col, q_high_pred_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in the dataframe: {required_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Scale to percentage if requested
    if scale_y_to_percent:
        for col in required_cols:
            df[col] = df[col] * 100

    # Filter to keep a fraction of outliers if requested
    if outlier_keep_fraction < 1.0:
        in_range_mask = (df[target_col] >= df[q_low_pred_col]) & (df[target_col] <= df[q_high_pred_col])
        in_range_samples = df[in_range_mask]
        out_of_range_samples = df[~in_range_mask]

        # Sample the outliers
        n_outliers_to_keep = int(len(out_of_range_samples) * outlier_keep_fraction)
        sampled_outliers = out_of_range_samples.sample(n=n_outliers_to_keep, random_state=42)

        # Combine and proceed
        df = pd.concat([in_range_samples, sampled_outliers])
        print(f"Filtered data: Kept {len(in_range_samples)} in-range samples and {len(sampled_outliers)} out-of-range samples.")

    # Sort the dataframe by the mean prediction to get a smooth line
    df_sorted = df.sort_values(by=mean_pred_col).reset_index(drop=True)

    # Save the data to CSV if a path is provided
    if output_csv_path:
        csv_cols = [target_col, mean_pred_col, q_low_pred_col, q_high_pred_col]
        df_to_save = df_sorted[csv_cols]
        csv_dir = os.path.dirname(output_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df_to_save.to_csv(output_csv_path, index=False)
        print(f"Plot data saved to: {output_csv_path}")

    # Create an index for the x-axis
    x_axis = np.arange(len(df_sorted))

    plt.style.use('grayscale')
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot the true values as a scatter plot
    ax.scatter(x_axis, df_sorted[target_col], s=30, alpha=0.7, label="True Values", color="dimgray")

    # Plot the mean prediction as a solid line
    ax.plot(x_axis, df_sorted[mean_pred_col], color='black', linewidth=2, label="Mean Prediction")

    # Plot the prediction interval as a shaded area
    ax.fill_between(
        x_axis,
        df_sorted[q_low_pred_col],
        df_sorted[q_high_pred_col],
        alpha=0.2,
        label="Predicted 25-75 Quantile Range",
        color="black"
    )

    ax.set_xlabel("Samples (Sorted by Mean Prediction)")
    ax.set_ylabel(y_axis_label)
    ax.set_title("Model Prediction Intervals vs. True Values")
    ax.legend()
    ax.grid(False)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_plot_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_plot_path}")
