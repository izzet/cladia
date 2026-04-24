import joblib
import os
import pandas as pd
from typing import List, Optional

from dfdiagnoser_ml.common import GLOBALS_DIR, drop_nonfinite_target, add_special_features
from dfdiagnoser_ml.evaluate_mean import evaluate_on_holdout, shap_layer_report, run_leak_checks, shap_holdout_pair_report
from dfdiagnoser_ml.evaluate_quantiles import (
    evaluate_quantiles,
    plot_view_layer_timeseries,
    plot_view_pair_layer_deltas,
    plot_view_shap_heatmaps,
    shap_holdout_pair_report_quantiles,
    shap_view_trajectory_report_quantiles,
)
from dfdiagnoser_ml.plot_quantiles import plot_shap_summary_for_holdout_pairs
from dfdiagnoser_ml.save_quantiles import save_shap_summary_to_csv
from dfdiagnoser_ml.training_mean import run_mean_training
from dfdiagnoser_ml.training_quantiles import run_quantile_training_and_calibration

RESULTS_DIR = os.environ.get("DFD_REPRO_RESULTS_DIR", os.environ.get("DFD_RESULTS_DIR", "results"))


def main(
    framework: str = "pytorch", 
    view_type: str = "epoch", 
    target_col: str = "epoch_time_max", 
    feature_groups: Optional[List[int]] = None,
    posix_only: bool = False, 
    q_method: str = "delta",
    q_low: int = 25,
    q_high: int = 75,
):
    dataset_dir = f"{GLOBALS_DIR}/{view_type}/datasets"
    train_p = f"{dataset_dir}/ml_data_{framework}_train_full.parquet"
    test_p = f"{dataset_dir}/ml_data_{framework}_test_full.parquet"
    hold_p = f"{dataset_dir}/ml_data_{framework}_holdout_full.parquet"
    train_df = pd.read_parquet(train_p)
    test_df = pd.read_parquet(test_p)
    holdout_df = pd.read_parquet(hold_p)

    train_df = drop_nonfinite_target(train_df, target_col)
    test_df = drop_nonfinite_target(test_df, target_col)
    holdout_df = drop_nonfinite_target(holdout_df, target_col)

    if feature_groups is not None and 3 in feature_groups:
        train_df = add_special_features(train_df)
        test_df = add_special_features(test_df)
        holdout_df = add_special_features(holdout_df)

    print("Running mean model training...")
    mean_pipe, mean_feats, mean_metrics = run_mean_training(train_df, test_df, target_col, posix_only=posix_only, feature_groups=feature_groups)
    # Save mean pipeline and features
    models_dir = f"{GLOBALS_DIR}/{view_type}/models"
    os.makedirs(models_dir, exist_ok=True)

    path_infix = f"{framework}_{view_type}_{target_col}"
    if feature_groups:
        fg_suffix = "fg" + "".join(str(int(x)) for x in feature_groups)
        path_infix = f"{path_infix}_{fg_suffix}"
    posix_suffix = ""
    if posix_only:
        posix_suffix = "_posix"
    q_suffix = f"{q_method}_{q_low}_{q_high}"

    mean_pipe_path = f"{models_dir}/mean_pipeline_{path_infix}{posix_suffix}.joblib"
    mean_feats_path = f"{models_dir}/mean_features_{path_infix}{posix_suffix}.joblib"
    joblib.dump(mean_pipe, mean_pipe_path)
    joblib.dump(mean_feats, mean_feats_path)
    print(f"Saved mean pipeline to: {mean_pipe_path}")
    print(f"Saved mean features to: {mean_feats_path}")
    
    # Reports for mean model
    _ = run_leak_checks(train_df, test_df, mean_feats, target_col, groups_col="run_id")
    mean_hold = evaluate_on_holdout(mean_pipe, mean_feats, holdout_df, target_col)
    mean_shap = shap_layer_report(mean_pipe, test_df, mean_feats)
    k_list = [1, 3, 5, 10]
    mean_shap_holdout_results = shap_holdout_pair_report(mean_pipe, holdout_df, mean_feats, target_col, k_list=k_list)

    print("\nRunning quantile training and AMS-baseline calibration study...")
    print(f"Using {target_col} as target column for quantiles; q_method={q_method}, q_pair=({q_low},{q_high})")
    q_low_pipe, q_high_pipe, q_feats, q_metrics = run_quantile_training_and_calibration(
        train_df,
        test_df,
        holdout_df,
        target_col,
        feature_groups=feature_groups,
        posix_only=posix_only,
        q_method=q_method,
        q_low=q_low,
        q_high=q_high,
    )
    # Save quantile pipelines separately (low/high) and features
    q_low_path = f"{models_dir}/quantile_pipeline_low_{path_infix}_{q_suffix}{posix_suffix}.joblib"
    q_high_path = f"{models_dir}/quantile_pipeline_high_{path_infix}_{q_suffix}{posix_suffix}.joblib"
    qfeats_path = f"{models_dir}/quantile_features_{path_infix}_{q_suffix}{posix_suffix}.joblib"
    joblib.dump(q_low_pipe, q_low_path)
    joblib.dump(q_high_pipe, q_high_path)
    joblib.dump(q_feats, qfeats_path)
    print(f"Saved q_low pipeline to: {q_low_path}")
    print(f"Saved q_high pipeline to: {q_high_path}")
    print(f"Saved quantile features to: {qfeats_path}")
    
    # Quantile evaluation reports (holdout only)
    q_hold = evaluate_quantiles(q_low_pipe, q_high_pipe, holdout_df, q_feats, target_col, framework, label="Holdout", q_method=q_method, q_pair=(q_low, q_high))
    
    shap_results = shap_holdout_pair_report_quantiles(
        q_low_pipe,
        q_high_pipe,
        holdout_df,
        q_feats,
        target_prefix=target_col,
        q_method=q_method,
        q_pair=(q_low, q_high),
        return_data=True,
        k_list=k_list,
    )
    if shap_results:
        plot_shap_summary_for_holdout_pairs(shap_results, out_dir=os.path.join(RESULTS_DIR, "plots"))
        
        # Save features-level summary
        output_path_features = os.path.join(
            RESULTS_DIR, f"holdout_shap_{path_infix}_{q_suffix}{posix_suffix}_features.csv"
        )
        save_shap_summary_to_csv(shap_results, output_path=output_path_features)

        # Save layers-level summary
        output_path_layers = os.path.join(
            RESULTS_DIR, f"holdout_shap_{path_infix}_{q_suffix}{posix_suffix}_layers.csv"
        )
        try:
            from dfdiagnoser_ml.save_quantiles import (
                save_shap_layers_to_csv, save_shap_feature_groups_to_csv,
                save_shap_first_sample_to_csv, save_shap_io_bound_summary_to_csv
            )
            save_shap_layers_to_csv(shap_results, output_path=output_path_layers)
            
            output_path_fgs = os.path.join(
                RESULTS_DIR, f"holdout_shap_{path_infix}_{q_suffix}{posix_suffix}_feature_groups.csv"
            )
            save_shap_feature_groups_to_csv(shap_results, output_path=output_path_fgs)

            output_path_sample = os.path.join(
                RESULTS_DIR, f"holdout_shap_{path_infix}_{q_suffix}{posix_suffix}_first_sample.csv"
            )
            save_shap_first_sample_to_csv(shap_results, output_path=output_path_sample)

            output_path_io = os.path.join(
                RESULTS_DIR, f"holdout_shap_{path_infix}_{q_suffix}{posix_suffix}_io_bound.csv"
            )
            save_shap_io_bound_summary_to_csv(shap_results, output_path=output_path_io)
        except Exception as exc:
            print(f"Failed to save SHAP aggregate CSVs: {exc}")

    # Append single-row summary CSV for later comparison
    try:
        import pandas as _pd
        summary = {
            "framework": framework,
            "view_type": view_type,
            "target_col": target_col,
            "feature_groups": ",".join(str(int(x)) for x in (feature_groups or [])),
            "posix_only": posix_only,
            "q_method": q_method,
            "q_low": int(q_low),
            "q_high": int(q_high),
            # Mean metrics
            "mean_train_mae": mean_metrics.get("mae_train", float("nan")),
            "mean_train_rmse": mean_metrics.get("rmse_train", float("nan")),
            "mean_train_r2": mean_metrics.get("r2_train", float("nan")),
            "mean_train_mdea": mean_metrics.get("mdea_train", float("nan")),
            "mean_train_mape": mean_metrics.get("mape_train", float("nan")),
            "mean_train_mdape": mean_metrics.get("mdape_train", float("nan")),
            "mean_test_mae": mean_metrics.get("mae_test", float("nan")),
            "mean_test_rmse": mean_metrics.get("rmse_test", float("nan")),
            "mean_test_r2": mean_metrics.get("r2_test", float("nan")),
            "mean_test_mdea": mean_metrics.get("mdea_test", float("nan")),
            "mean_test_mape": mean_metrics.get("mape_test", float("nan")),
            "mean_test_mdape": mean_metrics.get("mdape_test", float("nan")),
            "mean_holdout_mae": mean_hold.get("mae", float("nan")),
            "mean_holdout_rmse": mean_hold.get("rmse", float("nan")),
            "mean_holdout_r2": mean_hold.get("r2", float("nan")),
            # Quantile metrics (train)
            "q_train_ams_alpha2": q_metrics.get("q_ams_alpha2_train", float("nan")),
            "q_train_iqs": q_metrics.get("q_iqs_train", float("nan")),
            "q_train_overlap": q_metrics.get("q_overlap_train", float("nan")),
            "q_train_overlap_tolerant": q_metrics.get("q_overlap_tolerant_train", float("nan")),
            "q_train_picp": q_metrics.get("q_picp_train", float("nan")),
            "q_train_winkler": q_metrics.get("q_winkler_train", float("nan")),
            "q_train_center_r2": q_metrics.get("q_r2_center_train", float("nan")),
            "q_train_width_r2": q_metrics.get("q_r2_width_train", float("nan")),
            "q_train_low_mae": q_metrics.get("q_low_mae_train", float("nan")),
            "q_train_low_mdape": q_metrics.get("q_low_mdape_train", float("nan")),
            "q_train_low_mdea": q_metrics.get("q_low_mdea_train", float("nan")),
            "q_train_low_rmse": q_metrics.get("q_low_rmse_train", float("nan")),
            "q_train_low_r2": q_metrics.get("q_r2_low_train", float("nan")),
            "q_train_high_mae": q_metrics.get("q_high_mae_train", float("nan")),
            "q_train_high_mdape": q_metrics.get("q_high_mdape_train", float("nan")),
            "q_train_high_mdea": q_metrics.get("q_high_mdea_train", float("nan")),
            "q_train_high_rmse": q_metrics.get("q_high_rmse_train", float("nan")),
            "q_train_high_r2": q_metrics.get("q_r2_high_train", float("nan")),
            # Quantile metrics (test)
            "q_test_ams_alpha2": q_metrics.get("q_ams_alpha2_test", float("nan")),
            "q_test_iqs": q_metrics.get("q_iqs_test", float("nan")),
            "q_test_overlap": q_metrics.get("q_overlap_test", float("nan")),
            "q_test_overlap_tolerant": q_metrics.get("q_overlap_tolerant_test", float("nan")),
            "q_test_picp": q_metrics.get("q_picp_test", float("nan")),
            "q_test_winkler": q_metrics.get("q_winkler_test", float("nan")),
            "q_test_center_r2": q_metrics.get("q_r2_center_test", float("nan")),
            "q_test_width_r2": q_metrics.get("q_r2_width_test", float("nan")),
            "q_test_low_mae": q_metrics.get("q_low_mae_test", float("nan")),
            "q_test_low_mdape": q_metrics.get("q_low_mdape_test", float("nan")),
            "q_test_low_mdea": q_metrics.get("q_low_mdea_test", float("nan")),
            "q_test_low_rmse": q_metrics.get("q_low_rmse_test", float("nan")),
            "q_test_low_r2": q_metrics.get("q_r2_low_test", float("nan")),
            "q_test_high_mae": q_metrics.get("q_high_mae_test", float("nan")),
            "q_test_high_mdape": q_metrics.get("q_high_mdape_test", float("nan")),
            "q_test_high_mdea": q_metrics.get("q_high_mdea_test", float("nan")),
            "q_test_high_rmse": q_metrics.get("q_high_rmse_test", float("nan")),
            "q_test_high_r2": q_metrics.get("q_r2_high_test", float("nan")),
            # Quantile metrics (holdout)
            "q_holdout_ams_alpha2": q_hold.get("ams_alpha2", float("nan")),
            "q_holdout_iqs": q_hold.get("iqs", float("nan")),
            "q_holdout_overlap": q_hold.get("overlap", float("nan")),
            "q_holdout_overlap_tolerant": q_hold.get("overlap_tolerant", float("nan")),
            "q_holdout_picp": q_hold.get("picp", float("nan")),
            "q_holdout_winkler": q_hold.get("winkler", float("nan")),
            "q_holdout_center_r2": q_hold.get("r2_center", float("nan")),
            "q_holdout_width_r2": q_hold.get("r2_width", float("nan")),
            "q_holdout_low_mae": q_hold.get("low_mae", float("nan")),
            "q_holdout_low_mdape": q_hold.get("low_mdape", float("nan")),
            "q_holdout_low_mdea": q_hold.get("low_mdea", float("nan")),
            "q_holdout_low_rmse": q_hold.get("low_rmse", float("nan")),
            "q_holdout_low_r2": q_hold.get("r2_low", float("nan")),
            "q_holdout_high_mae": q_hold.get("high_mae", float("nan")),
            "q_holdout_high_mdape": q_hold.get("high_mdape", float("nan")),
            "q_holdout_high_mdea": q_hold.get("high_mdea", float("nan")),
            "q_holdout_high_rmse": q_hold.get("high_rmse", float("nan")),
            "q_holdout_high_r2": q_hold.get("r2_high", float("nan")),
        }
        # Top-S attributions
        top_s = 3
        
        # Define what to process
        attribute_sets = [
            (mean_shap, "top_features", "mean_feature"),
            (mean_shap, "top_layers", "mean_layer"),
            (q_hold, "q_low_top_features", "q_low_feature"),
            (q_hold, "q_low_top_layers", "q_low_layer"),
            (q_hold, "q_high_top_features", "q_high_feature"),
            (q_hold, "q_high_top_layers", "q_high_layer"),
        ]

        # Initialize and populate
        for data_dict, data_key, summary_prefix in attribute_sets:
            # Initialize
            for i in range(1, top_s + 1):
                summary[f"{summary_prefix}_{i}"] = ""
                summary[f"{summary_prefix}_{i}_value"] = float("nan")
            
            # Populate
            items = data_dict.get(data_key, []) or []
            for i, pair in enumerate(items[:top_s], start=1):
                summary[f"{summary_prefix}_{i}"] = pair[0]
                summary[f"{summary_prefix}_{i}_value"] = float(pair[1])

        # --- Bottleneck Hit Rate for Mean Model ---
        if mean_shap_holdout_results and mean_shap_holdout_results.get("bottleneck_hit_results"):
            mean_totals = {k: {"feature_hits": 0, "layer_hits": 0, "nrows": 0} for k in k_list}
            for _pair_id, data in mean_shap_holdout_results["bottleneck_hit_results"].items():
                if not data:
                    continue
                by_k = data.get("by_k", {})
                for k in k_list:
                    rec = by_k.get(k)
                    if not rec:
                        continue
                    mean_totals[k]["feature_hits"] += rec.get("feature_hit_count", 0)
                    mean_totals[k]["layer_hits"] += rec.get("layer_hit_count", 0)
                    mean_totals[k]["nrows"] += rec.get("nrows", 0)

            for k in k_list:
                summary[f"mean_bot_hit_feature_nrows_top{k}"] = mean_totals[k]["feature_hits"]
                summary[f"mean_bot_hit_layer_nrows_top{k}"] = mean_totals[k]["layer_hits"]
                summary[f"mean_bot_hit_evaluated_rows_top{k}"] = mean_totals[k]["nrows"]

        # --- Bottleneck Hit Rate for Quantile Model ---
        if shap_results:
            cw_totals = {k: {"feature_hits": 0, "layer_hits": 0, "nrows": 0} for k in k_list}
            qlh_totals = {k: {"feature_hits": 0, "layer_hits": 0, "nrows": 0} for k in k_list}

            for _pair_id, data in shap_results.items():
                if not isinstance(data, dict):
                    continue
                cw = data.get("bot_hit_center_width")
                qlh = data.get("bot_hit_q_low_high")
                if cw:
                    by_k = cw.get("by_k", {})
                    for k in k_list:
                        rec = by_k.get(k)
                        if not rec:
                            continue
                        cw_totals[k]["feature_hits"] += rec.get("feature_hit_count", 0)
                        cw_totals[k]["layer_hits"] += rec.get("layer_hit_count", 0)
                        cw_totals[k]["nrows"] += rec.get("nrows", 0)
                if qlh:
                    by_k = qlh.get("by_k", {})
                    for k in k_list:
                        rec = by_k.get(k)
                        if not rec:
                            continue
                        qlh_totals[k]["feature_hits"] += rec.get("feature_hit_count", 0)
                        qlh_totals[k]["layer_hits"] += rec.get("layer_hit_count", 0)
                        qlh_totals[k]["nrows"] += rec.get("nrows", 0)

            for k in k_list:
                summary[f"q_cw_bot_hit_feature_nrows_top{k}"] = cw_totals[k]["feature_hits"]
                summary[f"q_cw_bot_hit_layer_nrows_top{k}"] = cw_totals[k]["layer_hits"]
                summary[f"q_cw_bot_hit_evaluated_rows_top{k}"] = cw_totals[k]["nrows"]

                summary[f"q_qlh_bot_hit_feature_nrows_top{k}"] = qlh_totals[k]["feature_hits"]
                summary[f"q_qlh_bot_hit_layer_nrows_top{k}"] = qlh_totals[k]["layer_hits"]
                summary[f"q_qlh_bot_hit_evaluated_rows_top{k}"] = qlh_totals[k]["nrows"]

            # --- Final Hit Rate Printout (top-3 for continuity) ---
            if 3 in k_list:
                mean_total = summary.get("mean_bot_hit_evaluated_rows_top3", 0)
                mean_feat_hits = summary.get("mean_bot_hit_feature_nrows_top3", 0)
                mean_layer_hits = summary.get("mean_bot_hit_layer_nrows_top3", 0)
                mean_feat_rate = mean_feat_hits / mean_total if mean_total > 0 else 0.0
                mean_layer_rate = mean_layer_hits / mean_total if mean_total > 0 else 0.0

                q_total_cw = summary.get("q_cw_bot_hit_evaluated_rows_top3", 0)
                q_feat_hits_cw = summary.get("q_cw_bot_hit_feature_nrows_top3", 0)
                q_layer_hits_cw = summary.get("q_cw_bot_hit_layer_nrows_top3", 0)
                q_feat_rate_cw = q_feat_hits_cw / q_total_cw if q_total_cw > 0 else 0.0
                q_layer_rate_cw = q_layer_hits_cw / q_total_cw if q_total_cw > 0 else 0.0

                q_total_qlh = summary.get("q_qlh_bot_hit_evaluated_rows_top3", 0)
                q_feat_hits_qlh = summary.get("q_qlh_bot_hit_feature_nrows_top3", 0)
                q_layer_hits_qlh = summary.get("q_qlh_bot_hit_layer_nrows_top3", 0)
                q_feat_rate_qlh = q_feat_hits_qlh / q_total_qlh if q_total_qlh > 0 else 0.0
                q_layer_rate_qlh = q_layer_hits_qlh / q_total_qlh if q_total_qlh > 0 else 0.0

                summary["mean_bot_hit_feature_rate_top3"] = mean_feat_rate
                summary["mean_bot_hit_layer_rate_top3"] = mean_layer_rate
                summary["q_cw_bot_hit_feature_rate_top3"] = q_feat_rate_cw
                summary["q_cw_bot_hit_layer_rate_top3"] = q_layer_rate_cw
                summary["q_qlh_bot_hit_feature_rate_top3"] = q_feat_rate_qlh
                summary["q_qlh_bot_hit_layer_rate_top3"] = q_layer_rate_qlh

                print("\n--- Diagnostic Accuracy Summary (Per-Epoch, top-3) ---")
                print(f"Mean Model Feature Hit Rate: {mean_feat_rate:.3f} ({mean_feat_hits}/{mean_total})")
                print(f"Mean Model Layer Hit Rate:   {mean_layer_rate:.3f} ({mean_layer_hits}/{mean_total})")
                print(f"Quantile CW Feature Hit Rate: {q_feat_rate_cw:.3f} ({q_feat_hits_cw}/{q_total_cw})")
                print(f"Quantile CW Layer Hit Rate:   {q_layer_rate_cw:.3f} ({q_layer_hits_cw}/{q_total_cw})")
                print(f"Quantile Q-low/high Feature Hit Rate: {q_feat_rate_qlh:.3f} ({q_feat_hits_qlh}/{q_total_qlh})")
                print(f"Quantile Q-low/high Layer Hit Rate:   {q_layer_rate_qlh:.3f} ({q_layer_hits_qlh}/{q_total_qlh})")
                print("---------------------------------------------")

            # Add calculated rates to summary dictionary for all k
            for k in k_list:
                mean_total = summary.get(f"mean_bot_hit_evaluated_rows_top{k}", 0)
                mean_feat_hits = summary.get(f"mean_bot_hit_feature_nrows_top{k}", 0)
                mean_layer_hits = summary.get(f"mean_bot_hit_layer_nrows_top{k}", 0)
                summary[f"mean_bot_hit_feature_rate_top{k}"] = mean_feat_hits / mean_total if mean_total > 0 else 0.0
                summary[f"mean_bot_hit_layer_rate_top{k}"] = mean_layer_hits / mean_total if mean_total > 0 else 0.0

                q_total_cw = summary.get(f"q_cw_bot_hit_evaluated_rows_top{k}", 0)
                q_feat_hits_cw = summary.get(f"q_cw_bot_hit_feature_nrows_top{k}", 0)
                q_layer_hits_cw = summary.get(f"q_cw_bot_hit_layer_nrows_top{k}", 0)
                summary[f"q_cw_bot_hit_feature_rate_top{k}"] = q_feat_hits_cw / q_total_cw if q_total_cw > 0 else 0.0
                summary[f"q_cw_bot_hit_layer_rate_top{k}"] = q_layer_hits_cw / q_total_cw if q_total_cw > 0 else 0.0

                q_total_qlh = summary.get(f"q_qlh_bot_hit_evaluated_rows_top{k}", 0)
                q_feat_hits_qlh = summary.get(f"q_qlh_bot_hit_feature_nrows_top{k}", 0)
                q_layer_hits_qlh = summary.get(f"q_qlh_bot_hit_layer_nrows_top{k}", 0)
                summary[f"q_qlh_bot_hit_feature_rate_top{k}"] = q_feat_hits_qlh / q_total_qlh if q_total_qlh > 0 else 0.0
                summary[f"q_qlh_bot_hit_layer_rate_top{k}"] = q_layer_hits_qlh / q_total_qlh if q_total_qlh > 0 else 0.0

        # Write/append
        out_csv = os.path.join(RESULTS_DIR, "training_runs_summary.csv")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df_row = _pd.DataFrame([summary])
        if not os.path.exists(out_csv):
            df_row.to_csv(out_csv, index=False)
        else:
            try:
                df_old = _pd.read_csv(out_csv)
                # Add any missing columns to both frames
                for col in df_row.columns:
                    if col not in df_old.columns:
                        df_old[col] = _pd.NA
                for col in df_old.columns:
                    if col not in df_row.columns:
                        df_row[col] = _pd.NA
                # Align column order to existing file
                df_row = df_row[df_old.columns]
                df_all = _pd.concat([df_old, df_row], ignore_index=True)
                df_all.to_csv(out_csv, index=False)
            except Exception:
                df_row.to_csv(out_csv, mode='a', header=False, index=False)
        print(f"Appended summary to {out_csv}")
    except Exception as _exc:
        print(f"Failed to append summary CSV: {_exc}")

    exit()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--framework", type=str, default="pytorch")
    parser.add_argument("--view_type", type=str, default="epoch")
    parser.add_argument("--target_col", type=str, default="epoch_time_max")
    parser.add_argument('--feature_groups', nargs='+', type=int)
    parser.add_argument("--posix_only", action="store_true")
    parser.add_argument("--q_method", type=str, choices=["mc","delta"], default="delta")
    parser.add_argument("--q_low", type=int, default=25)
    parser.add_argument("--q_high", type=int, default=75)
    args = parser.parse_args()
    print("Running training with the following parameters:")
    print(f"   framework: {args.framework}")
    print(f"   view_type: {args.view_type}")
    print(f"   target_col: {args.target_col}")
    print(f"   feature_groups: {args.feature_groups}")
    print(f"   posix_only: {args.posix_only}")
    print(f"   q_method: {args.q_method}")
    print(f"   q_low: {args.q_low}")
    print(f"   q_high: {args.q_high}")
    main(
        framework=args.framework, 
        view_type=args.view_type, 
        target_col=args.target_col, 
        feature_groups=args.feature_groups,
        posix_only=args.posix_only, 
        q_method=args.q_method,
        q_low=args.q_low,
        q_high=args.q_high,
    )
