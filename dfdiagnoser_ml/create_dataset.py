import argparse
import os
import pandas as pd
import shutil
import warnings

warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

GLOBALS_DIR = os.environ.get("DFD_REPRO_GLOBALS_DIR", os.environ.get("DFD_GLOBALS_DIR", "globals"))

def create_train_test_df(framework, view_type, threshold, clean):
    input_dir = os.path.join(GLOBALS_DIR, view_type)
    output_dir = os.path.join(GLOBALS_DIR, view_type, "datasets")

    if clean:
        if os.path.exists(output_dir):
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)

    run_df = pd.read_parquet(f"{GLOBALS_DIR}/ml_workload_all.parquet")
    df = pd.read_parquet(f"{input_dir}/ml_data_{framework}_all.parquet")
    print("Initial dataframe shape:", df.shape)

    # --- Initial Data Cleaning and Merging ---
    run_df["workload_name"] = run_df["workload_name"].str.lower()
    df["workload_name"] = df["workload_name"].str.lower()
    run_df.index = run_df.index.str.lower()
    df["run_id"] = df["run_id"].str.lower()
    print("Unique run ids:", df["run_id"].nunique())
    print("Unique workload names:", df["workload_name"].nunique())

    run_config_cols = [col for col in run_df.columns if col.startswith("config_")]
    config_cols = [col for col in df.columns if col.startswith("config_")]
    df = df.drop(columns=config_cols).merge(run_df[run_config_cols], left_on="run_id", right_index=True)
    print("Cleaned and merged dataframe shape:", df.shape)

    # --- Add Config ID ---
    df["config_id"] = pd.util.hash_pandas_object(df[run_config_cols], index=False)
    df["workload_config_id"] = df["workload_name"] + "_" + df["config_id"].astype(str)
    print("Unique config ids:", df["config_id"].nunique())
    print("Unique workload config ids:", df["workload_config_id"].nunique())

    # --- Impute Missing Values ---
    cosmoflow_v100_mask = df["workload_name"] == "cosmoflow_v100"
    resnet50_v100_mask = df["workload_name"] == "resnet50_v100"
    unet3d_v100_npz_mask = (df["workload_name"] == "unet3d_v100") & (df["config_dataset_format_npz"])
    unet3d_v100_hdf5_mask = (df["workload_name"] == "unet3d_v100") & (~df["config_dataset_format_npz"])
    df.loc[cosmoflow_v100_mask, "config_reader_transfer_size"] = df.loc[cosmoflow_v100_mask, "config_reader_transfer_size"].fillna(262144)
    df.loc[resnet50_v100_mask, "config_reader_transfer_size"] = df.loc[resnet50_v100_mask, "config_reader_transfer_size"].fillna(262144)
    df.loc[unet3d_v100_npz_mask, "config_reader_transfer_size"] = df.loc[unet3d_v100_npz_mask, "config_reader_transfer_size"].fillna(4194304)
    df.loc[unet3d_v100_hdf5_mask, "config_reader_transfer_size"] = df.loc[unet3d_v100_hdf5_mask, "config_reader_transfer_size"].fillna(146579449)

    # --- Filter Unwanted ---
    df = df.query("~run_id.str.contains('deepspeed') | config_train_computation_time == 2.44")

    # --- I/O-Bound vs. Compute-Bound Split ---
    io_bound_df = df[df['compute_time_frac_epoch'] < threshold].copy()
    print(f"Total I/O-bound data points: {len(io_bound_df)}")

    holdout_dict = {
        "tensorflow": {
            "cosmoflow_v100_all_bad": [
                "workload_name == 'cosmoflow_v100'",
                "config_num_nodes == 4",
                "config_num_processes == 4",
                "config_reader_read_threads == 1",
            ],
            "cosmoflow_v100_all_good": [
                "workload_name == 'cosmoflow_v100'",
                "config_num_nodes == 4",
                "config_num_processes == 4",
                "config_reader_read_threads == 4",
            ],
            "cosmoflow_v100_posix_bad": [
                "workload_name == 'cosmoflow_v100'",
                "config_num_nodes == 4",
                "config_num_processes == 4",
                "config_reader_read_threads == 1",
                "config_dataset_num_files_train == 524288",
            ],
            "cosmoflow_v100_posix_good": [
                "workload_name == 'cosmoflow_v100'",
                "config_num_nodes == 4",
                "config_num_processes == 4",
                "config_reader_read_threads == 1",
                "config_dataset_num_files_train == 5242",
            ],
            "resnet50_v100_all_bad": [
                "workload_name == 'resnet50_v100'",
                "config_num_nodes == 32",
                "config_reader_read_threads == 1",
                "config_dataset_num_files_train == 64",
            ],
            "resnet50_v100_all_good": [
                "workload_name == 'resnet50_v100'",
                "config_num_nodes == 32",
                "config_reader_read_threads == 1",
                "config_dataset_num_files_train == 256",
            ],
        },
        "pytorch": {       
            "unet3d_v100_all_bad": [
                "workload_name == 'unet3d_v100'",
                'config_num_nodes == 1',
                'config_reader_read_threads == 1',
                'config_reader_batch_size == 4',
                'config_dataset_format_npz == True',
                'config_train_epochs == 10',
                "config_dataset_num_files_train == 168",
                "config_reader_prefetch_workers == False"
            ],
            "unet3d_v100_all_good": [
                "workload_name == 'unet3d_v100'",
                'config_num_nodes == 1',
                'config_reader_read_threads == 1',
                'config_reader_batch_size == 4',
                'config_dataset_format_npz == True',
                'config_train_epochs == 10',
                "config_dataset_num_files_train == 168",
                "config_reader_prefetch_workers == True"
            ],
            "unet3d_v100_posix_bad": [
                "workload_name == 'unet3d_v100'",
                'config_num_nodes == 1',
                'config_reader_read_threads == 1',
                'config_reader_batch_size == 4',
                'config_dataset_format_npz == True',
                'config_train_epochs == 10',
                "config_dataset_num_files_train == 168",
                "config_reader_prefetch_workers == False"
            ],
            "unet3d_v100_posix_good": [
                "workload_name == 'unet3d_v100'",
                'config_num_nodes == 1',
                'config_reader_read_threads == 1',
                'config_reader_batch_size == 4',
                'config_dataset_format_npz == False',
                'config_train_epochs == 10',
                "config_dataset_num_files_train == 168",
                "config_reader_prefetch_workers == False"
            ],
            "deepspeed_all_bad": [
                "workload_name == 'megatron_deepspeed_llnl'",
                "config_num_nodes == 32",
                "config_train_computation_time == 2.44",
                "config_checkpoint_scr == False",
            ],
            "deepspeed_all_good": [
                "workload_name == 'megatron_deepspeed_llnl'",
                "config_num_nodes == 32",
                "config_train_computation_time == 2.44",
                "config_checkpoint_scr == True",
                "config_checkpoint_scr_flush_async == True"
            ],
        }
    }

    holdout_df_list = []
    for holdout_name, holdout_conditions in holdout_dict[framework].items():
        holdout_df = df.query(" & ".join(holdout_conditions))
        print(f"holdout_df.shape for {holdout_name}", holdout_df.shape)
        holdout_df["holdout_name"] = holdout_name
        holdout_df_list.append(holdout_df)

    holdout_df_full = pd.concat(holdout_df_list)
    print("holdout_df.shape", holdout_df_full.shape)

    holdout_workload_config_ids = holdout_df_full["workload_config_id"].unique()
    print("holdout_workload_config_ids", holdout_workload_config_ids)

    rest_df = df[~df["workload_config_id"].isin(holdout_workload_config_ids)]
    print("rest_df.shape", rest_df.shape)
    
    rest_workload_config_ids = pd.Series(rest_df["workload_config_id"].unique())
    train_workload_config_ids = rest_workload_config_ids.sample(frac=0.8, random_state=42)
    test_workload_config_ids = rest_workload_config_ids[~rest_workload_config_ids.isin(train_workload_config_ids)]

    train_df_full = rest_df[rest_df["workload_config_id"].isin(train_workload_config_ids)]
    test_df_full = rest_df[rest_df["workload_config_id"].isin(test_workload_config_ids)]

    # --- Create I/O-Bound Datasets ---
    train_df_io_bound = train_df_full[train_df_full['compute_time_frac_epoch'] < threshold].copy()
    test_df_io_bound = test_df_full[test_df_full['compute_time_frac_epoch'] < threshold].copy()
    holdout_df_io_bound = holdout_df_full[holdout_df_full['compute_time_frac_epoch'] < threshold].copy()

    # --- Save Datasets ---
    os.makedirs(output_dir, exist_ok=True)

    # Full datasets
    holdout_df_full.to_parquet(f"{output_dir}/ml_data_{framework}_holdout_full.parquet")
    train_df_full.to_parquet(f"{output_dir}/ml_data_{framework}_train_full.parquet")
    test_df_full.to_parquet(f"{output_dir}/ml_data_{framework}_test_full.parquet")
    
    # I/O-bound datasets
    holdout_df_io_bound.to_parquet(f"{output_dir}/ml_data_{framework}_holdout_io_bound.parquet")
    train_df_io_bound.to_parquet(f"{output_dir}/ml_data_{framework}_train_io_bound.parquet")
    test_df_io_bound.to_parquet(f"{output_dir}/ml_data_{framework}_test_io_bound.parquet")

    print("\n--- Dataset Creation Summary ---")
    print(f"Framework: {framework}, View: {view_type}, Threshold: {threshold}")
    print(f"  train_full: {train_df_full.shape}")
    print(f"  test_full: {test_df_full.shape}")
    print(f"  holdout_full: {holdout_df_full.shape}")
    print(f"  train_io_bound: {train_df_io_bound.shape}")
    print(f"  test_io_bound: {test_df_io_bound.shape}")
    print(f"  holdout_io_bound: {holdout_df_io_bound.shape}")
    print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test/holdout datasets.")
    parser.add_argument("--framework", type=str, required=True, help="Framework (e.g., tensorflow, pytorch)")
    parser.add_argument("--view_type", type=str, required=True, help="View type (e.g., epoch, time_range)")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for compute_time_frac_epoch to determine I/O-bound runs.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before creating new datasets.")
    args = parser.parse_args()
    create_train_test_df(args.framework, args.view_type, args.threshold, args.clean)
