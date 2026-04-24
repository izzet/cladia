#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import re
import warnings
import yaml
from pathlib import Path

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

CATEGORICAL_CONFIG_PARAMS = ['config_dataset_format', 'config_framework']
DATASET_FORMAT_COLUMNS = {
    'tfrecord': 'config_dataset_format_tfrecord',
    'mmap_indexed_binary': 'config_dataset_format_mmap_indexed_binary',
    'npz': 'config_dataset_format_npz',
}
REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = os.environ.get('DFD_REPRO_WORK_DIR', str(REPO_ROOT))
GLOBALS_DIR = os.environ.get('DFD_REPRO_GLOBALS_DIR', f"{WORK_DIR}/globals")
TRACE_PATHS_CI = os.environ.get('DFD_REPRO_TRACE_PATHS_CI', f"{GLOBALS_DIR}/trace_paths_ci.csv")
DEFAULT_TRACE_PATHS = f"{GLOBALS_DIR}/trace_paths.csv"
TRACE_PATHS_CUSTOM = (
    os.environ.get('DFD_REPRO_TRACE_PATHS_CUSTOM')
    or os.environ.get('DFD_REPRO_TRACE_PATHS')
    or (DEFAULT_TRACE_PATHS if Path(DEFAULT_TRACE_PATHS).exists() else f"{GLOBALS_DIR}/trace_paths_custom.csv")
)
CHECKPOINT_CI_DIR = os.environ.get('DFD_REPRO_CHECKPOINT_CI_DIR', f"{GLOBALS_DIR}/checkpoints_ci")
CHECKPOINT_CUSTOM_DIR = os.environ.get('DFD_REPRO_CHECKPOINT_CUSTOM_DIR', f"{GLOBALS_DIR}/checkpoints")
SIZE_RELATED_METRICS = [
    '_bw',
    '_intensity',
    '_size',
    '_xfer',
]


def _as_dir(path):
    return str(Path(path)) + '/'


CHECKPOINT_CI_DIR = _as_dir(CHECKPOINT_CI_DIR)
CHECKPOINT_CUSTOM_DIR = _as_dir(CHECKPOINT_CUSTOM_DIR)


def drop_unrelated_cols(df):
    drop_cols = [c for c in df.columns if any([c.endswith(m) for m in SIZE_RELATED_METRICS]) and 'posix' not in c]
    return df.drop(columns=drop_cols)


def set_cat_cols(df):
    df = df.copy()
    if 'config_framework' in df.columns:
        normalized_framework = df['config_framework'].fillna('').astype(str).str.lower()
        df['config_framework_tensorflow'] = normalized_framework.eq('tensorflow')
        df.drop(columns=['config_framework'], inplace=True)

    if 'config_dataset_format' in df.columns:
        normalized_format = df['config_dataset_format'].fillna('').astype(str).str.lower()
        for dataset_format, column_name in DATASET_FORMAT_COLUMNS.items():
            df[column_name] = normalized_format.eq(dataset_format)
        df.drop(columns=['config_dataset_format'], inplace=True)

    for col in CATEGORICAL_CONFIG_PARAMS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    return df


def set_full_id(row):
    framework = 'pytorch'
    if bool(row.get('config_framework_tensorflow', False)):
        framework = 'tensorflow'
    file_type = 'unknown'
    if bool(row.get('config_dataset_format_tfrecord', False)):
        file_type = 'tfrecord'
    elif bool(row.get('config_dataset_format_mmap_indexed_binary', False)):
        file_type = 'mmap_bin'
    elif bool(row.get('config_dataset_format_npz', False)):
        file_type = 'npz'
    else:
        file_type = 'hdf5'
    try:
        xfer_size = int(row['config_reader_transfer_size'])
    except (TypeError, ValueError):
        xfer_size = 0
    full_id = [
        row['workload_name'].lower(),
        framework,
        f"n{str(row['config_num_nodes']).zfill(2)}",
        f"p{str(row['config_num_processes']).zfill(2)}",
        f"e{str(row['config_train_epochs']).zfill(2)}",
        f"b{str(row['config_reader_batch_size']).zfill(2)}",
        f"r{str(row['config_reader_read_threads']).zfill(2)}",
        f"x{xfer_size}",
        file_type,
        f"f{row['config_dataset_num_files_train']}",
        f"fs{row['config_dataset_num_samples_per_file']}",
    ]
    if bool(row.get('config_reader_prefetch_workers', False)):
        full_id.append('prefetch')
    if bool(row.get('config_checkpoint_scr', False)):
        scr = ['scr']
        if row['config_checkpoint_scr_cache_size'] > 0:
            scr.append(f"c{row['config_checkpoint_scr_cache_size']}")
        else:
            scr.append('c0')
        if row['config_checkpoint_scr_file_buf_size'] > 0:
            scr.append(f"b{row['config_checkpoint_scr_file_buf_size']}")
        else:
            scr.append('b0')
        full_id.append('_'.join(scr))
    if row.get('config_train_computation_time', 0) > 0:
        full_id.append(
            f"t{str(row['config_train_computation_time']).replace('.', '')}",
        )
    full_id.append(str(row['version']))
    full_id.append(str(row['ci_date']))
    return '-'.join(full_id)


def create_workload_df():
    workload_frames = []
    if TRACE_PATHS_CI and Path(TRACE_PATHS_CI).exists():
        ci_workload_df = pd.read_csv(TRACE_PATHS_CI)
        ci_workload_df['num_nodes'] = ci_workload_df['trace_path'].str.extract(r'nodes-(\d+)').astype(int)  # temp fix
        ci_workload_df['run_id'] = ci_workload_df['trace_path'].map(lambda x: x.split('/')[7])
        ci_workload_df['run_id'] = (
            ci_workload_df['version'] + '_' + ci_workload_df['run_id'] + '_n' + ci_workload_df['num_nodes'].astype(str)
        )
        ci_workload_df['checkpoint_dir'] = CHECKPOINT_CI_DIR + ci_workload_df['run_id']
        ci_workload_df.rename(columns={'num_nodes': 'config_num_nodes'}, inplace=True)
        workload_frames.append(ci_workload_df)
    else:
        print(f'[Workload] CI trace manifest not found; skipping: {TRACE_PATHS_CI}')

    if TRACE_PATHS_CUSTOM and Path(TRACE_PATHS_CUSTOM).exists():
        custom_workload_df = pd.read_csv(TRACE_PATHS_CUSTOM)
        custom_workload_df['run_id'] = (
            custom_workload_df['version']
            + '_'
            + custom_workload_df['workload_name']
            + '_'
            + custom_workload_df['ci_date'].astype(str)
        )
        custom_workload_df['checkpoint_dir'] = CHECKPOINT_CUSTOM_DIR + custom_workload_df['run_id']
        custom_workload_df.rename(columns={'num_nodes': 'config_num_nodes'}, inplace=True)
        workload_frames.append(custom_workload_df)
    else:
        print(f'[Workload] Custom trace manifest not found; skipping: {TRACE_PATHS_CUSTOM}')

    if not workload_frames:
        raise FileNotFoundError(
            'No trace manifest found. Set DFD_REPRO_TRACE_PATHS_CUSTOM or '
            'DFD_REPRO_TRACE_PATHS_CI to an existing CSV.'
        )

    workload_df = pd.concat(workload_frames, ignore_index=True)
    workload_df.drop(columns=['num_nodes'], errors='ignore', inplace=True)
    workload_df = workload_df.query('workload_name != "dlrm"')
    invalid_rows = []

    for i, row in workload_df.iterrows():
        trace_root_path = Path(row['trace_path']).parent
        if row['version'] == 'custom':
            trace_root_path = Path(row['trace_path'])
        dlio_log_path = trace_root_path / 'dlio.log'
        hydra_config_path = trace_root_path / '.hydra/config.yaml'
        if not dlio_log_path.exists() or not hydra_config_path.exists():
            print(f"[Workload] Missing trace metadata; skipping run_id={row['run_id']} trace_root={trace_root_path}")
            invalid_rows.append(i)
            continue
        max_epoch = 0
        with open(dlio_log_path, 'r') as dlio_log_file:
            for line in dlio_log_file:
                process_match = re.search(r'Running DLIO .* with (\d+) process\(es\)', line)
                if process_match:
                    num_total_processes = int(process_match.group(1))
                    workload_df.loc[i, 'config_num_processes'] = (
                        num_total_processes / workload_df.loc[i, 'config_num_nodes']
                    )
                epoch_match = re.search(r'Ending epoch (\d+) - \d+ steps completed', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    max_epoch = max(max_epoch, current_epoch)

        workload_df.loc[i, 'config_train_epochs'] = max_epoch
        with open(hydra_config_path, 'r') as hydra_config_file:
            hydra_config = yaml.safe_load(hydra_config_file)
        workload_df.loc[i, 'config_framework'] = hydra_config['workload']['framework']

        if 'checkpoint' in hydra_config['workload']:
            workload_df.loc[i, 'config_checkpoint_scr'] = False
            workload_df.loc[i, 'config_checkpoint_scr_cache_size'] = 0
            workload_df.loc[i, 'config_checkpoint_scr_file_buf_size'] = 0
            workload_df.loc[i, 'config_checkpoint_scr_flush'] = False
            workload_df.loc[i, 'config_checkpoint_scr_flush_async'] = False
            if 'checkpoint_mechanism_classname' in hydra_config['workload']['checkpoint']:
                workload_df.loc[i, 'config_checkpoint_scr'] = True

                env_path = None
                for candidate in ('env.txt', 'run_env.txt'):
                    candidate_path = trace_root_path / candidate
                    if candidate_path.exists():
                        env_path = candidate_path
                        break
                if env_path is not None:
                    with open(env_path, 'r') as env_file:
                        for line in env_file:
                            if 'SCR_CACHE_SIZE' in line:
                                scr_cache_size = int(line.split('=')[1].strip())
                                workload_df.loc[i, 'config_checkpoint_scr_cache_size'] = scr_cache_size
                            if 'SCR_FILE_BUF_SIZE' in line:
                                scr_file_buf_size = int(line.split('=')[1].strip())
                                workload_df.loc[i, 'config_checkpoint_scr_file_buf_size'] = scr_file_buf_size
                            if 'SCR_FLUSH' in line and 'SCR_FLUSH_TYPE' not in line:
                                scr_flush = int(line.split('=')[1].strip()) == 1
                                workload_df.loc[i, 'config_checkpoint_scr_flush'] = scr_flush
                            if 'SCR_FLUSH_ASYNC' in line:
                                scr_flush_async = int(line.split('=')[1].strip()) == 1
                                workload_df.loc[i, 'config_checkpoint_scr_flush_async'] = scr_flush_async

        config_pairs = [
            ('dataset', 'format', None),
            ('dataset', 'num_files_train', None),
            ('dataset', 'num_samples_per_file', None),
            ('dataset', 'record_length_bytes', None),
            ('reader', 'batch_size', None),
            ('reader', 'prefetch_workers', False),
            ('reader', 'read_threads', None),
            ('reader', 'transfer_size', None),
            ('train', 'computation_time', None),
        ]
        for conf_parent, conf, default_conf in config_pairs:
            conf_name = f"config_{conf_parent}_{conf}"
            if conf in hydra_config['workload'][conf_parent]:
                workload_df.loc[i, conf_name] = hydra_config['workload'][conf_parent][conf]
            elif default_conf is not None:
                workload_df.loc[i, conf_name] = default_conf

    int_config_cols = [
        'config_checkpoint_scr_cache_size',
        'config_checkpoint_scr_file_buf_size',
        'config_dataset_num_files_train',
        'config_dataset_num_samples_per_file',
        'config_dataset_record_length_bytes',
        'config_num_nodes',
        'config_num_processes',
        'config_reader_batch_size',
        'config_reader_read_threads',
        'config_train_epochs',
    ]
    existing_int_config_cols = [c for c in int_config_cols if c in workload_df.columns]
    workload_df[existing_int_config_cols] = workload_df[existing_int_config_cols].astype(int)
    if invalid_rows:
        workload_df = workload_df.drop(index=invalid_rows).copy()
    workload_df = set_cat_cols(workload_df)
    workload_df['full_id'] = workload_df.apply(set_full_id, axis=1)
    workload_df = workload_df.set_index('run_id')
    workload_df = workload_df.sort_index(axis=1)
    return workload_df

def load_view_data(workload_df, view_type, progress_interval=10):
    config_cols = [c for c in workload_df.columns if c.startswith('config_')]
    per_run_data = {}
    total = len(workload_df)
    loaded = 0
    skipped = 0
    not_found = 0

    print(f'[Load] Reading parquet for view_type={view_type} for', total, 'runs')
    for idx, (run_id, workload) in enumerate(workload_df.iterrows(), start=1):
        if idx == 1 or idx % progress_interval == 0 or idx == total:
            print(f'[Load] {idx}/{total} run_id={run_id}')
        try:
            df = pd.read_parquet(workload['checkpoint_dir'] + f"/_flat_view_{view_type}_5.parquet").reset_index()
        except FileNotFoundError:
            print('[Skip] Checkpoint not found for', run_id)
            not_found += 1
            continue
        if 'reader_posix_size_max' in df.columns:
            print('[Skip] Wrong layer hierarchy', run_id)
            skipped += 1
            continue
        if 'reader_time_q10_q90_mean' not in df.columns:
            print('[Skip] Missing quantile calc. columns', run_id)
            skipped += 1
            continue
        if 'reader_posix_lustre_read_size_bin_0_4kib_sum' not in df.columns:
            print('[Skip] Missing size bin columns', run_id)
            skipped += 1
            continue
        df['run_id'] = run_id
        df['workload_name'] = workload['workload_name']
        df[config_cols] = workload[config_cols]
        per_run_data[run_id] = df
        loaded += 1
    print(f'[Load] Done: loaded={loaded}, skipped={skipped}, not_found={not_found}')
    return per_run_data


def create_data_from_loaded(per_run_data):
    if len(per_run_data) == 0:
        print('[CreateData] No data loaded; returning empty dataframe')
        return pd.DataFrame()
    print('[CreateData] Concatenating data...')
    all_data = pd.concat(list(per_run_data.values()), ignore_index=True)
    print('[CreateData] Replacing inf and nan...')
    numeric_cols = all_data.select_dtypes(include=np.number).columns
    all_data[numeric_cols] = all_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    try:
        all_data = drop_unrelated_cols(all_data)
        print('[CreateData] Unrelated columns dropped')
    except ValueError:
        print('[CreateData] Error dropping unrelated columns; skipping')
        pass
    try:
        all_data = set_cat_cols(all_data)
        print('[CreateData] Categorical columns set')
    except ValueError:
        print('[CreateData] Error setting categorical columns; skipping')
        pass
    print('[CreateData] Sorting data...')
    all_data = all_data.sort_index(axis=1)
    print('[CreateData] Done')
    return all_data

def create_agg_data_from_all_data(all_data, view_type):
    if all_data is None or len(all_data) == 0:
        print('[CreateAggAll] Empty all_data; returning empty dataframe')
        return pd.DataFrame()
    print('[CreateAggAll] Preparing aggregation dictionary...')
    agg_dict = {col: 'mean' for col in all_data.columns if col}
    agg_dict.update({col: 'sum' for col in all_data.columns if col.endswith('_sum')})
    agg_dict.update({col: 'max' for col in all_data.columns if col.endswith('_max')})
    agg_dict.update({col: 'min' for col in all_data.columns if col.endswith('_min')})
    agg_dict.update({col: 'max' for col in all_data.columns if col.endswith('_nunique')})
    agg_dict['workload_name'] = 'first'
    agg_dict[view_type] = 'nunique'
    del agg_dict['run_id']
    print('[CreateAggAll] Grouping by run_id and aggregating...')
    grouped = all_data.groupby('run_id').agg(agg_dict)
    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    print('[CreateAggAll] Cleaning columns...')
    grouped = grouped.drop(columns=['file_name', 'proc_name', view_type], errors='ignore')
    try:
        grouped = drop_unrelated_cols(grouped)
        print('[CreateAggAll] Unrelated columns dropped')
    except ValueError:
        print('[CreateAggAll] Error dropping unrelated columns; skipping')
        pass
    try:
        grouped = set_cat_cols(grouped)
        print('[CreateAggAll] Categorical columns set')
    except ValueError:
        print('[CreateAggAll] Error setting categorical columns; skipping')
        pass
    grouped = grouped.sort_index(axis=1)
    return grouped


def split_by_framework(df):
    if df is None or len(df) == 0:
        empty = pd.DataFrame(columns=[] if df is None else df.columns)
        return empty.copy(), empty.copy()
    framework_series = df.get('config_framework_tensorflow', pd.Series(False, index=df.index))
    framework_mask = framework_series.fillna(False).astype(bool)
    pytorch_data = df.loc[~framework_mask].copy()
    tensorflow_data = df.loc[framework_mask].copy()
    return pytorch_data, tensorflow_data


def main():
    # Improved progress reporting
    print('Building workload dataframe...')
    Path(GLOBALS_DIR).mkdir(parents=True, exist_ok=True)
    workload_df = create_workload_df()
    print('Workload dataframe created with', len(workload_df), 'rows')
    workload_df.to_parquet(f"{GLOBALS_DIR}/ml_workload_all.parquet")

    # Cache to avoid re-reading per view type
    view_raw_data = {}

    for view_type in ['epoch']:
    # for view_type in ['time_range', 'epoch']:
    # for view_type in ['time_range', 'epoch']:
        print(f'\n[Start] Processing view_type={view_type}')
        Path(f'{GLOBALS_DIR}/{view_type}').mkdir(parents=True, exist_ok=True)
        raw_data = load_view_data(workload_df, view_type, progress_interval=10)
        view_raw_data[view_type] = raw_data

        all_data = create_data_from_loaded(raw_data)
        all_data.to_parquet(f'{GLOBALS_DIR}/{view_type}/ml_data_all.parquet')
        print(f'[Write] Full parquet datasets for {view_type} written')
        pytorch_data, tensorflow_data = split_by_framework(all_data)
        pytorch_data.to_parquet(f'{GLOBALS_DIR}/{view_type}/ml_data_pytorch_all.parquet')
        print(f'[Write] PyTorch parquet datasets for {view_type} written')
        tensorflow_data.to_parquet(f'{GLOBALS_DIR}/{view_type}/ml_data_tensorflow_all.parquet')
        print(f'[Write] TensorFlow parquet datasets for {view_type} written')

        # all_data = pd.read_parquet(f'{GLOBALS_DIR}/ml_data_{view_type}_all.parquet')
        # pytorch_data = all_data.query('config_framework_tensorflow == False')
        # tensorflow_data = all_data.query('config_framework_tensorflow == True')

        all_data.columns.to_series().to_csv(f'{GLOBALS_DIR}/{view_type}/ml_data_all_columns.csv', index=False)
        pytorch_data.columns.to_series().to_csv(f'{GLOBALS_DIR}/{view_type}/ml_data_pytorch_all_columns.csv', index=False)
        tensorflow_data.columns.to_series().to_csv(f'{GLOBALS_DIR}/{view_type}/ml_data_tensorflow_all_columns.csv', index=False)
        print(f'[Write] Columns for {view_type} written')

        print('[Agg] Aggregating from all_data...')
        all_agg_data = create_agg_data_from_all_data(all_data, view_type)
        all_agg_data.to_parquet(f'{GLOBALS_DIR}/{view_type}/ml_agg_data_all.parquet')
        print(f'[Write] All aggregated parquet datasets for {view_type} written')
        pytorch_agg_data, tensorflow_agg_data = split_by_framework(all_agg_data)
        pytorch_agg_data.to_parquet(f'{GLOBALS_DIR}/{view_type}/ml_agg_data_pytorch.parquet')
        print(f'[Write] PyTorch aggregated parquet datasets for {view_type} written')
        tensorflow_agg_data.to_parquet(f'{GLOBALS_DIR}/{view_type}/ml_agg_data_tensorflow.parquet')
        print(f'[Write] TensorFlow aggregated parquet datasets for {view_type} written')

        print(f'[Stats] Number of workloads for {view_type}', len(workload_df))
        print(f'[Stats] All agg data size for {view_type}', len(all_agg_data))
        print(f'[Stats] PyTorch agg data size for {view_type}', len(pytorch_agg_data))
        print(f'[Stats] TensorFlow agg data size for {view_type}', len(tensorflow_agg_data))
        print(f'[Stats] All data size for {view_type}', len(all_data))
        print(f'[Stats] PyTorch data size for {view_type}', len(pytorch_data))
        print(f'[Stats] TensorFlow data size for {view_type}', len(tensorflow_data))


if __name__ == "__main__":
    main()
