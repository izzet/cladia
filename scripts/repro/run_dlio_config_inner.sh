#!/usr/bin/env bash
set -euo pipefail

# Created for artifact reviewers: run one DLIO configuration inside a Flux
# allocation and emit the trace/config files consumed by the reproduction flow.

REPRO_ROOT="${REPRO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [[ -z "${DLIO_BENCHMARK_DIR:-}" && -d "$REPRO_ROOT/dlio_benchmark" ]]; then
    DLIO_BENCHMARK_DIR="$REPRO_ROOT/dlio_benchmark"
fi
: "${DLIO_BENCHMARK_DIR:?Set DLIO_BENCHMARK_DIR to the DLIO benchmark checkout, or initialize dlio_benchmark/ in the repository root.}"
: "${WORKLOAD_NAME:?Set WORKLOAD_NAME.}"
: "${NUM_NODES:?Set NUM_NODES.}"

cd "$DLIO_BENCHMARK_DIR"
if [[ -f "${DLIO_SETUP_ENV:-setup_env.sh}" ]]; then
    # shellcheck disable=SC1090
    source "${DLIO_SETUP_ENV:-setup_env.sh}"
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    printf 'No python interpreter found on PATH.\n' >&2
    exit 1
fi

if [[ -z "${DLIO_WORKLOAD_CONFIG_DIR:-}" ]]; then
    if [[ -d "$DLIO_BENCHMARK_DIR/dlio_benchmark/configs/workload" ]]; then
        DLIO_WORKLOAD_CONFIG_DIR="$DLIO_BENCHMARK_DIR/dlio_benchmark/configs/workload"
    elif [[ -d "$DLIO_BENCHMARK_DIR/libs/dlio_benchmark/dlio_benchmark/configs/workload" ]]; then
        DLIO_WORKLOAD_CONFIG_DIR="$DLIO_BENCHMARK_DIR/libs/dlio_benchmark/dlio_benchmark/configs/workload"
    fi
fi
: "${DLIO_WORKLOAD_CONFIG_DIR:?Could not find DLIO workload configs. Set DLIO_WORKLOAD_CONFIG_DIR.}"

export NUM_NODES="${NUM_NODES}"
export PPN="${PPN:-4}"
export DFTRACER_BIND_SIGNALS="${DFTRACER_BIND_SIGNALS:-0}"
export DFTRACER_ENABLE="${DFTRACER_ENABLE:-1}"
export DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA:-1}"
export DFTRACER_TRACE_COMPRESSION="${DFTRACER_TRACE_COMPRESSION:-1}"
export DARSHAN_DUMP_CONFIG="${DARSHAN_DUMP_CONFIG:-1}"
export DARSHAN_ENABLE="${DARSHAN_ENABLE:-0}"

yaml_get() {
    "$PYTHON_BIN" - "$DLIO_WORKLOAD_CONFIG_DIR" "$1" "$2" <<'PY'
import sys
import yaml
from pathlib import Path

config_dir, workload_name, dotted = sys.argv[1], sys.argv[2], sys.argv[3]
config_path = Path(config_dir) / f"{workload_name}.yaml"
with config_path.open() as fh:
    data = yaml.safe_load(fh)
value = data
for part in dotted.split("."):
    value = value[part]
print(value)
PY
}

yaml_get_optional() {
    "$PYTHON_BIN" - "$DLIO_WORKLOAD_CONFIG_DIR" "$1" "$2" <<'PY'
import sys
import yaml
from pathlib import Path

config_dir, workload_name, dotted = sys.argv[1], sys.argv[2], sys.argv[3]
config_path = Path(config_dir) / f"{workload_name}.yaml"
with config_path.open() as fh:
    data = yaml.safe_load(fh)
value = data
for part in dotted.split("."):
    if not isinstance(value, dict) or part not in value:
        print("")
        raise SystemExit
    value = value[part]
print(value)
PY
}

WORKLOAD_LOWER="$(printf '%s' "$WORKLOAD_NAME" | tr '[:upper:]' '[:lower:]')"
DATASET_FORMAT="${DATASET_FORMAT:-$(yaml_get "$WORKLOAD_NAME" dataset.format)}"
NUM_FILES="${NUM_FILES:-$(yaml_get "$WORKLOAD_NAME" dataset.num_files_train)}"
NUM_SAMPLES="${NUM_SAMPLES:-$(yaml_get "$WORKLOAD_NAME" dataset.num_samples_per_file)}"
RECORD_LENGTH_BYTES="${RECORD_LENGTH_BYTES:-$(yaml_get_optional "$WORKLOAD_NAME" dataset.record_length_bytes)}"
READER_THREADS="${READER_THREADS:-$(yaml_get "$WORKLOAD_NAME" reader.read_threads)}"
READER_BATCH_SIZE="${READER_BATCH_SIZE:-$(yaml_get_optional "$WORKLOAD_NAME" reader.batch_size)}"
READER_PREFETCH_WORKERS="${READER_PREFETCH_WORKERS:-$(yaml_get_optional "$WORKLOAD_NAME" reader.prefetch_workers)}"
READER_TRANSFER_SIZE="${READER_TRANSFER_SIZE:-$(yaml_get_optional "$WORKLOAD_NAME" reader.transfer_size)}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-$(yaml_get "$WORKLOAD_NAME" train.epochs)}"

DATA_ROOT="${DLIO_DATA_ROOT:-/p/lustre3/${USER}/dlio-benchmark-test}"
OUTPUT_ROOT="${DLIO_OUTPUT_ROOT:-/p/lustre3/${USER}/dlio-benchmark-output}"
DATA_DIR="${DATA_ROOT}/${WORKLOAD_LOWER}_${DATASET_FORMAT}_${NUM_FILES}"
CHECKPOINT_DIR="${DATA_DIR}/checkpoint"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ "${GENERATE_DATA:-0}" == "1" ]]; then
    OUTPUT_DIR="${OUTPUT_ROOT}/${WORKLOAD_LOWER}/data_gen_${DATASET_FORMAT}_${NUM_FILES}_${TIMESTAMP}"
else
    OUTPUT_DIR="${OUTPUT_ROOT}/${WORKLOAD_LOWER}/train_${TIMESTAMP}"
fi

mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR"
rm -rf "${CHECKPOINT_DIR:?}/"*

export DARSHAN_LOGPATH="$OUTPUT_DIR"
export DARSHAN_LOGFILE="$OUTPUT_DIR/${WORKLOAD_LOWER}.darshan"
export DFTRACER_DATA_DIR="$DATA_DIR"

hydra_args=(
    "hydra.run.dir=$OUTPUT_DIR"
    "workload=$WORKLOAD_NAME"
    "++workload.checkpoint.checkpoint_folder=$CHECKPOINT_DIR"
    "++workload.dataset.data_folder=$DATA_DIR/data"
    "++workload.dataset.format=$DATASET_FORMAT"
    "++workload.dataset.num_files_train=$NUM_FILES"
    "++workload.dataset.num_samples_per_file=$NUM_SAMPLES"
    "++workload.output.folder=$OUTPUT_DIR"
    "++workload.reader.read_threads=$READER_THREADS"
)

if [[ -n "$RECORD_LENGTH_BYTES" ]]; then
    hydra_args+=("++workload.dataset.record_length_bytes=$RECORD_LENGTH_BYTES")
fi
if [[ -n "$READER_BATCH_SIZE" ]]; then
    hydra_args+=("++workload.reader.batch_size=$READER_BATCH_SIZE")
fi
if [[ -n "$READER_PREFETCH_WORKERS" ]]; then
    hydra_args+=("++workload.reader.prefetch_workers=$READER_PREFETCH_WORKERS")
fi
if [[ -n "$READER_TRANSFER_SIZE" ]]; then
    hydra_args+=("++workload.reader.transfer_size=$READER_TRANSFER_SIZE")
fi

if [[ "${GENERATE_DATA:-0}" == "1" ]]; then
    hydra_args+=(
        "++workload.workflow.generate_data=True"
        "++workload.workflow.train=False"
    )
else
    hydra_args+=(
        "++workload.train.epochs=$TRAIN_EPOCHS"
        "++workload.workflow.generate_data=False"
        "++workload.workflow.train=True"
    )
fi

if [[ -n "${CHECKPOINT_STEPS:-}" ]]; then
    hydra_args+=("++workload.checkpoint.steps_between_checkpoints=$CHECKPOINT_STEPS")
fi
if [[ -n "${CHUNK_SIZE:-}" ]]; then
    hydra_args+=("++workload.dataset.chunk_size=$CHUNK_SIZE")
fi
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    hydra_args+=("++workload.train.total_training_steps=$TOTAL_TRAINING_STEPS")
fi
if [[ -n "${COMPUTATION_TIME:-}" ]]; then
    hydra_args+=("++workload.train.computation_time=$COMPUTATION_TIME")
fi
if [[ -n "${CHECKPOINT_MECHANISM:-}" ]]; then
    hydra_args+=("++workload.checkpoint.checkpoint_mechanism_classname=$CHECKPOINT_MECHANISM")
    SCR_CACHE_BASE="${DLIO_SSD_ROOT:-/l/ssd/${USER}/dlio-benchmark-test}/checkpoint"
    SCR_CACHE_DIR="${SCR_CACHE_BASE}/${WORKLOAD_NAME}_${TIMESTAMP}"
    mkdir -p "$SCR_CACHE_DIR"
    export DFTRACER_DATA_DIR="$DATA_DIR:$SCR_CACHE_DIR"
    export SCR_CACHE_BASE="$SCR_CACHE_DIR"
    export SCR_CACHE_BYPASS=0
    export SCR_CACHE_DIR="$SCR_CACHE_DIR"
    export SCR_CACHE_PURGE=1
    export SCR_CACHE_SIZE="${SCR_CACHE_SIZE:-2}"
    export SCR_COPY_TYPE="${SCR_COPY_TYPE:-SINGLE}"
    export SCR_DEBUG="${SCR_DEBUG:-1}"
    export SCR_FILE_BUF_SIZE="${SCR_FILE_BUF_SIZE:-33554432}"
    export SCR_FLUSH="${SCR_FLUSH:-1}"
    export SCR_FLUSH_ASYNC="${SCR_FLUSH_ASYNC:-1}"
    export SCR_FLUSH_TYPE="${SCR_FLUSH_TYPE:-PTHREAD}"
    export SCR_PREFIX="$CHECKPOINT_DIR"
    export SCR_PREFIX_PURGE=1
fi

{
    printf '{\n'
    printf '  "workload_name": "%s",\n' "$WORKLOAD_NAME"
    printf '  "num_nodes": %s,\n' "$NUM_NODES"
    printf '  "ppn": %s,\n' "$PPN"
    printf '  "dataset_format": "%s",\n' "$DATASET_FORMAT"
    printf '  "dataset_num_files_train": %s,\n' "$NUM_FILES"
    printf '  "dataset_num_samples_per_file": %s,\n' "$NUM_SAMPLES"
    printf '  "dataset_record_length_bytes": %s,\n' "${RECORD_LENGTH_BYTES:-0}"
    printf '  "reader_read_threads": %s,\n' "$READER_THREADS"
    printf '  "reader_batch_size": %s,\n' "${READER_BATCH_SIZE:-0}"
    printf '  "reader_prefetch_workers": "%s",\n' "${READER_PREFETCH_WORKERS:-}"
    printf '  "reader_transfer_size": %s,\n' "${READER_TRANSFER_SIZE:-0}"
    printf '  "train_epochs": %s,\n' "$TRAIN_EPOCHS"
    printf '  "train_computation_time": "%s",\n' "${COMPUTATION_TIME:-}"
    printf '  "total_training_steps": "%s",\n' "${TOTAL_TRAINING_STEPS:-}"
    printf '  "checkpoint_steps": "%s",\n' "${CHECKPOINT_STEPS:-}"
    printf '  "checkpoint_mechanism": "%s",\n' "${CHECKPOINT_MECHANISM:-}"
    printf '  "scr_cache_size": "%s",\n' "${SCR_CACHE_SIZE:-}"
    printf '  "scr_file_buf_size": "%s",\n' "${SCR_FILE_BUF_SIZE:-}"
    printf '  "scr_flush": "%s",\n' "${SCR_FLUSH:-}"
    printf '  "scr_flush_async": "%s",\n' "${SCR_FLUSH_ASYNC:-}"
    printf '  "output_dir": "%s"\n' "$OUTPUT_DIR"
    printf '}\n'
} >"$OUTPUT_DIR/config.json"
env >"$OUTPUT_DIR/env.txt"

printf 'Submitting DLIO job: workload=%s nodes=%s ppn=%s output=%s\n' "$WORKLOAD_NAME" "$NUM_NODES" "$PPN" "$OUTPUT_DIR"
flux submit -N "$NUM_NODES" -o cpu-affinity=off -o mpibind=off --tasks-per-node="$PPN" dlio_benchmark "${hydra_args[@]}"
job_id="$(flux job last)"
flux job attach "$job_id"
