#!/usr/bin/env bash
set -euo pipefail

# Created for artifact reviewers: submit the explicit paper reproduction run
# plan through Flux. DRY_RUN=1 prints commands without submitting.

REPRO_ROOT="${REPRO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
SWEEP_CSV="${SWEEP_CSV:-$REPRO_ROOT/scripts/repro/dlio_sweep_config_full.csv}"
INNER_SCRIPT="${INNER_SCRIPT:-$REPRO_ROOT/scripts/repro/run_dlio_config_inner.sh}"
if [[ -z "${DLIO_BENCHMARK_DIR:-}" && -d "$REPRO_ROOT/dlio_benchmark" ]]; then
    DLIO_BENCHMARK_DIR="$REPRO_ROOT/dlio_benchmark"
fi
: "${DLIO_BENCHMARK_DIR:?Set DLIO_BENCHMARK_DIR to the DLIO benchmark checkout, or initialize dlio_benchmark/ in the repository root.}"

FLUX_QUEUE="${FLUX_QUEUE:-pbatch}"
JOB_TIME="${JOB_TIME:-120}"
PPN="${PPN:-4}"
DRY_RUN="${DRY_RUN:-1}"
GENERATE_DATA="${GENERATE_DATA:-0}"
ONLY_WORKLOAD="${ONLY_WORKLOAD:-}"
LOG_DIR="${LOG_DIR:-$REPRO_ROOT/logs/repro_dlio_sweep}"
mkdir -p "$LOG_DIR"

split_values() {
    local value="$1"
    if [[ -z "$value" ]]; then
        printf '\n'
    else
        tr ';' '\n' <<<"$value"
    fi
}

safe_name() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

submit_config() {
    local workload_group="$1"
    local workload_name="$2"
    local node_count="$3"
    local ppn_value="$4"
    local reader_threads="$5"
    local reader_batch_size="$6"
    local reader_prefetch_workers="$7"
    local reader_transfer_size="$8"
    local dataset_format="$9"
    local num_files="${10}"
    local num_samples="${11}"
    local record_length_bytes="${12}"
    local checkpoint_steps="${13}"
    local checkpoint_mechanism="${14}"
    local scr_cache_size="${15}"
    local scr_file_buf_size="${16}"
    local scr_flush="${17}"
    local scr_flush_async="${18}"
    local train_epochs="${19}"
    local total_training_steps="${20}"
    local chunk_size="${21}"
    local computation_time="${22}"
    local repeat_index="${23}"

    ppn_value="${ppn_value:-$PPN}"
    local job_name_raw="${workload_group}_${workload_name}_n${node_count}_p${ppn_value}_r${reader_threads}_b${reader_batch_size}_d${dataset_format}_f${num_files}_s${num_samples}_e${train_epochs}_t${computation_time}_scr${scr_cache_size}_${scr_file_buf_size}_${scr_flush}_${scr_flush_async}_rep${repeat_index}"
    local job_name
    job_name="$(safe_name "$job_name_raw")"
    local env_args=(
        "DLIO_BENCHMARK_DIR=$DLIO_BENCHMARK_DIR"
        "WORKLOAD_NAME=$workload_name"
        "NUM_NODES=$node_count"
        "PPN=$ppn_value"
        "GENERATE_DATA=$GENERATE_DATA"
    )

    [[ -n "$reader_threads" ]] && env_args+=("READER_THREADS=$reader_threads")
    [[ -n "$reader_batch_size" ]] && env_args+=("READER_BATCH_SIZE=$reader_batch_size")
    [[ -n "$reader_prefetch_workers" && "$reader_prefetch_workers" != "false" ]] && env_args+=("READER_PREFETCH_WORKERS=$reader_prefetch_workers")
    [[ -n "$reader_transfer_size" ]] && env_args+=("READER_TRANSFER_SIZE=$reader_transfer_size")
    [[ -n "$dataset_format" ]] && env_args+=("DATASET_FORMAT=$dataset_format")
    [[ -n "$num_files" ]] && env_args+=("NUM_FILES=$num_files")
    [[ -n "$num_samples" ]] && env_args+=("NUM_SAMPLES=$num_samples")
    [[ -n "$record_length_bytes" ]] && env_args+=("RECORD_LENGTH_BYTES=$record_length_bytes")
    [[ -n "$checkpoint_steps" ]] && env_args+=("CHECKPOINT_STEPS=$checkpoint_steps")
    [[ -n "$checkpoint_mechanism" ]] && env_args+=("CHECKPOINT_MECHANISM=$checkpoint_mechanism")
    [[ -n "$scr_cache_size" ]] && env_args+=("SCR_CACHE_SIZE=$scr_cache_size")
    [[ -n "$scr_file_buf_size" ]] && env_args+=("SCR_FILE_BUF_SIZE=$scr_file_buf_size")
    [[ -n "$scr_flush" ]] && env_args+=("SCR_FLUSH=$scr_flush")
    [[ -n "$scr_flush_async" ]] && env_args+=("SCR_FLUSH_ASYNC=$scr_flush_async")
    [[ -n "$train_epochs" ]] && env_args+=("TRAIN_EPOCHS=$train_epochs")
    [[ -n "$total_training_steps" ]] && env_args+=("TOTAL_TRAINING_STEPS=$total_training_steps")
    [[ -n "$chunk_size" ]] && env_args+=("CHUNK_SIZE=$chunk_size")
    [[ -n "$computation_time" ]] && env_args+=("COMPUTATION_TIME=$computation_time")

    local flux_cmd=(
        flux alloc
        -t "$JOB_TIME"
        -q "$FLUX_QUEUE"
        -N "$node_count"
        --exclusive
        --broker-opts=--setattr=log-filename="$LOG_DIR/$job_name.flux.log"
        env "${env_args[@]}"
        "$INNER_SCRIPT"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] '
        printf '%q ' "${flux_cmd[@]}"
        printf '\n'
    else
        "${flux_cmd[@]}"
    fi
}

run_extended_plan() {
    tail -n +2 "$SWEEP_CSV" | while IFS=, read -r plan_component workload_group workload_name observed_workload_names source_versions node_values ppn_value reader_values reader_batch_size reader_prefetch_workers reader_transfer_size format_values file_values sample_values record_length_bytes checkpoint_steps checkpoint_mechanism scr_cache_size scr_file_buf_size scr_flush scr_flush_async train_epochs total_steps chunk_size computation_time repeat_count; do
        [[ -z "$workload_name" ]] && continue
        if [[ -n "$ONLY_WORKLOAD" && "$workload_name" != "$ONLY_WORKLOAD" && "$workload_group" != "$ONLY_WORKLOAD" && "$plan_component" != "$ONLY_WORKLOAD" ]]; then
            continue
        fi
        repeat_count="${repeat_count:-1}"
        while IFS= read -r node_count; do
            while IFS= read -r reader_threads; do
                while IFS= read -r dataset_format; do
                    while IFS= read -r num_files; do
                        while IFS= read -r num_samples; do
                            for repeat_index in $(seq 1 "$repeat_count"); do
                                submit_config "$workload_group" "$workload_name" "$node_count" "$ppn_value" "$reader_threads" "$reader_batch_size" "$reader_prefetch_workers" "$reader_transfer_size" "$dataset_format" "$num_files" "$num_samples" "$record_length_bytes" "$checkpoint_steps" "$checkpoint_mechanism" "$scr_cache_size" "$scr_file_buf_size" "$scr_flush" "$scr_flush_async" "$train_epochs" "$total_steps" "$chunk_size" "$computation_time" "$repeat_index"
                            done
                        done < <(split_values "$sample_values")
                    done < <(split_values "$file_values")
                done < <(split_values "$format_values")
            done < <(split_values "$reader_values")
        done < <(split_values "$node_values")
    done
}

run_legacy_sweep() {
    tail -n +2 "$SWEEP_CSV" | while IFS=, read -r workload_group workload_name node_values reader_values format_values file_values sample_values checkpoint_steps checkpoint_mechanism train_epochs total_steps chunk_size computation_time repeat_count; do
        [[ -z "$workload_name" ]] && continue
        if [[ -n "$ONLY_WORKLOAD" && "$workload_name" != "$ONLY_WORKLOAD" && "$workload_group" != "$ONLY_WORKLOAD" ]]; then
            continue
        fi
        repeat_count="${repeat_count:-1}"
        while IFS= read -r node_count; do
            while IFS= read -r reader_threads; do
                while IFS= read -r dataset_format; do
                    while IFS= read -r num_files; do
                        while IFS= read -r num_samples; do
                            for repeat_index in $(seq 1 "$repeat_count"); do
                                submit_config "$workload_group" "$workload_name" "$node_count" "$PPN" "$reader_threads" "" "" "" "$dataset_format" "$num_files" "$num_samples" "" "$checkpoint_steps" "$checkpoint_mechanism" "" "" "" "" "$train_epochs" "$total_steps" "$chunk_size" "$computation_time" "$repeat_index"
                            done
                        done < <(split_values "$sample_values")
                    done < <(split_values "$file_values")
                done < <(split_values "$format_values")
            done < <(split_values "$reader_values")
        done < <(split_values "$node_values")
    done
}

header="$(head -n 1 "$SWEEP_CSV")"
if [[ "$header" == plan_component,* ]]; then
    run_extended_plan
else
    run_legacy_sweep
fi
