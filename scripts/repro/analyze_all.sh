#!/usr/bin/env bash
set -euo pipefail

# Created for artifact reviewers: analyze reviewer-generated DLIO traces only.
# This intentionally does not rely on author-side CI trace data.

REPRO_ROOT="${REPRO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
GLOBALS_DIR="${GLOBALS_DIR:-$REPRO_ROOT/globals}"
TRACE_MANIFEST="${1:-${TRACE_MANIFEST:-$GLOBALS_DIR/trace_paths.csv}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GLOBALS_DIR/checkpoints}"
FLUX_LOGS_DIR="${FLUX_LOGS_DIR:-$REPRO_ROOT/logs/repro_trace_analysis}"
ANALYZE_SINGLE="${ANALYZE_SINGLE:-$REPRO_ROOT/scripts/analyze_single.sh}"
DEFAULT_QUEUE="${DEFAULT_QUEUE:-pdebug}"
DEFAULT_TIME_LIMIT="${DEFAULT_TIME_LIMIT:-60}"
LONG_QUEUE="${LONG_QUEUE:-pbatch}"
LONG_TIME_LIMIT="${LONG_TIME_LIMIT:-300}"

if [[ ! -f "$TRACE_MANIFEST" ]]; then
    printf 'Trace manifest not found: %s\n' "$TRACE_MANIFEST" >&2
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR" "$FLUX_LOGS_DIR"

tail -n +2 "$TRACE_MANIFEST" | while IFS=, read -r version workload_name node_num ci_date trace_path size_bytes size_formatted; do
    workload_id="${version}_${workload_name}_${ci_date}"
    log_path="$FLUX_LOGS_DIR/$workload_id"
    queue="$DEFAULT_QUEUE"
    time_limit="$DEFAULT_TIME_LIMIT"

    case "$workload_name" in
        *cosmoflow*|*deepspeed*|*resnet*)
            queue="$LONG_QUEUE"
            time_limit="$LONG_TIME_LIMIT"
            ;;
    esac

    if [[ -f "$CHECKPOINT_DIR/$workload_id/_flat_view_epoch_5.parquet" ]]; then
        printf 'Skipping %s; analyzer output already exists.\n' "$workload_id"
        continue
    fi

    printf 'Submitting analyzer job for %s\n' "$workload_id"
    flux_cmd=(
        flux submit
        --job-name="dd_ml_$workload_id"
        --queue="$queue"
        --time-limit="$time_limit"
        --output="$log_path.out"
        --error="$log_path.err"
        "$ANALYZE_SINGLE" "$trace_path" "$workload_id" "$CHECKPOINT_DIR"
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[DRY_RUN] '
        printf '%q ' "${flux_cmd[@]}"
        printf '\n'
        continue
    fi

    "${flux_cmd[@]}"
done

printf 'Analyzer jobs submitted.\n'
