#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="${REPRO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

if [[ -z "$VENV_ACTIVATE" ]]; then
    for candidate in "$REPRO_ROOT/.venv/bin/activate" "$REPRO_ROOT/.venv-py310/bin/activate"; do
        if [[ -f "$candidate" ]]; then
            VENV_ACTIVATE="$candidate"
            break
        fi
    done
fi

if [[ -n "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

trace_path="${1:?usage: analyze_single.sh TRACE_PATH WORKLOAD_ID CHECKPOINT_DIR}"
workload_id="${2:?usage: analyze_single.sh TRACE_PATH WORKLOAD_ID CHECKPOINT_DIR}"
checkpoint_dir="${3:?usage: analyze_single.sh TRACE_PATH WORKLOAD_ID CHECKPOINT_DIR}"
output_root="${DFD_ANALYZE_OUTPUT_DIR:-$REPRO_ROOT/results/repro_analysis_outputs}"
scratch_base="${DFD_ANALYZE_SCRATCH_BASE:-${TMPDIR:-/tmp}/dfdiagnoser-ml-dask}"
notify_topic="${DFD_ANALYZE_NOTIFY_TOPIC:-}"

run_dfanalyzer() {
    if command -v dfanalyzer >/dev/null 2>&1; then
        dfanalyzer "$@"
    else
        uv run dfanalyzer "$@"
    fi
}

stage_input() {
    local source_path="$1"
    local staged_root="$2"

    if [[ "${DFD_ANALYZE_STAGE_INPUT:-1}" != "1" ]]; then
        printf '%s\n' "$source_path"
        return 0
    fi

    mkdir -p "$staged_root"

    if [[ -d "$source_path" ]]; then
        local staged_dir="$staged_root/$(basename "$source_path")"
        mkdir -p "$staged_dir"
        find "$source_path" -mindepth 1 -maxdepth 1 | while IFS= read -r item; do
            ln -sfn "$item" "$staged_dir/$(basename "$item")"
        done
        printf '%s\n' "$staged_dir"
        return 0
    fi

    local staged_file="$staged_root/$(basename "$source_path")"
    ln -sfn "$source_path" "$staged_file"
    printf '%s\n' "$staged_file"
}

notify() {
    local message="$1"
    if [[ -z "$notify_topic" ]]; then
        return 0
    fi
    curl -fsS -m 10 -d "$message" "https://ntfy.sh/$notify_topic" >/dev/null || true
}

validate_trace_input() {
    local source_path="$1"
    if find "$source_path" -maxdepth 1 \( -name '*.pfw' -o -name '*.pfw.gz' \) | grep -q .; then
        return 0
    fi

    if find "$source_path" -maxdepth 1 -name 'bedrock_*_group.json' | grep -q .; then
        printf 'Trace directory %s does not contain file-backed dftracer traces (*.pfw / *.pfw.gz).\n' "$source_path" >&2
        printf 'Found Bedrock/Mofka group metadata instead; this offline analyzer wrapper expects the file-backed traces emitted by the reviewer DLIO launcher.\n' >&2
        return 1
    fi

    printf 'Trace directory %s does not contain any *.pfw or *.pfw.gz files.\n' "$source_path" >&2
    return 1
}

echo "Processing trace at: $trace_path"
echo "Using workload ID: $workload_id"

mkdir -p "$checkpoint_dir" "$output_root"

scratch_dir="$scratch_base/$workload_id"
mkdir -p "$scratch_dir"
rm -rf "$scratch_dir"/*
echo "Created scratch directory: $scratch_dir"

validate_trace_input "$trace_path"
analysis_input="$(stage_input "$trace_path" "$scratch_dir/input")"
echo "Analyzer input path: $analysis_input"

notify "Starting analysis: $workload_id"

run_dfanalyzer analyzer/preset=dlio \
    analyzer.assign_epochs=True \
    analyzer.checkpoint=True \
    analyzer.checkpoint_dir="$checkpoint_dir/$workload_id" \
    analyzer.quantile_stats=True \
    analyzer.time_granularity=5 \
    analyzer.time_sliced=True \
    cluster=local \
    cluster.local_directory="$scratch_dir" \
    hydra.run.dir="$output_root/$workload_id" \
    hydra.runtime.output_dir="$output_root/$workload_id" \
    input.path="$analysis_input" \
    view_types=[proc_name,time_range,epoch]

rm -rf "$scratch_dir"
echo "Removed scratch directory: $scratch_dir"

notify "Analysis completed: $workload_id"
