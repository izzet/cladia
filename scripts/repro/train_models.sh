#!/usr/bin/env bash
set -euo pipefail

# Created for artifact reviewers: train the paper model profiles with defaults
# instead of requiring reviewers to pass raw q_method/q_low/q_high settings.

REPRO_ROOT="${REPRO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
GLOBALS_DIR="${GLOBALS_DIR:-$REPRO_ROOT/globals}"
export DFD_REPRO_GLOBALS_DIR="$GLOBALS_DIR"
export DFD_REPRO_RESULTS_DIR="${DFD_REPRO_RESULTS_DIR:-$REPRO_ROOT/results}"
RESULTS_DIR="${RESULTS_DIR:-$REPRO_ROOT/results/repro_training_runs}"
FRAMEWORKS="${FRAMEWORKS:-pytorch tensorflow}"
VIEW_TYPE="${VIEW_TYPE:-epoch}"
TARGET_COL="${TARGET_COL:-compute_time_frac_epoch}"
FEATURE_GROUPS="${FEATURE_GROUPS:-0 1 2}"
mkdir -p "$RESULTS_DIR"

run_profile() {
    local framework="$1"
    local profile="$2"
    local q_low q_high
    case "$profile" in
        iqr)
            q_low=25
            q_high=75
            ;;
        tail)
            q_low=90
            q_high=95
            ;;
        *)
            printf 'Unknown quantile profile: %s\n' "$profile" >&2
            exit 1
            ;;
    esac

    local fg_suffix
    fg_suffix="$(tr -d ' ' <<<"$FEATURE_GROUPS")"
    local log_path="$RESULTS_DIR/${framework}_${VIEW_TYPE}_${TARGET_COL}_fg${fg_suffix}_${profile}.txt"
    printf 'Training %s profile for %s; log=%s\n' "$profile" "$framework" "$log_path"
    uv run python -m dfdiagnoser_ml.training \
        --framework "$framework" \
        --view_type "$VIEW_TYPE" \
        --target_col "$TARGET_COL" \
        --feature_groups $FEATURE_GROUPS \
        --q_method mc \
        --q_low "$q_low" \
        --q_high "$q_high" \
        >"$log_path" 2>&1
}

for framework in $FRAMEWORKS; do
    run_profile "$framework" iqr
    run_profile "$framework" tail
done
