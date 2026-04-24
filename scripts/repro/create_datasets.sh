#!/usr/bin/env bash
set -euo pipefail

# Created for artifact reviewers: build the train/test/holdout parquet splits
# from the postprocessed globals directory with the paper defaults.

REPRO_ROOT="${REPRO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
GLOBALS_DIR="${GLOBALS_DIR:-$REPRO_ROOT/globals}"
export DFD_REPRO_GLOBALS_DIR="$GLOBALS_DIR"
FRAMEWORKS="${FRAMEWORKS:-pytorch tensorflow}"
VIEW_TYPE="${VIEW_TYPE:-epoch}"
THRESHOLD="${THRESHOLD:-0.9}"
CLEAN_DATASETS="${CLEAN_DATASETS:-1}"

clean_flag=()
if [[ "$CLEAN_DATASETS" == "1" ]]; then
    clean_flag=(--clean)
fi

for framework in $FRAMEWORKS; do
    printf 'Creating datasets for %s from %s\n' "$framework" "$GLOBALS_DIR"
    uv run python -m dfdiagnoser_ml.create_dataset \
        --framework "$framework" \
        --view_type "$VIEW_TYPE" \
        --threshold "$THRESHOLD" \
        "${clean_flag[@]}"
    clean_flag=()
done
