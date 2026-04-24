#!/usr/bin/env bash
set -euo pipefail

# Created for artifact reviewers: run post-processing with reviewer-generated
# traces and without assuming author-side CI trace data exists.

REPRO_ROOT="${REPRO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
GLOBALS_DIR="${GLOBALS_DIR:-$REPRO_ROOT/globals}"
TRACE_MANIFEST="${1:-${TRACE_MANIFEST:-$GLOBALS_DIR/trace_paths.csv}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GLOBALS_DIR/checkpoints}"

export DFD_REPRO_WORK_DIR="$REPRO_ROOT"
export DFD_REPRO_GLOBALS_DIR="$GLOBALS_DIR"
export DFD_REPRO_TRACE_PATHS="$TRACE_MANIFEST"
export DFD_REPRO_TRACE_PATHS_CUSTOM="$TRACE_MANIFEST"
export DFD_REPRO_TRACE_PATHS_CI="${DFD_REPRO_TRACE_PATHS_CI:-}"
export DFD_REPRO_CHECKPOINT_CUSTOM_DIR="$CHECKPOINT_DIR"

uv run python "$REPRO_ROOT/scripts/postproc_all.py"
