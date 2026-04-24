# DFDiagnoser-ML

ML-based diagnosis of HPC I/O bottlenecks in deep learning

DFDiagnoser-ML studies deep-learning I/O behavior on HPC systems by combining
DLIO trace data, multi-layer postprocessing, and machine-learning models for
diagnosis and interpretation. This public repository contains the code,
configuration sweep definitions, tests, and reproducibility scripts for
rebuilding the paper artifacts. Generated data under `globals/` and `results/`
is not versioned here; the workflow below creates those directories locally.

## Installation

Requirements:

- Python 3.10 or newer
- `uv`
- the public `dfanalyzer` and `dlio_benchmark` submodules

Initialize the required submodules and create the environment:

```bash
git submodule update --init dfanalyzer dlio_benchmark
uv sync
source .venv/bin/activate
```

If you also want the development tools:

```bash
uv sync --group dev
```

For the supported workflow in this repository, only `dfanalyzer/` and
`dlio_benchmark/` are required.

## Workflow

The repository supports a standard end-to-end workflow:

1. Collect DLIO outputs and trace data for the workloads of interest.
2. Analyze the traces with `dfanalyzer` to produce checkpointed flat views.
3. Postprocess analyzed runs into workload- and epoch-level parquet tables
   under `globals/`.
4. Create train/test/holdout datasets for PyTorch and TensorFlow.
5. Train diagnostic models and generate result tables and interpretation
   outputs.
6. Summarize case studies and downstream paper artifacts.

The main implementation lives in:

- `scripts/` for processing and automation
- `scripts/repro/` for reviewer-facing reproducibility helpers
- `dfdiagnoser_ml/` for dataset creation, feature selection, training, and
  evaluation

## Reproducibility

The supported paper pipeline is under `scripts/repro/`. The DLIO execution
steps assume a Flux-based HPC environment. The offline analyzer scripts expect
DLIO output directories that contain file-backed `trace-*.pfw` or
`trace-*.pfw.gz` traces. This public repository does not ship generated
traces, parquet tables, trained models, or result summaries.

### 1. Prepare the environment

```bash
git submodule update --init dfanalyzer dlio_benchmark
uv sync
source .venv/bin/activate
```

### 2. Inspect the reproduction sweep inputs

The repository includes the explicit sweep descriptions used by the
reproducibility workflow:

- `scripts/repro/dlio_sweep_config_base.csv`
- `scripts/repro/dlio_sweep_config_full.csv`

The full sweep can be previewed without submission:

```bash
DRY_RUN=1 bash scripts/repro/run_dlio_sweep_flux.sh
```

### 3. Run the DLIO sweep

Submit the configured DLIO sweep through Flux:

```bash
DRY_RUN=0 FLUX_QUEUE=pbatch bash scripts/repro/run_dlio_sweep_flux.sh
```

`scripts/repro/run_dlio_config_inner.sh` is the per-configuration worker used
inside each Flux allocation.

### 4. Build the trace manifest

After the DLIO runs complete, build a manifest over the produced output
directories:

```bash
TRACE_ROOT=/path/to/dlio-benchmark-output \
  bash scripts/repro/create_trace_manifest.sh /path/to/trace_paths.csv
```

### 5. Analyze traces into checkpoints

Run `dfanalyzer` over the trace manifest to create flat-view checkpoints:

```bash
TRACE_MANIFEST=/path/to/trace_paths.csv \
CHECKPOINT_DIR=/path/to/checkpoints \
  bash scripts/repro/analyze_all.sh /path/to/trace_paths.csv
```

For a single run, the lower-level entry point is:

```bash
bash scripts/analyze_single.sh TRACE_PATH RUN_ID CHECKPOINT_DIR
```

### 6. Postprocess analyzed runs

Create the workload- and epoch-level parquet files used by the ML pipeline:

```bash
GLOBALS_DIR=/path/to/globals \
TRACE_MANIFEST=/path/to/trace_paths.csv \
CHECKPOINT_DIR=/path/to/checkpoints \
  bash scripts/repro/postprocess.sh /path/to/trace_paths.csv
```

### 7. Create train/test/holdout datasets

Build the framework-specific datasets from the postprocessed `globals/`
artifacts:

```bash
GLOBALS_DIR=/path/to/globals \
  bash scripts/repro/create_datasets.sh
```

### 8. Train the model profiles

Run the default paper training profiles:

```bash
GLOBALS_DIR=/path/to/globals \
DFD_REPRO_RESULTS_DIR=/path/to/results \
  bash scripts/repro/train_models.sh
```

By default this trains both frameworks and emits result tables, model files,
and SHAP-based summaries under the chosen results directory.

### 9. Summarize the case-study artifacts

Generate broad grouped summaries from the processed globals and model results:

```bash
uv run python scripts/repro/summarize_case_studies.py \
  --globals-dir /path/to/globals \
  --results-dir /path/to/results \
  --output-dir /path/to/repro_summary
```

Generate the paper-aligned baseline-versus-optimized comparisons with:

```bash
uv run python scripts/repro/compare_case_studies.py \
  --globals-dir /path/to/globals \
  --results-dir /path/to/results \
  --output-dir /path/to/repro_compare
```

The first script provides broad grouped summaries. The second reproduces the
paper-facing CosmoFlow, motivation, UNet3D, and Megatron-DeepSpeed comparison
outputs from locally generated `globals/` and `results/`.
