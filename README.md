# Distortion-Aware Stitching Benchmark

A fixed synthetic microscopy benchmark comparing iterative feedback between
position estimation and radial-distortion correction with one-shot sequential
estimate--correct--stitching.  The benchmark uses real MERFISH microscopy
texture while retaining exact ground-truth tile positions and injected
distortion coefficients.

## Repository layout

- `benchmark/`: primary 36-case benchmark manifest and local input data.
- `benchmark_k1_magnitude/`: 24-case lower-magnitude benchmark at
  `k1 = -0.004, -0.002, +0.002, +0.004`.
- `results/comparison/`: primary paired result summaries.
- `results/distortcorrect/`: primary joint-method JSON results.
- `results/sequential/`: primary sequential-method JSON results.
- `results/ablation_feedback/`: 1-, 2-, 5-, and 10-iteration feedback results.
- `results/ablation_prestitch/`: no-pre-stitch diagnostic results.
- `results/ablation_search/`: sequential search-range sensitivity results.
- `results/ablation_oracle/`: known-distortion diagnostic results.
- `results/ablation_joint_oracle/`: joint known-distortion diagnostic results.
- `results/ablation_k1_magnitude/`: raw lower-magnitude ablation results.
- `results/ablation/`: clean combined CSVs, compact per-ablation CSVs, and
  `ablation_results.xlsx` with one sheet per ablation.
- `figures/`: publication figures generated from the completed results.
- `run_final_experiment.py`, `run_experiments.py`: method and batch runners.
- `run_ablations.py`: reproduces the five standard ablation runs.
- `analyze_results.py`, `make_figures.py`, `make_ablation_csv.py`: analysis,
  figure, and export utilities.
- `test_benchmark.py`: completeness, provenance, and smoke checks.

Technical benchmark inputs are kept out of Git by the repository `.gitignore`
because of their size; the manifests and result tables remain available for
inspection.

## Rebuild the readable exports

The local lower-magnitude results are discovered automatically, so this command
regenerates the combined CSVs, all compact CSVs, and the multi-sheet workbook:

```bash
python make_ablation_csv.py --bench benchmark
```

To use an alternate copy of the magnitude experiment, override both paths:

```bash
python make_ablation_csv.py --bench benchmark \
  --k1-magnitude-dir /path/to/results_ablation_k1_magnitude \
  --k1-magnitude-bench /path/to/benchmark_k1_magnitude
```

The workbook includes a `K1 magnitude` sheet with 72 case-method rows covering
the joint paper-style arm, the pre-stitched sequential arm, and the sequential
no-pre-stitch diagnostic arm.

## Analyze and reproduce

The primary summaries can be regenerated with:

```bash
python analyze_results.py \
  --bench benchmark \
  --results results/comparison \
  --methods dcs_paper_style,sequential_paper_matched
```

Publication figures can be regenerated with:

```bash
python make_figures.py --bench benchmark --out figures
```

The primary and standard ablation runs use the resumable batch runner.  CUDA
execution requires one worker because the GPU adapter temporarily redirects
shared alignment functions within the process:

```bash
python run_experiments.py --bench benchmark --out results/distortcorrect \
  --methods dcs_paper_style --device cuda --workers 1
python run_experiments.py --bench benchmark --out results/sequential \
  --methods sequential_paper_matched --device cuda --workers 1
python run_ablations.py --bench benchmark --device cuda
```

## Verification

```bash
python test_benchmark.py
python -m py_compile *.py
```

The result directories contain per-case JSON records for provenance and the
`results/ablation/` directory contains the cleaner tables intended for reading.
