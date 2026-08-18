# Distortion-Aware Stitching Benchmark

A fixed synthetic microscopy benchmark comparing iterative feedback between
position estimation and radial-distortion correction against a one-shot
sequential estimate-correct-stitch pipeline.

## Structure

- `benchmark/`: benchmark manifest; TIFF crops and NPZ cases are excluded.
- `results/distortcorrect/`: paper-style joint pipeline per-case results.
- `results/sequential/`: paper-matched sequential baseline per-case results.
- `results/ablation_feedback/`: 1, 2, 5, and 10 joint-iteration results.
- `results/ablation_prestitch/`: joint and sequential no-pre-stitch results.
- `results/ablation_search/`: sequential one-shot search-range sensitivity.
- `results/ablation_oracle/`: oracle-`k1` diagnostic control.
- `distortcorrect_stitcher_gpu.py`: joint pipeline.
- `sequential_stitcher_gpu.py`: sequential baseline.
- `run_final_experiment.py`: method execution.
- `run_experiments.py`: batch runner.
- `run_ablations.py`: reproduces all four ablation runs.
- `analyze_results.py`: paired analysis.
- `test_benchmark.py`: completeness and provenance checks.

## Primary analysis

```bash
python analyze_results.py \
  --bench benchmark \
  --results results/sequential \
  --reference-results results/distortcorrect \
  --methods dcs_paper_style,sequential_paper_matched \
  --plot
```

## Ablation analysis

Each ablation is analyzed against its paper-style reference arm.  The search
and oracle ablations cover only the 24 nonzero-distortion cases.

```bash
python analyze_results.py \
  --bench benchmark \
  --results results/ablation_feedback \
  --reference-results results/distortcorrect \
  --methods dcs_paper_iter_1,dcs_paper_iter_2,dcs_paper_iter_5,dcs_paper_iter_10,dcs_paper_style

python analyze_results.py \
  --bench benchmark \
  --results results/ablation_prestitch \
  --reference-results results/distortcorrect \
  --methods dcs_paper_no_prestitch,sequential_paper_no_prestitch,dcs_paper_style

python analyze_results.py \
  --bench benchmark \
  --results results/ablation_search \
  --reference-results results/sequential \
  --methods sequential_paper_bound_010,sequential_paper_bound_020,sequential_paper_matched \
  --cases distorted

python analyze_results.py \
  --bench benchmark \
  --results results/ablation_oracle \
  --reference-results results/sequential \
  --methods oracle_k1_paper,sequential_paper_matched \
  --cases distorted
```

## Reproducing the runs

The primary run and each ablation use the same resumable runner.  CUDA
execution requires one worker.

```bash
python run_experiments.py --bench benchmark --out results/distortcorrect \
  --methods dcs_paper_style --device cuda --workers 1
python run_experiments.py --bench benchmark --out results/sequential \
  --methods sequential_paper_matched --device cuda --workers 1
python run_ablations.py --bench benchmark --device cuda
```
