# Distortion-Aware Stitching Benchmark

A fixed synthetic microscopy benchmark comparing the paper-style
DistortCorrect Stitcher pipeline with a paper-matched sequential baseline.

## Structure

- `benchmark/`: benchmark manifest; TIFF crops and NPZ cases are excluded.
- `results/`: per-case results, summaries, and comparison figure.
- `distortcorrect_stitcher_gpu.py`: joint pipeline.
- `sequential_stitcher_gpu.py`: sequential baseline.
- `run_final_experiment.py`: method execution.
- `run_experiments.py`: batch runner.
- `analyze_results.py`: paired analysis.
