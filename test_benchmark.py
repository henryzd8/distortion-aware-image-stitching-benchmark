from pathlib import Path
import json

import numpy as np

from analyze_results import (
    check_complete,
    check_results,
    load_all,
    paired_deltas,
)
from distortcorrect_stitcher_gpu import (
    DistortCorrectStitcher,
    DistortCorrectStitcherGPU,
)
from run_experiments import check_manifest
from sequential_stitcher_gpu import SequentialStitcher, SequentialStitcherGPU


ROOT = Path(__file__).resolve().parent


def test_cpu_and_cuda_classes_are_distinct():
    assert DistortCorrectStitcher is not DistortCorrectStitcherGPU
    assert SequentialStitcher is not SequentialStitcherGPU


def test_cpu_sequential_pipeline_runs():
    rng = np.random.default_rng(7)
    source = rng.random((64, 96), dtype=np.float32)
    tiles = np.stack((source[:, :64], source[:, 32:96]))
    positions = np.array([[0, 0], [0, 32]])
    result = SequentialStitcher(
        interpolation_order=1,
        gamma_schedule=[1.0],
        boundary_frac=0.35,
        local_search=0,
        border=4,
    ).run(tiles, positions, do_sharpen=False)
    assert result.positions.shape == positions.shape
    assert result.tiles.shape == tiles.shape
    assert result.mosaic.ndim == 2


def test_primary_results_are_complete_and_paired():
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    check_manifest(ROOT / "benchmark", manifest)
    methods = ("dcs_paper_style", "sequential_paper_matched")
    rows = load_all(ROOT / "results" / "distortcorrect", manifest)
    rows.extend(load_all(ROOT / "results" / "sequential", manifest))
    check_complete(rows, manifest, methods)
    check_results(rows)
    deltas = paired_deltas(rows, *methods)
    assert len(deltas) == manifest["case_count"]


def main():
    test_cpu_and_cuda_classes_are_distinct()
    test_cpu_sequential_pipeline_runs()
    test_primary_results_are_complete_and_paired()
    print("benchmark checks passed")


if __name__ == "__main__":
    main()
