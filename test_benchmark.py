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


def _manifest_for(manifest, cases):
    manifest = dict(manifest)
    manifest["cases"] = [c for c in manifest["cases"] if c["case"] in cases]
    manifest["case_count"] = len(manifest["cases"])
    return manifest


def _check_dir(manifest, rel_dir, methods, case_filter=None):
    cases = [c for c in manifest["cases"]]
    if case_filter is not None:
        cases = [c for c in cases if case_filter(c)]
    names = {c["case"] for c in cases}
    sub_manifest = _manifest_for(manifest, names)
    rows = load_all(ROOT / rel_dir, sub_manifest)
    check_complete(rows, sub_manifest, methods)
    check_results(rows)
    return rows


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


def test_feedback_ablation_is_complete():
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("dcs_paper_iter_1", "dcs_paper_iter_2",
               "dcs_paper_iter_5", "dcs_paper_iter_10")
    rows = _check_dir(manifest, "results/ablation_feedback", methods)
    assert len(rows) == manifest["case_count"] * len(methods)


def test_prestitch_ablation_is_complete():
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("dcs_paper_no_prestitch", "sequential_paper_no_prestitch")
    rows = _check_dir(manifest, "results/ablation_prestitch", methods)
    assert len(rows) == manifest["case_count"] * len(methods)


def test_search_ablation_is_complete_on_distorted_cases():
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("sequential_paper_bound_010", "sequential_paper_bound_020")
    rows = _check_dir(manifest, "results/ablation_search", methods,
                      case_filter=lambda c: abs(c["k1_true"]) > 1e-12)
    assert len(rows) == 24 * len(methods)


def test_oracle_ablation_is_complete_on_distorted_cases():
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("oracle_k1_paper",)
    rows = _check_dir(manifest, "results/ablation_oracle", methods,
                      case_filter=lambda c: abs(c["k1_true"]) > 1e-12)
    assert len(rows) == 24


def main():
    test_cpu_and_cuda_classes_are_distinct()
    test_cpu_sequential_pipeline_runs()
    test_primary_results_are_complete_and_paired()
    test_feedback_ablation_is_complete()
    test_prestitch_ablation_is_complete()
    test_search_ablation_is_complete_on_distorted_cases()
    test_oracle_ablation_is_complete_on_distorted_cases()
    print("benchmark checks passed")


if __name__ == "__main__":
    main()
