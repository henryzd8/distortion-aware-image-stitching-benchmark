"""Regression and completeness checks for the submitted benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

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
Record = dict[str, Any]
Manifest = dict[str, Any]
CaseFilter = Callable[[dict[str, Any]], bool]


def test_cpu_and_cuda_classes_are_distinct() -> None:
    """Ensure CPU and CUDA adapters remain separate classes."""
    assert DistortCorrectStitcher is not DistortCorrectStitcherGPU
    assert SequentialStitcher is not SequentialStitcherGPU


def test_cpu_sequential_pipeline_runs() -> None:
    """Run a small CPU smoke test through the sequential pipeline."""
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


def _manifest_for(manifest: Manifest, cases: set[str]) -> Manifest:
    """Return a manifest containing only the requested case names."""
    manifest = dict(manifest)
    manifest["cases"] = [c for c in manifest["cases"] if c["case"] in cases]
    manifest["case_count"] = len(manifest["cases"])
    return manifest


def _check_dir(
    manifest: Manifest,
    rel_dir: str,
    methods: tuple[str, ...],
    case_filter: CaseFilter | None = None,
) -> list[Record]:
    """Validate one result directory against the expected method set."""
    cases = [c for c in manifest["cases"]]
    if case_filter is not None:
        cases = [c for c in cases if case_filter(c)]
    names = {c["case"] for c in cases}
    sub_manifest = _manifest_for(manifest, names)
    rows = load_all(ROOT / rel_dir, sub_manifest)
    check_complete(rows, sub_manifest, methods)
    check_results(rows)
    return rows


def test_primary_results_are_complete_and_paired() -> None:
    """Check the primary joint/sequential result pairing."""
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    check_manifest(ROOT / "benchmark", manifest)
    methods = ("dcs_paper_style", "sequential_paper_matched")
    rows = load_all(ROOT / "results" / "distortcorrect", manifest)
    rows.extend(load_all(ROOT / "results" / "sequential", manifest))
    check_complete(rows, manifest, methods)
    check_results(rows)
    deltas = paired_deltas(rows, *methods)
    assert len(deltas) == manifest["case_count"]


def test_feedback_ablation_is_complete() -> None:
    """Check all feedback iteration arms across the primary cases."""
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("dcs_paper_iter_1", "dcs_paper_iter_2",
               "dcs_paper_iter_5", "dcs_paper_iter_10")
    rows = _check_dir(manifest, "results/ablation_feedback", methods)
    assert len(rows) == manifest["case_count"] * len(methods)


def test_prestitch_ablation_is_complete() -> None:
    """Check both no-pre-stitch diagnostic arms."""
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("dcs_paper_no_prestitch", "sequential_paper_no_prestitch")
    rows = _check_dir(manifest, "results/ablation_prestitch", methods)
    assert len(rows) == manifest["case_count"] * len(methods)


def test_search_ablation_is_complete_on_distorted_cases() -> None:
    """Check search-range arms on nonzero-distortion cases."""
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("sequential_paper_bound_010", "sequential_paper_bound_020")
    rows = _check_dir(manifest, "results/ablation_search", methods,
                      case_filter=lambda c: abs(c["k1_true"]) > 1e-12)
    assert len(rows) == 24 * len(methods)


def test_oracle_ablation_is_complete_on_distorted_cases() -> None:
    """Check the known-coefficient sequential diagnostic."""
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("oracle_k1_paper",)
    rows = _check_dir(manifest, "results/ablation_oracle", methods,
                      case_filter=lambda c: abs(c["k1_true"]) > 1e-12)
    assert len(rows) == 24


def test_joint_oracle_injects_the_true_k1() -> None:
    """Check that the joint known-coefficient control reaches its target."""
    # The oracle control must drive the cumulative correction to the injected
    # coefficient regardless of the search, without changing the loop shape.
    rng = np.random.default_rng(7)
    source = rng.random((64, 96), dtype=np.float32)
    tiles = np.stack((source[:, :64], source[:, 32:96]))
    positions = np.array([[0, 0], [0, 32]])
    st = DistortCorrectStitcher(
        interpolation_order=1,
        n_iterations=3,
        boundary_frac=0.35,
        local_search=0,
        border=4,
        pre_stitch_gamma_schedule=[],
        oracle_k1=0.004,
    )
    result = st.run(tiles, positions, do_sharpen=False)
    k1_total = float(np.sum(np.asarray(result.k1_history)[:, 0]))
    assert abs(k1_total - 0.004) < 1e-12
    assert result.k1_history[0][0] == 0.004  # full residual on the first step


def test_joint_oracle_results_are_complete_on_distorted_cases() -> None:
    """Check the persisted joint known-coefficient result directory."""
    manifest = json.loads((ROOT / "benchmark" / "manifest.json").read_text())
    methods = ("dcs_paper_joint_oracle",)
    rows = _check_dir(
        manifest,
        "results/ablation_joint_oracle",
        methods,
        case_filter=lambda c: abs(c["k1_true"]) > 1e-12,
    )
    assert len(rows) == 24


def test_k1_magnitude_results_are_complete() -> None:
    """Check all methods in the lower-magnitude benchmark."""
    bench = ROOT / "benchmark_k1_magnitude"
    manifest = json.loads((bench / "manifest.json").read_text())
    check_manifest(bench, manifest)
    methods = (
        "dcs_paper_style",
        "sequential_paper_matched",
        "sequential_paper_no_prestitch",
    )
    rows = load_all(ROOT / "results/ablation_k1_magnitude", manifest)
    check_complete(rows, manifest, methods)
    check_results(rows)
    assert len(rows) == manifest["case_count"] * len(methods)


def main() -> None:
    """Run the benchmark checks without requiring a test runner."""
    test_cpu_and_cuda_classes_are_distinct()
    test_cpu_sequential_pipeline_runs()
    test_primary_results_are_complete_and_paired()
    test_feedback_ablation_is_complete()
    test_prestitch_ablation_is_complete()
    test_search_ablation_is_complete_on_distorted_cases()
    test_oracle_ablation_is_complete_on_distorted_cases()
    test_joint_oracle_injects_the_true_k1()
    test_joint_oracle_results_are_complete_on_distorted_cases()
    test_k1_magnitude_results_are_complete()
    print("benchmark checks passed")


if __name__ == "__main__":
    main()
