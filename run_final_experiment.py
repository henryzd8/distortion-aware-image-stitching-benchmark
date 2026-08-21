"""Implement benchmark methods and compute per-case measurements."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile

from distortcorrect_stitcher_gpu import (
    DistortCorrectStitcher,
    DistortCorrectStitcherGPU,
    pr_stitching,
    sharpen_tiles,
    undistort_tiles,
)
from mosaic_metrics import metrics_vs_source, seam_error
from sequential_stitcher_gpu import (
    SequentialStitcher,
    SequentialStitcherGPU,
    correct_tiles_once,
    estimate_k1_once,
)

K1_BOUNDS = (-0.015, 0.015)
GAMMA = 0.5
N_ITERATIONS = 25
K1_TOL = 0.0
STABILIZED_JOINT_K1_TOL = 1e-5
INTERP_ORDER = 5
BOUNDARY_FRAC = 0.35
POSITION_DAMPING = 1.0
PRE_STITCH = [0.1, 0.2, 0.5, 1.0]
BORDER = 64
LOCAL_SEARCH = 0
PAPER_K1_BOUNDS = (-0.005, 0.005)
PAPER_SEQUENTIAL_K1_BOUNDS = (-0.015, 0.015)
PAPER_BOUNDARY_PX = 400
MI_QUANTIZATION = {
    "algorithm": "MiniBatchKMeans",
    "batch_size": 10000,
    "max_iter": 300,
    "random_seed": 1000,
}
WARP_BACKEND = "torch_grid_sample_bicubic"
PROTOCOL_VERSION = "2026-08-11-rigorous-v12"

PRIMARY_METHODS = (
    "dcs_paper_style", "joint_matched", "sequential_matched",
)
EXPLORATORY_METHODS = (
    "sequential_paper_matched", "adaptive_multistart", "joint_stabilized",
    "joint_matched_no_prestitch", "sequential_matched_no_prestitch",
    "gt_k1_control", "gt_position_control",
    "dcs_paper_iter_1", "dcs_paper_iter_2", "dcs_paper_iter_5",
    "dcs_paper_iter_10", "dcs_paper_no_prestitch",
    "sequential_paper_no_prestitch", "sequential_paper_bound_010",
    "sequential_paper_bound_020", "oracle_k1_paper",
    "dcs_paper_joint_oracle",
)
METHODS = PRIMARY_METHODS + EXPLORATORY_METHODS

PAPER_ITERATION_COUNTS = {
    "dcs_paper_iter_1": 1,
    "dcs_paper_iter_2": 2,
    "dcs_paper_iter_5": 5,
    "dcs_paper_iter_10": 10,
}
PAPER_SEQUENTIAL_SENSITIVITY_BOUNDS = {
    "sequential_paper_bound_010": (-0.010, 0.010),
    "sequential_paper_bound_020": (-0.020, 0.020),
}
PAPER_DCS_METHODS = {"dcs_paper_style", "dcs_paper_no_prestitch"} \
    | set(PAPER_ITERATION_COUNTS)
PAPER_SEQUENTIAL_METHODS = {"sequential_paper_matched",
                            "sequential_paper_no_prestitch"} \
    | set(PAPER_SEQUENTIAL_SENSITIVITY_BOUNDS)
Record = dict[str, Any]


def pos_stats(
    hat: np.ndarray,
    true: np.ndarray,
) -> tuple[float, float, float]:
    """Return translation-gauge-free RMSE, MAE, and maximum position error."""
    e = (hat - hat[0]).astype(float) - (true - true[0]).astype(float)
    rmse = float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))
    mae = float(np.mean(np.abs(e)))
    mx = float(np.max(np.abs(e)))
    return rmse, mae, mx


def at_bound(
    k1_increments: Sequence[float] | np.ndarray,
    bounds: tuple[float, float] = K1_BOUNDS,
    gamma: float = 1.0,
) -> bool:
    """Report whether any applied increment is near its configured bound."""
    lo, hi = (gamma * bounds[0], gamma * bounds[1])
    tol = 0.01 * (hi - lo)
    return bool(any(v <= lo + tol or v >= hi - tol for v in k1_increments))


def method_settings(method: str) -> dict[str, Any] | None:
    """Return serializable provenance settings for one method name."""
    if method in PAPER_DCS_METHODS:
        return {
            "paper_style": True,
            "n_iterations": PAPER_ITERATION_COUNTS.get(method, N_ITERATIONS),
            "pre_stitch_gamma_schedule": (
                "default" if method != "dcs_paper_no_prestitch"
                else "disabled"),
            "k1_bounds": list(PAPER_K1_BOUNDS),
            "k1_update_gamma": GAMMA,
            "boundary_px": PAPER_BOUNDARY_PX,
            "local_search": 0,
            "skip_diagonal": False,
            "update_order": "positions_first",
            "correction_mode": "incremental",
        }
    if method in PAPER_SEQUENTIAL_METHODS:
        return {
            "paper_style": True,
            "pre_stitch_gamma_schedule": (
                "default" if method != "sequential_paper_no_prestitch"
                else "disabled"),
            "k1_bounds": list(PAPER_SEQUENTIAL_SENSITIVITY_BOUNDS.get(
                method, PAPER_SEQUENTIAL_K1_BOUNDS)),
            "boundary_px": PAPER_BOUNDARY_PX,
            "local_search": 0,
            "skip_diagonal": False,
            "final_position_gamma": 1.0,
        }
    if method == "oracle_k1_paper":
        return {
            "paper_style": True,
            "pre_stitch_gamma_schedule": "default",
            "k1_source": "synthetic_ground_truth",
            "boundary_px": PAPER_BOUNDARY_PX,
            "local_search": 0,
            "skip_diagonal": False,
            "final_position_gamma": 1.0,
        }
    if method == "dcs_paper_joint_oracle":
        return {
            "paper_style": True,
            "n_iterations": N_ITERATIONS,
            "pre_stitch_gamma_schedule": "default",
            "k1_source": "synthetic_ground_truth",
            "k1_bounds": list(PAPER_K1_BOUNDS),
            "k1_update_gamma": GAMMA,
            "boundary_px": PAPER_BOUNDARY_PX,
            "local_search": 0,
            "skip_diagonal": False,
            "update_order": "positions_first",
            "correction_mode": "incremental",
        }
    return None


def benchmark_path(bench: Path, name: str) -> Path:
    """Return a manifest-relative path with portable separators."""
    return bench / Path(str(name).replace("\\", "/"))


def resolve_device(requested: str) -> str:
    """Resolve a requested CPU/CUDA device and fail clearly when unavailable."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA was requested but is not available")
    return "cuda" if cuda_ok and requested in ("auto", "cuda") else "cpu"


def _gpu_context(device: str) -> Any:
    """Return the shared GPU primitive-patching context when requested."""
    if device != "cuda":
        return nullcontext()
    from distortcorrect_stitcher_gpu import _gpu_backend
    return _gpu_backend()


def _correct_raw_tiles(
    tiles: np.ndarray,
    k1: float,
    device: str,
) -> np.ndarray:
    """Correct raw tiles with the selected CPU or CUDA warp backend."""
    if device == "cuda":
        from distortcorrect_stitcher_gpu import undistort_tiles_gpu
        return undistort_tiles_gpu(tiles, [k1], order=INTERP_ORDER)
    return undistort_tiles(tiles, [k1], order=INTERP_ORDER)


def run_joint_matched(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    pre_stitch_on: bool,
    ls: int,
    k1_tol: float = K1_TOL,
    update_order: str = "positions_first",
    device: str = "cpu",
    gamma: float = GAMMA,
) -> Record:
    """Run the matched iterative joint pipeline."""
    stitcher_cls = (DistortCorrectStitcherGPU if device == "cuda"
                    else DistortCorrectStitcher)
    st = stitcher_cls(
        k1_bounds=K1_BOUNDS, gamma=gamma, n_iterations=N_ITERATIONS,
        k1_tol=k1_tol, interpolation_order=INTERP_ORDER,
        pre_stitch_gamma_schedule=PRE_STITCH if pre_stitch_on else [],
        ncc_threshold=0.0, sharpen=False, boundary_frac=BOUNDARY_FRAC,
        position_damping=POSITION_DAMPING, local_search=ls,
        update_order=update_order,
        correction_mode="pristine_cumulative",
    )
    t0 = time.time()
    res = st.run(tiles, p_ini, do_sharpen=True, verbose=False)
    dt = time.time() - t0
    hist = np.asarray(res.k1_history)
    k1 = float(np.sum(hist[:, 0])) if hist.ndim == 2 else float(np.sum(hist))
    pos_hist = res.position_history
    return {
        "k1": k1,
        "k1_hist": [float(x[0]) for x in hist],
        "position_history": [np.asarray(p).tolist() for p in pos_hist]
        if pos_hist else None,
        "iterations": int(res.iterations_used),
        "positions": np.asarray(res.positions).tolist(),
        "positions_pre": np.asarray(res.positions_pre).tolist(),
        "time_s": dt,
    }


def run_joint_stabilized(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    device: str = "cpu",
) -> Record:
    """Run the exploratory distortion-first stabilized joint variant."""
    return run_joint_matched(
        tiles, p_ini, pre_stitch_on=False, ls=0,
        k1_tol=STABILIZED_JOINT_K1_TOL,
        update_order="distortion_first", device=device, gamma=1.0)


def run_dcs_paper_style(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    device: str = "cpu",
    n_iterations: int = N_ITERATIONS,
    pre_stitch: bool = True,
) -> Record:
    """Run the paper-style joint schedule used in the primary comparison."""
    stitcher_cls = (DistortCorrectStitcherGPU if device == "cuda"
                    else DistortCorrectStitcher)
    st = stitcher_cls(
        k1_bounds=PAPER_K1_BOUNDS, gamma=GAMMA, n_iterations=n_iterations,
        k1_tol=0.0, interpolation_order=INTERP_ORDER,
        pre_stitch_gamma_schedule=None if pre_stitch else [],
        ncc_threshold=0.0, sharpen=False,
        boundary_frac=BOUNDARY_FRAC, boundary_px=PAPER_BOUNDARY_PX,
        position_damping=1.0, local_search=0,
        update_order="positions_first", correction_mode="incremental",
        skip_diagonal=False,
    )
    t0 = time.time()
    res = st.run(tiles, p_ini, do_sharpen=True, verbose=False)
    dt = time.time() - t0
    hist = np.asarray(res.k1_history)
    k1 = float(np.sum(hist[:, 0])) if hist.ndim == 2 else float(np.sum(hist))
    return {
        "k1": k1,
        "k1_hist": [float(x[0]) for x in hist],
        "position_history": [np.asarray(p).tolist() for p in res.position_history]
        if res.position_history else None,
        "iterations": int(res.iterations_used),
        "positions": np.asarray(res.positions).tolist(),
        "positions_pre": np.asarray(res.positions_pre).tolist(),
        "time_s": dt,
    }


def run_sequential_matched(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    pre_stitch_on: bool,
    ls: int,
    device: str = "cpu",
) -> Record:
    """Run the non-paper matched sequential baseline."""
    t0 = time.time()
    if pre_stitch_on:
        tiles_proc = sharpen_tiles(tiles)
        with _gpu_context(device):
            p_pre = pr_stitching(
                tiles_proc, p_ini, gamma_schedule=PRE_STITCH,
                ncc_threshold=0.0, sharpen=False, verbose=False)
    else:
        p_pre = p_ini
    stitcher_cls = (SequentialStitcherGPU if device == "cuda"
                    else SequentialStitcher)
    res = stitcher_cls(
        k1_bounds=K1_BOUNDS, interpolation_order=INTERP_ORDER,
        gamma_schedule=[1.0], ncc_threshold=0.0, sharpen=False,
        boundary_frac=BOUNDARY_FRAC, local_search=ls,
    ).run(tiles, p_pre, do_sharpen=True, verbose=False)
    dt = time.time() - t0
    return {
        "k1": float(res.k1[0]),
        "k1_hist": [float(res.k1[0])],
        "iterations": 1,
        "positions": np.asarray(res.positions).tolist(),
        "positions_pre": np.asarray(p_pre).tolist(),
        "time_s": dt,
    }


def run_sequential_paper_matched(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    device: str = "cpu",
    k1_bounds: tuple[float, float] = PAPER_SEQUENTIAL_K1_BOUNDS,
    pre_stitch: bool = True,
) -> Record:
    """Run the paper-matched one-shot sequential pipeline."""
    t0 = time.time()
    tiles_proc = sharpen_tiles(tiles)
    if pre_stitch:
        with _gpu_context(device):
            p_pre = pr_stitching(
                tiles_proc, p_ini, gamma_schedule=None, ncc_threshold=0.0,
                sharpen=False, skip_diagonal=False, verbose=False)
    else:
        p_pre = np.asarray(p_ini, dtype=int)
    stitcher_cls = (SequentialStitcherGPU if device == "cuda"
                    else SequentialStitcher)
    res = stitcher_cls(
        k1_bounds=k1_bounds, interpolation_order=INTERP_ORDER,
        gamma_schedule=[1.0], ncc_threshold=0.0, sharpen=False,
        boundary_frac=BOUNDARY_FRAC, boundary_px=PAPER_BOUNDARY_PX,
        local_search=0, skip_diagonal=False,
    ).run(tiles_proc, p_pre, do_sharpen=False, verbose=False)
    return {
        "k1": float(res.k1[0]),
        "k1_hist": [float(res.k1[0])],
        "iterations": 1,
        "positions": np.asarray(res.positions).tolist(),
        "positions_pre": np.asarray(p_pre).tolist(),
        "time_s": time.time() - t0,
    }


def run_oracle_k1_paper(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    k1_true: float,
    device: str = "cpu",
) -> Record:
    """Run the sequential diagnostic with the synthetic coefficient supplied."""
    t0 = time.time()
    tiles_proc = sharpen_tiles(tiles)
    with _gpu_context(device):
        p_pre = pr_stitching(
            tiles_proc, p_ini, gamma_schedule=None, ncc_threshold=0.0,
            sharpen=False, skip_diagonal=False, verbose=False)
        tiles_corr = correct_tiles_once(
            tiles_proc, np.array([k1_true]), order=INTERP_ORDER)
        positions = pr_stitching(
            tiles_corr, p_pre, gamma_schedule=[1.0], ncc_threshold=0.0,
            sharpen=False, skip_diagonal=False, verbose=False)
    return {
        "k1": float(k1_true),
        "k1_hist": [float(k1_true)],
        "iterations": 1,
        "positions": np.asarray(positions).tolist(),
        "positions_pre": np.asarray(p_pre).tolist(),
        "time_s": time.time() - t0,
    }


def run_dcs_paper_joint_oracle(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    k1_true: float,
    device: str = "cpu",
) -> Record:
    """Run the joint schedule with the synthetic coefficient supplied."""
    # Diagnostic control for the joint pipeline: same paper-style schedule
    # and 25 feedback iterations, but the per-iteration k1 increment is the
    # exact residual to the injected true coefficient instead of an estimate.
    stitcher_cls = (DistortCorrectStitcherGPU if device == "cuda"
                    else DistortCorrectStitcher)
    st = stitcher_cls(
        k1_bounds=PAPER_K1_BOUNDS, gamma=GAMMA, n_iterations=N_ITERATIONS,
        k1_tol=0.0, interpolation_order=INTERP_ORDER,
        pre_stitch_gamma_schedule=None, ncc_threshold=0.0, sharpen=False,
        boundary_frac=BOUNDARY_FRAC, boundary_px=PAPER_BOUNDARY_PX,
        position_damping=1.0, local_search=0,
        update_order="positions_first", correction_mode="incremental",
        skip_diagonal=False, oracle_k1=k1_true,
    )
    t0 = time.time()
    res = st.run(tiles, p_ini, do_sharpen=True, verbose=False)
    dt = time.time() - t0
    hist = np.asarray(res.k1_history)
    k1 = float(np.sum(hist[:, 0])) if hist.ndim == 2 else float(np.sum(hist))
    return {
        "k1": k1,
        "k1_hist": [float(x[0]) for x in hist],
        "position_history": [np.asarray(p).tolist() for p in res.position_history]
        if res.position_history else None,
        "iterations": int(res.iterations_used),
        "positions": np.asarray(res.positions).tolist(),
        "positions_pre": np.asarray(res.positions_pre).tolist(),
        "time_s": dt,
    }


def _select_multistart_candidate(
    candidates: dict[str, Record],
    scores: dict[str, float],
) -> str:
    """Select the finite candidate with the lowest seam score."""
    if not candidates or set(candidates) != set(scores):
        raise ValueError("candidate and score names must form the same nonempty set")
    finite = [(float(scores[name]), name) for name in candidates
              if np.isfinite(scores[name])]
    if not finite:
        raise RuntimeError("all adaptive multistart seam scores are non-finite")
    _, selected = min(finite, key=lambda item: item[0])
    return selected


def run_adaptive_multistart(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    ls: int,
    device: str = "cpu",
) -> Record:
    """Evaluate stage-first and pre-stitch-first sequential starts."""
    t0 = time.time()
    candidates = {
        "stage_first": run_sequential_matched(
            tiles, p_ini, False, ls, device=device),
        "prestitch_first": run_sequential_matched(
            tiles, p_ini, True, ls, device=device),
    }
    def _score(candidate):
        corrected = _correct_raw_tiles(tiles, candidate["k1"], device)
        return float(seam_error(
            corrected, np.asarray(candidate["positions"], dtype=int),
            border=BORDER))

    scores = {}
    for name, candidate in candidates.items():
        scores[name] = _score(candidate)

    selected = _select_multistart_candidate(candidates, scores)
    result = dict(candidates[selected])
    result["time_s"] = time.time() - t0
    result["selected_start"] = selected
    result["selection_scores"] = scores
    result["candidate_k1"] = {
        name: float(candidate["k1"])
        for name, candidate in candidates.items()
    }
    result["candidate_positions"] = {
        name: candidate["positions"] for name, candidate in candidates.items()
    }
    result["candidate_times_s"] = {
        name: float(candidate["time_s"])
        for name, candidate in candidates.items()
    }
    result["final_selection_score"] = float(scores[selected])
    return result


def run_gt_k1_control(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    k1_true: float,
    ls: int,
    device: str = "cpu",
) -> Record:
    """Correct at the true coefficient before solving positions."""
    # control: correct at the true k1, then stitch
    t0 = time.time()
    tiles_proc = sharpen_tiles(tiles)
    with _gpu_context(device):
        tiles_corr = correct_tiles_once(
            tiles_proc, np.array([k1_true]), order=INTERP_ORDER)
        positions = pr_stitching(tiles_corr, p_ini, gamma_schedule=[1.0],
                                 ncc_threshold=0.0, sharpen=False, verbose=False)
    dt = time.time() - t0
    return {
        "k1": float(k1_true),
        "k1_hist": [float(k1_true)],
        "iterations": 1,
        "positions": np.asarray(positions).tolist(),
        "positions_pre": np.asarray(p_ini).tolist(),
        "time_s": dt,
    }


def run_gt_position_control(
    tiles: np.ndarray,
    p_ini: np.ndarray,
    p_true: np.ndarray,
    ls: int,
    device: str = "cpu",
) -> Record:
    """Estimate distortion at true positions before the final position solve."""
    # control: estimate k1 at the true positions, then correct + stitch
    # from the real initial positions
    t0 = time.time()
    tiles_proc = sharpen_tiles(tiles)
    with _gpu_context(device):
        k1 = estimate_k1_once(
            tiles_proc, np.asarray(p_true, dtype=int),
            k1_bounds=K1_BOUNDS, order=INTERP_ORDER,
            boundary_frac=BOUNDARY_FRAC, local_search=ls)
        tiles_corr = correct_tiles_once(tiles_proc, k1, order=INTERP_ORDER)
        positions = pr_stitching(tiles_corr, p_ini, gamma_schedule=[1.0],
                                 ncc_threshold=0.0, sharpen=False, verbose=False)
    dt = time.time() - t0
    return {
        "k1": float(k1[0]),
        "k1_hist": [float(k1[0])],
        "iterations": 1,
        "positions": np.asarray(positions).tolist(),
        "positions_pre": np.asarray(p_ini).tolist(),
        "time_s": dt,
    }


def run_method(
    method: str,
    tiles: np.ndarray,
    p_ini: np.ndarray,
    p_true: np.ndarray,
    k1_true: float,
    ls: int,
    k1_tol: float,
    update_order: str,
    device: str,
) -> Record:
    """Dispatch one method name to its implementation."""
    if method == "dcs_paper_style":
        return run_dcs_paper_style(tiles, p_ini, device=device)
    if method in PAPER_ITERATION_COUNTS:
        return run_dcs_paper_style(
            tiles, p_ini, device=device,
            n_iterations=PAPER_ITERATION_COUNTS[method])
    if method == "dcs_paper_no_prestitch":
        return run_dcs_paper_style(tiles, p_ini, device=device,
                                   pre_stitch=False)
    if method == "joint_matched":
        return run_joint_matched(
            tiles, p_ini, True, ls, k1_tol, update_order, device=device)
    if method == "sequential_matched":
        return run_sequential_matched(tiles, p_ini, True, ls, device=device)
    if method == "sequential_paper_matched":
        return run_sequential_paper_matched(tiles, p_ini, device=device)
    if method == "sequential_paper_no_prestitch":
        return run_sequential_paper_matched(tiles, p_ini, device=device,
                                            pre_stitch=False)
    if method in PAPER_SEQUENTIAL_SENSITIVITY_BOUNDS:
        return run_sequential_paper_matched(
            tiles, p_ini, device=device,
            k1_bounds=PAPER_SEQUENTIAL_SENSITIVITY_BOUNDS[method])
    if method == "oracle_k1_paper":
        return run_oracle_k1_paper(tiles, p_ini, k1_true, device=device)
    if method == "dcs_paper_joint_oracle":
        return run_dcs_paper_joint_oracle(tiles, p_ini, k1_true, device=device)
    if method == "adaptive_multistart":
        return run_adaptive_multistart(tiles, p_ini, ls, device=device)
    if method == "joint_stabilized":
        return run_joint_stabilized(tiles, p_ini, device=device)
    if method == "joint_matched_no_prestitch":
        return run_joint_matched(
            tiles, p_ini, False, ls, k1_tol, update_order, device=device)
    if method == "sequential_matched_no_prestitch":
        return run_sequential_matched(tiles, p_ini, False, ls, device=device)
    if method == "gt_k1_control":
        return run_gt_k1_control(tiles, p_ini, k1_true, ls, device=device)
    if method == "gt_position_control":
        return run_gt_position_control(
            tiles, p_ini, p_true, ls, device=device)
    raise ValueError(f"unknown method: {method}")


def process_case(
    bench: Path,
    meta: dict[str, Any],
    source_crops: dict[int, np.ndarray],
    out_dir: Path,
    methods: Sequence[str],
    local_search: int = LOCAL_SEARCH,
    joint_k1_tol: float = K1_TOL,
    joint_update_order: str = "positions_first",
    device: str = "auto",
) -> None:
    """Run selected methods, calculate metrics, and write JSON records."""
    device = resolve_device(device)
    case = meta["case"]
    d = np.load(benchmark_path(bench, meta["file"]))
    tiles = d["tiles"]
    p_true = d["positions_true"]
    p_ini = d["positions_ini"]
    k1_true = float(d["true_k1"])
    k2_true = float(d["true_k2"]) if "true_k2" in d.files else 0.0
    noise = int(meta["position_noise_max_px"])
    ls = int(local_search)
    source_crop = source_crops[meta["crop_id"]]

    for method in methods:
        effective_joint_update_order = (
            "distortion_first"
            if method in ("joint_matched_no_prestitch", "joint_stabilized")
            else joint_update_order
        )
        rec = {"case": case, "method": method, "status": "ok",
               "k1_true": k1_true, "noise": noise, "ls": ls,
               "k2_true": k2_true,
               "device": device,
               "protocol_version": PROTOCOL_VERSION,
               "mi_quantization": MI_QUANTIZATION.copy(),
               "warp_backend": (WARP_BACKEND if device == "cuda"
                               else "scipy_spline"),
               "joint_k1_tol": float(joint_k1_tol),
               "joint_update_order": effective_joint_update_order}
        settings = method_settings(method)
        if settings is not None:
            rec["method_settings"] = settings
        out_path = out_dir / f"{case}__{method}.json"
        if out_path.exists():
            try:
                old = json.loads(out_path.read_text("utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"cannot safely inspect existing result {out_path}: {exc}")
            old_cfg = (old.get("protocol_version"), old.get("ls"),
                       old.get("joint_k1_tol"),
                       old.get("joint_update_order"), old.get("device"),
                       old.get("warp_backend"),
                       old.get("method_settings"))
            warp_backend = WARP_BACKEND if device == "cuda" else "scipy_spline"
            new_cfg = (PROTOCOL_VERSION, ls, float(joint_k1_tol),
                       effective_joint_update_order, device, warp_backend,
                       settings)
            if (old.get("status") == "ok" or old.get("protocol_version")) \
                    and old_cfg != new_cfg:
                raise RuntimeError(
                    f"{out_path} uses protocol {old_cfg}, not {new_cfg}; "
                    "use a separate output directory for each protocol")
        try:
            r = run_method(
                method, tiles, p_ini, p_true, k1_true, ls, joint_k1_tol,
                effective_joint_update_order, device)

            pos_rmse, pos_mae, pos_max = pos_stats(
                np.array(r["positions"]), p_true)
            pre_rmse, pre_mae, _ = pos_stats(
                np.array(r["positions_pre"]), p_true)
            if method in PAPER_DCS_METHODS:
                search_bounds, search_gamma = PAPER_K1_BOUNDS, GAMMA
            elif method in PAPER_SEQUENTIAL_METHODS:
                search_bounds = PAPER_SEQUENTIAL_SENSITIVITY_BOUNDS.get(
                    method, PAPER_SEQUENTIAL_K1_BOUNDS)
                search_gamma = 1.0
            elif method in ("gt_k1_control", "oracle_k1_paper",
                            "dcs_paper_joint_oracle"):
                search_bounds, search_gamma = None, None
            elif method in ("joint_matched", "joint_matched_no_prestitch"):
                search_bounds, search_gamma = K1_BOUNDS, GAMMA
            else:
                search_bounds, search_gamma = K1_BOUNDS, 1.0

            rec.update({
                "k1": r["k1"],
                "k1_err": abs(r["k1"] - k1_true),
                "k1_signed": r["k1"] - k1_true,
                "k1_hist": r["k1_hist"],
                "position_history": r.get("position_history"),
                "iterations": r["iterations"],
                "positions": r["positions"],
                "positions_pre": r["positions_pre"],
                "selected_start": r.get("selected_start"),
                "selection_scores": r.get("selection_scores"),
                "candidate_k1": r.get("candidate_k1"),
                "candidate_positions": r.get("candidate_positions"),
                "candidate_times_s": r.get("candidate_times_s"),
                "final_selection_score": r.get("final_selection_score"),
                "pos_rmse": pos_rmse,
                "pos_mae": pos_mae,
                "pos_max": pos_max,
                "pre_rmse": pre_rmse,
                "pre_mae": pre_mae,
                "at_bound": (False if search_bounds is None else at_bound(
                    r["k1_hist"], search_bounds, search_gamma)),
                "time_s": round(r["time_s"], 2),
            })
            corr = _correct_raw_tiles(tiles, r["k1"], device)
            distortion_metrics = metrics_vs_source(
                corr, p_true, source_crop, border=BORDER)
            rec.update({f"distortion_{k}": v
                        for k, v in distortion_metrics.items()})

            p_hat = np.asarray(r["positions"], dtype=float)
            p_eval = p_hat - p_hat[0] + np.asarray(p_true, dtype=float)[0]
            mosaic_metrics = metrics_vs_source(
                corr, p_eval, source_crop, border=BORDER,
                evaluation_positions=p_true)
            rec.update(mosaic_metrics)
            rec["seam_true"] = seam_error(corr, p_true, border=BORDER)
            rec["seam_est"] = seam_error(
                corr, np.round(p_eval).astype(int),
                border=BORDER)
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        print(f"{case} {method}: {rec['status']}", flush=True)


def main() -> None:
    """Parse CLI options and process the selected benchmark cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=str, default="all",
                        help="comma-separated case names or 'all'")
    parser.add_argument("--methods", type=str, default=",".join(PRIMARY_METHODS))
    parser.add_argument(
        "--local-search", type=int, default=LOCAL_SEARCH,
        help="fixed NCC translation-search radius; 0 is the faithful primary")
    parser.add_argument(
        "--joint-k1-tol", type=float, default=K1_TOL,
        help="0 runs the faithful fixed iteration budget")
    parser.add_argument(
        "--joint-update-order", choices=("positions_first", "distortion_first"),
        default="positions_first")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()

    manifest = json.loads((args.bench / "manifest.json").read_text("utf-8"))
    source_crops = {}
    for c in manifest["crops"]:
        source_crops[c["id"]] = np.asarray(
            tifffile.imread(benchmark_path(args.bench, c["file"]))
        )

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods:
        raise ValueError("at least one method is required")
    if len(methods) != len(set(methods)):
        raise ValueError("method list contains duplicates")
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {', '.join(unknown_methods)}")
    if args.cases == "all":
        cases = [m for m in manifest["cases"]]
    else:
        wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
        if not wanted:
            raise ValueError("case list is empty")
        cases = [m for m in manifest["cases"] if m["case"] in wanted]
        found = {m["case"] for m in cases}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"unknown cases: {', '.join(missing)}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for meta in cases:
        process_case(
            args.bench, meta, source_crops, out_dir, methods,
            local_search=args.local_search,
            joint_k1_tol=args.joint_k1_tol,
            joint_update_order=args.joint_update_order,
            device=args.device)


if __name__ == "__main__":
    main()
