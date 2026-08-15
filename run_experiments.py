# Resumable batch runner for the final benchmark.
# These limits also apply when worker processes import this module.
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

_G = {}


def check_manifest(bench, manifest):
    cases = manifest.get("cases", [])
    crops = manifest.get("crops", [])
    if manifest.get("case_count") != len(cases):
        raise ValueError("manifest case_count does not match its case list")
    case_names = [case.get("case") for case in cases]
    if None in case_names or len(case_names) != len(set(case_names)):
        raise ValueError("manifest case names are missing or duplicated")
    crop_ids = [crop.get("id") for crop in crops]
    if None in crop_ids or len(crop_ids) != len(set(crop_ids)):
        raise ValueError("manifest crop ids are missing or duplicated")
    if any(case.get("crop_id") not in crop_ids for case in cases):
        raise ValueError("a case refers to an unknown crop")

    root = bench.resolve()
    files = [crop.get("file") for crop in crops]
    files += [case.get("file") for case in cases]
    for name in files:
        if not name:
            raise ValueError("manifest contains an empty file name")
        path = (bench / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"manifest path escapes the benchmark: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)


def joint_order(method, requested_order):
    if method in ("joint_matched_no_prestitch", "joint_stabilized"):
        return "distortion_first"
    return requested_order


def run_config(protocol_version, local_search, joint_k1_tol,
               joint_update_order, device, warp_backend, mi_quantization):
    """Return the provenance fields that must match before a result is reused."""
    return (
        protocol_version, int(local_search), float(joint_k1_tol),
        joint_update_order, device, warp_backend,
        json.dumps(mi_quantization, sort_keys=True),
    )


def _init(bench, out, local_search, joint_k1_tol, joint_update_order, device):
    import run_final_experiment as rfe

    bench = Path(bench)
    manifest = json.loads((bench / "manifest.json").read_text("utf-8"))
    _G["rfe"] = rfe
    _G["bench"] = bench
    _G["out"] = Path(out)
    _G["local_search"] = int(local_search)
    _G["joint_k1_tol"] = float(joint_k1_tol)
    _G["joint_update_order"] = joint_update_order
    _G["device"] = device
    _G["cases"] = {c["case"]: c for c in manifest["cases"]}
    _G["crops"] = {
        c["id"]: np.asarray(tifffile.imread(bench / c["file"]))
        for c in manifest["crops"]
    }


def _task(task):
    # Record a corrupt case without losing the rest of the batch. Failed
    # records are retried on the next run.
    case, method = task
    effective_order = joint_order(
        method, _G.get("joint_update_order"))
    try:
        _G["rfe"].process_case(
            _G["bench"], _G["cases"][case], _G["crops"], _G["out"],
            [method], local_search=_G["local_search"],
            joint_k1_tol=_G["joint_k1_tol"],
            joint_update_order=_G["joint_update_order"],
            device=_G["device"])
    except Exception as exc:
        rec = {"case": case, "method": method, "status": "failed",
               "error": f"{type(exc).__name__}: {exc}",
               "protocol_version": _G["rfe"].PROTOCOL_VERSION,
               "mi_quantization": _G["rfe"].MI_QUANTIZATION.copy(),
               "warp_backend": (_G["rfe"].WARP_BACKEND
                                if _G.get("device") == "cuda"
                                else "scipy_spline"),
               "ls": _G.get("local_search"),
               "joint_k1_tol": _G.get("joint_k1_tol"),
               "joint_update_order": effective_order,
               "device": _G.get("device")}
        try:
            (_G["out"] / f"{case}__{method}.json").write_text(
                json.dumps(rec, indent=1), encoding="utf-8")
        except Exception:
            pass
    return case, method


def main():
    import run_final_experiment as rfe

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--methods", type=str,
                    default=",".join(rfe.PRIMARY_METHODS))
    ap.add_argument("--cases", type=str, default="all")
    ap.add_argument("--local-search", type=int, default=0)
    ap.add_argument("--joint-k1-tol", type=float, default=rfe.K1_TOL)
    ap.add_argument("--joint-update-order",
                    choices=("positions_first", "distortion_first"),
                    default="positions_first")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"),
                    default="auto",
                    help="GPU uses one worker because CUDA backend patching is process-global")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs and list task counts without running methods")
    args = ap.parse_args()

    args.device = rfe.resolve_device(args.device)
    if args.device == "cuda" and args.workers != 1:
        raise ValueError("CUDA execution requires --workers 1; the GPU backend "
                         "uses process-global primitive dispatch")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bench / "manifest.json").read_text("utf-8"))
    check_manifest(args.bench, manifest)
    if args.cases == "all":
        metas = manifest["cases"]
    else:
        wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
        if not wanted:
            raise ValueError("case list is empty")
        metas = [c for c in manifest["cases"] if c["case"] in wanted]
        found = {c["case"] for c in metas}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"unknown cases: {', '.join(missing)}")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("at least one method is required")
    if len(methods) != len(set(methods)):
        raise ValueError("method list contains duplicates")
    unknown_methods = sorted(set(methods) - set(rfe.METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {', '.join(unknown_methods)}")

    tasks = []
    for meta in metas:
        for method in methods:
            effective_order = joint_order(
                method, args.joint_update_order)
            jp = args.out / f"{meta['case']}__{method}.json"
            if jp.exists():
                try:
                    old = json.loads(jp.read_text("utf-8"))
                    if old.get("status") == "ok":
                        old_cfg = run_config(
                            old.get("protocol_version"), old.get("ls", -1),
                            old.get("joint_k1_tol", float("nan")),
                            old.get("joint_update_order"), old.get("device"),
                            old.get("warp_backend"), old.get("mi_quantization"),
                        )
                        warp_backend = (rfe.WARP_BACKEND
                                        if args.device == "cuda"
                                        else "scipy_spline")
                        new_cfg = run_config(
                            rfe.PROTOCOL_VERSION, args.local_search,
                            args.joint_k1_tol, effective_order, args.device,
                            warp_backend, rfe.MI_QUANTIZATION)
                        if old_cfg != new_cfg:
                            raise RuntimeError(
                                f"{jp} was produced by protocol {old_cfg}, "
                                f"not {new_cfg}; use a separate --out directory "
                                "for each ablation")
                        continue
                except Exception:
                    raise
            tasks.append((meta["case"], method))

    total_all = len(metas) * len(methods)
    print(f"{len(metas)} cases x {len(methods)} methods = {total_all} tasks; "
          f"{len(tasks)} to run, {total_all - len(tasks)} already done "
          f"({args.workers} workers)", flush=True)
    if args.dry_run:
        print("dry run complete", flush=True)
        return
    if not tasks:
        print("nothing to do", flush=True)
        return

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.bench), str(args.out),
                                       args.local_search, args.joint_k1_tol,
                                       args.joint_update_order, args.device)) as ex:
        futs = [ex.submit(_task, t) for t in tasks]
        for fut in as_completed(futs):
            case, method = fut.result()
            done += 1
            el = time.time() - t0
            eta = el / done * (len(tasks) - done)
            print(f"[{done}/{len(tasks)}] {case} {method}  "
                  f"elapsed {el/60:.1f}m  ETA {eta/60:.1f}m", flush=True)
    print(f"all done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
