# Summarize completed benchmark results and paired method differences.

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from run_final_experiment import METHODS, PRIMARY_METHODS

ERROR_METRICS = ("k1_err", "k1_signed", "pos_rmse", "pos_mae", "pos_max",
                 "seam_true", "seam_est")
QUALITY_METRICS = ("psnr", "ssim", "ncc",
                   "mse",
                   "coverage_fraction", "distortion_psnr",
                   "distortion_ssim", "distortion_ncc",
                   "distortion_mse",
                   "distortion_coverage_fraction")
ALL_METRICS = ERROR_METRICS + QUALITY_METRICS + ("time_s", "iterations")
EXTRA_COLUMNS = ("pre_rmse", "pre_mae", "at_bound")
COMPARISONS = {
    "joint-matched - sequential": ("joint_matched", "sequential_matched"),
    "paper-style - sequential": ("dcs_paper_style", "sequential_matched"),
    "paper-style - paper-matched sequential": (
        "dcs_paper_style", "sequential_paper_matched"),
    "joint-matched - paper-style": ("joint_matched", "dcs_paper_style"),
    "adaptive-multistart - sequential": (
        "adaptive_multistart", "sequential_matched"),
    "adaptive-multistart - paper-style": (
        "adaptive_multistart", "dcs_paper_style"),
    "stabilized-joint - adaptive-multistart": (
        "joint_stabilized", "adaptive_multistart"),
    "stabilized-joint - sequential(no-pre)": (
        "joint_stabilized", "sequential_matched_no_prestitch"),
    "joint(no-pre) - sequential(no-pre)": (
        "joint_matched_no_prestitch", "sequential_matched_no_prestitch"),
}

N_BOOT = 10000
SEED = 0


def load_all(results_dir, manifest):
    cases = {}
    for meta in manifest["cases"]:
        cases[meta["case"]] = meta
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        rec = json.loads(path.read_text("utf-8"))
        case = rec["case"]
        if case not in cases:
            raise RuntimeError(f"{path.name} belongs to a different benchmark")
        rec.update({"crop_id": cases[case]["crop_id"],
                    "seed": cases[case]["seed"],
                    "k1_true": cases[case]["k1_true"],
                    "k2_true": cases[case].get("k2_true", 0.0),
                    "noise": cases[case]["position_noise_max_px"]})
        rows.append(rec)
    return rows


def check_results(rows):
    protocols = {r.get("protocol_version") for r in rows}
    quantizers = {json.dumps(r.get("mi_quantization"), sort_keys=True)
                  for r in rows}
    warp_backends = {r.get("warp_backend") for r in rows}
    if len(protocols) > 1:
        raise RuntimeError("results directory mixes protocol versions: "
                           f"{sorted(protocols, key=str)}")
    if len(quantizers) > 1:
        raise RuntimeError("results directory mixes MI quantization settings")
    if len(warp_backends) > 1:
        raise RuntimeError("results directory mixes warp backends")
    if None in protocols or "null" in quantizers or None in warp_backends:
        raise RuntimeError("one or more results are missing provenance fields")


def check_complete(rows, manifest, methods, allow_incomplete=False):
    expected = {(c["case"], m) for c in manifest["cases"] for m in methods}
    found = [(r["case"], r["method"]) for r in rows]
    duplicates = sorted(pair for pair, count in Counter(found).items()
                        if count > 1)
    if duplicates:
        raise RuntimeError(f"duplicate case-method results: {duplicates[:3]}")
    extra = sorted(set(found) - expected)
    if extra:
        raise RuntimeError(
            "results directory contains methods outside this analysis: "
            f"{extra[:3]}"
        )
    missing = sorted(expected - set(found))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"results are incomplete: {len(missing)} case-method files missing; "
            "use --allow-incomplete only for progress checks"
        )
    return missing


def paired_deltas(rows, method_a, method_b):
    by_case = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        if r["method"] not in (method_a, method_b):
            continue
        by_case.setdefault(r["case"], {})[r["method"]] = r
    deltas = {}
    for case, recs in by_case.items():
        if method_a not in recs or method_b not in recs:
            continue
        method_a_record = recs[method_a]
        method_b_record = recs[method_b]
        delta = {}
        for m in ALL_METRICS:
            a_value = method_a_record.get(m)
            b_value = method_b_record.get(m)
            if (a_value is not None and b_value is not None
                    and np.isfinite(a_value) and np.isfinite(b_value)):
                delta[m] = a_value - b_value
        if delta:
            delta["case"] = case
            delta["crop_id"] = method_a_record["crop_id"]
            delta["seed"] = method_a_record["seed"]
            delta["noise"] = method_a_record["noise"]
            delta["k1_true"] = method_a_record["k1_true"]
            deltas[case] = delta
    return deltas


def bootstrap_ci(deltas, metric, rng, n_boot=N_BOOT):
    # Content-block bootstrap: simulated stage-noise seeds are repeated
    # measurements nested within a crop, not independent microscopy content.
    # deltas; cases with a NaN delta for this metric are dropped from it
    cases = [c for c in deltas
             if metric in deltas[c] and np.isfinite(deltas[c][metric])]
    if not cases:
        return (float("nan"), float("nan"), float("nan"))
    vals = np.array([deltas[c][metric] for c in cases])
    med = float(np.median(vals))
    block_of = {c: deltas[c]["crop_id"] for c in cases}
    blocks = sorted(set(block_of.values()))
    by_block = {b: [c for c in cases if block_of[c] == b] for b in blocks}
    if len(blocks) < 2:
        return (med, float("nan"), float("nan"))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = []
        for idx in rng.integers(0, len(blocks), size=len(blocks)):
            picked.extend(by_block[blocks[idx]])
        boot[b] = np.nanmedian([deltas[c].get(metric, np.nan) for c in picked])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return (med, float(lo), float(hi))


def failure_counts(rows, methods):
    out = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        out[m] = (sum(1 for r in rs if r["status"] != "ok"), len(rs))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path,
                        help="additional result directory for paired reference arms")
    parser.add_argument("--out", type=Path,
                        help="directory for summaries and figures; defaults to --results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--boot", type=int, default=N_BOOT)
    parser.add_argument("--methods", default=",".join(PRIMARY_METHODS))
    parser.add_argument("--cases", default="all",
                        help="comma-separated case names for a preflight subset")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.results
    args.out.mkdir(parents=True, exist_ok=True)

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods:
        raise ValueError("at least one method is required")
    if len(methods) != len(set(methods)):
        raise ValueError("method list contains duplicates")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")

    manifest = json.loads((args.bench / "manifest.json").read_text("utf-8"))
    rows = load_all(args.results, manifest)
    if args.reference_results is not None:
        rows.extend(r for r in load_all(args.reference_results, manifest)
                    if r["method"] in methods)
    if args.cases != "all":
        wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
        if not wanted:
            raise ValueError("case list is empty")
        known = {case["case"] for case in manifest["cases"]}
        unknown_cases = sorted(wanted - known)
        if unknown_cases:
            raise ValueError(f"unknown cases: {', '.join(unknown_cases)}")
        manifest = dict(manifest)
        manifest["cases"] = [case for case in manifest["cases"]
                             if case["case"] in wanted]
        manifest["case_count"] = len(manifest["cases"])
        rows = [row for row in rows if row["case"] in wanted]
    missing = check_complete(
        rows, manifest, methods, allow_incomplete=args.allow_incomplete)
    check_results(rows)
    rng = np.random.default_rng(SEED)

    with (args.out / "results_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "crop_id", "seed", "k1_true", "k2_true", "noise",
                         "method", "status", "protocol_version", "mi_quantization",
                         "warp_backend"]
                        + list(ALL_METRICS)
                        + list(EXTRA_COLUMNS))
        for r in sorted(rows, key=lambda x: x["case"]):
            writer.writerow([r["case"], r["crop_id"], r["seed"],
                             r["k1_true"], r["k2_true"], r["noise"], r["method"],
                             r["status"], r.get("protocol_version"),
                             json.dumps(r.get("mi_quantization"), sort_keys=True),
                             r.get("warp_backend")] +
                            [r.get(m) for m in ALL_METRICS] +
                            [r.get(m) for m in EXTRA_COLUMNS])

    if missing:
        print(f"warning: {len(missing)} planned results are missing")
    for m, (bad, tot) in failure_counts(rows, methods).items():
        print(f"{m}: {bad}/{tot} failed")

    k2_levels = sorted({r["k2_true"] for r in rows}) or [0.0]
    k1_levels = sorted({r["k1_true"] for r in rows})
    noise_levels = sorted({r["noise"] for r in rows})
    comparisons = {
        label: pair for label, pair in COMPARISONS.items()
        if set(pair).issubset(methods)
    }
    paired_rows = []
    for label, (method_a, method_b) in comparisons.items():
        for k2 in k2_levels:
            for k1 in k1_levels:
                for noise in noise_levels:
                    stratum = [r for r in rows if r["k2_true"] == k2
                               and r["k1_true"] == k1
                               and r["noise"] == noise]
                    deltas = paired_deltas(stratum, method_a, method_b)
                    if not deltas:
                        continue
                    n_cases = len(deltas)
                    key_stats = {}
                    for metric in ALL_METRICS:
                        med, lo, hi = bootstrap_ci(
                            deltas, metric, rng, args.boot)
                        paired_rows.append(
                            [label, k1, k2, noise, metric, n_cases,
                             med, lo, hi])
                        if metric in ("pos_rmse", "k1_err"):
                            key_stats[metric] = (med, lo, hi)
                    print(f"{label}, k1={k1:+.4f}, noise={noise:2d}, "
                          f"n={n_cases}")
                    for metric in ("pos_rmse", "k1_err"):
                        med, lo, hi = key_stats[metric]
                        print(f"  {metric}: {med:+.4f} "
                              f"[{lo:+.4f}, {hi:+.4f}]")

    with (args.out / "paired_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "k1_true", "k2_true", "noise",
                         "metric", "n", "median", "ci_low", "ci_high"])
        writer.writerows(paired_rows)

    condition_rows = []
    for method in methods:
        for k1 in sorted({r["k1_true"] for r in rows}):
            for k2 in sorted({r["k2_true"] for r in rows}):
                for noise in sorted({r["noise"] for r in rows}):
                    rs = [r for r in rows
                          if r["method"] == method and r["k1_true"] == k1
                          and r["k2_true"] == k2 and r["noise"] == noise
                          and r["status"] == "ok"]
                    if rs:
                        ke = float(np.nanmedian([r["k1_err"] for r in rs]))
                        pe = float(np.nanmedian([r["pos_rmse"] for r in rs]))
                        condition_rows.append(
                            [method, k1, k2, noise, len(rs), ke, pe])

    with (args.out / "condition_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "k1_true", "k2_true", "noise", "n",
                         "median_k1_error", "median_position_rmse"])
        writer.writerows(condition_rows)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots")
            return
        plot_pairs = list(comparisons.items())[:2]
        if not plot_pairs:
            print("no requested method pair is available; skipping plots")
            return
        fig, axes = plt.subplots(2, len(plot_pairs), figsize=(11, 7),
                                 squeeze=False)
        for col, (label, pair) in enumerate(plot_pairs):
            deltas = paired_deltas(
                [r for r in rows if abs(r["k2_true"]) < 1e-15], *pair)
            noises = sorted({deltas[c]["noise"] for c in deltas})
            for row, metric in enumerate(("pos_rmse", "k1_err")):
                meds, los, his = [], [], []
                for noise in noises:
                    subset = {case: deltas[case] for case in deltas
                              if deltas[case]["noise"] == noise}
                    med, lo, hi = bootstrap_ci(
                        subset, metric, np.random.default_rng(SEED), args.boot)
                    meds.append(med)
                    los.append(lo)
                    his.append(hi)
                ax = axes[row, col]
                yerr = [np.abs(np.array(meds) - np.array(los)),
                        np.abs(np.array(his) - np.array(meds))]
                ax.errorbar(noises, meds, yerr=yerr, marker="o", capsize=3)
                ax.axhline(0, color="k", lw=0.8, ls=":")
                ax.set_title(label)
                ax.set_xlabel("position noise a (px)")
                ax.set_ylabel(f"delta {metric}")
        fig.tight_layout()
        fig.savefig(args.out / "primary_effects.png", dpi=150)
        print("saved primary_effects.png")


if __name__ == "__main__":
    main()
