# Generate the paper figures from the completed result directories.
#
#   python make_figures.py --bench benchmark --out figures
#
# Produces five PNGs: the primary paired effect, the feedback dose, the
# pre-stitch interaction, the accuracy-runtime tradeoff, and the signed-k1
# calibration.

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_results import bootstrap_ci, load_all, paired_deltas

COLORS = {
    "joint": "#1f77b4",
    "joint_np": "#7fb3d5",
    "sequential": "#d62728",
    "sequential_np": "#f5a0a0",
    "oracle": "#2ca02c",
}
LABELS = {
    "joint": "Joint (paper-style)",
    "joint_np": "Joint, no pre-stitch",
    "sequential": "Sequential (paper-matched)",
    "sequential_np": "Sequential, no pre-stitch",
    "oracle": "Oracle k1",
}
BOOT_SEED = 0


def med(rows, metric):
    vals = [r.get(metric) for r in rows if r["status"] == "ok"]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def paired_by_case(rows_a, rows_b, metric):
    out = {}
    for a in rows_a:
        if a["status"] != "ok":
            continue
        match = [b for b in rows_b
                 if b["case"] == a["case"] and b["status"] == "ok"]
        if not match:
            continue
        b = match[0]
        av, bv = a.get(metric), b.get(metric)
        if av is None or bv is None or not (np.isfinite(av) and np.isfinite(bv)):
            continue
        out[a["case"]] = (av, bv)
    return out


def paired_ci(rows_a, rows_b, method_a, method_b, metric):
    # delta = rows_a - rows_b per case; negative favours rows_a for errors.
    deltas = paired_deltas(list(rows_a) + list(rows_b), method_a, method_b)
    rng = np.random.default_rng(BOOT_SEED)
    return bootstrap_ci(deltas, metric, rng)


def fig1_primary(rows_joint, rows_seq, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric, ylab in (
            (axes[0], "pos_rmse", "Position RMSE (px)"),
            (axes[1], "k1_err", "Absolute $k_1$ error")):
        pairs = paired_by_case(rows_joint, rows_seq, metric)
        cases = list(pairs)
        j = np.array([pairs[c][0] for c in cases])
        s = np.array([pairs[c][1] for c in cases])
        x = np.array([0.0, 1.0])
        for k, c in enumerate(cases):
            ax.plot(x, [j[k], s[k]], color="0.75", lw=0.6, zorder=1)
        rng = np.random.default_rng(0)
        jx = rng.normal(0, 0.04, len(cases))
        sx = rng.normal(1, 0.04, len(cases))
        ax.scatter(jx, j, s=14, color=COLORS["joint"], zorder=2, alpha=0.8)
        ax.scatter(sx, s, s=14, color=COLORS["sequential"], zorder=2, alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LABELS["joint"], LABELS["sequential"]],
                           fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.4, 1.4)
        med_j, med_s = med(rows_joint, metric), med(rows_seq, metric)
        ax.axhline(med_j, color=COLORS["joint"], lw=1.2, ls="--", alpha=0.7)
        ax.axhline(med_s, color=COLORS["sequential"], lw=1.2, ls="--",
                   alpha=0.7)
        delta, lo, hi = paired_ci(
            rows_joint, rows_seq, "dcs_paper_style",
            "sequential_paper_matched", metric)
        ax.text(0.02, 0.98,
                "median delta = %.3f\n[%.3f, %.3f]" % (delta, lo, hi),
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", lw=0.6))
    fig.tight_layout()
    fig.savefig(out / "fig1_primary_effect.png", dpi=300)
    plt.close(fig)


def fig2_feedback_dose(feedback, rows_joint, out):
    iters = [1, 2, 5, 10, 25]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, ylab in (
            (axes[0], "pos_rmse", "Position RMSE (px)"),
            (axes[1], "k1_err", "Absolute $k_1$ error"),
            (axes[2], "time_s", "Runtime (s)")):
        by_case = {}
        for it, rows in zip(iters, feedback):
            for r in rows:
                if r["status"] != "ok":
                    continue
                by_case.setdefault(r["case"], {})[it] = r.get(metric)
        for case, vals in by_case.items():
            xs, ys = [], []
            for it in iters:
                if it in vals and vals[it] is not None \
                        and np.isfinite(vals[it]):
                    xs.append(it)
                    ys.append(vals[it])
            if len(xs) >= 2:
                ax.plot(xs, ys, color="0.8", lw=0.5, alpha=0.6, zorder=1)
        meds = [med(rows, metric) for rows in feedback]
        meds.append(med(rows_joint, metric))
        ax.plot(iters, meds, marker="o", color="k", lw=1.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(iters)
        ax.set_xticklabels([str(i) for i in iters], fontsize=8)
        ax.set_xlabel("Iterations")
        ax.set_ylabel(ylab)
        for x, y in zip(iters, meds):
            ax.annotate("%.2f" % y, (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig2_feedback_dose.png", dpi=300)
    plt.close(fig)


def distorted(rows):
    return [r for r in rows if abs(r["k1_true"]) > 1e-12]


def fig3_prestitch(rows_joint, rows_joint_np, rows_seq, rows_seq_np,
                   rows_oracle, out):
    groups = [
        ("joint", distorted(rows_joint)),
        ("joint_np", distorted(rows_joint_np)),
        ("sequential", distorted(rows_seq)),
        ("sequential_np", distorted(rows_seq_np)),
        ("oracle", distorted(rows_oracle)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric, ylab in (
            (axes[0], "pos_rmse", "Position RMSE (px)"),
            (axes[1], "k1_err", "Absolute $k_1$ error")):
        names = [g[0] for g in groups]
        for idx, (name, rows) in enumerate(groups):
            vals = [r.get(metric) for r in rows if r["status"] == "ok"]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            rng = np.random.default_rng(idx)
            ax.scatter(rng.normal(idx, 0.06, len(vals)), vals, s=10,
                       color=COLORS[name], alpha=0.55, zorder=2)
            ax.plot(idx, np.median(vals), marker="o", color="k", zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABELS[n] for n in names], rotation=15,
                           ha="right", fontsize=7.5)
        ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(out / "fig3_prestitch_interaction.png", dpi=300)
    plt.close(fig)


def fig4_accuracy_runtime(groups, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for name, rows in groups:
        rmse = [r.get("pos_rmse") for r in rows if r["status"] == "ok"]
        tm = [r.get("time_s") for r in rows if r["status"] == "ok"]
        rmse = [v for v in rmse if v is not None and np.isfinite(v)]
        tm = [v for v in tm if v is not None and np.isfinite(v)]
        ax.scatter(np.median(tm), np.median(rmse), s=70, color=COLORS[name],
                   zorder=3, edgecolor="k", lw=0.5)
        ax.annotate(LABELS[name], (np.median(tm), np.median(rmse)),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Median runtime (s)")
    ax.set_ylabel("Median position RMSE (px)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "fig4_accuracy_runtime.png", dpi=300)
    plt.close(fig)


def fig5_signed_k1(rows_joint, rows_seq, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for rows, name in ((rows_joint, "joint"), (rows_seq, "sequential")):
        est = [r.get("k1") for r in rows if r["status"] == "ok"]
        tru = [r.get("k1_true") for r in rows if r["status"] == "ok"]
        est = [v for v in est if v is not None and np.isfinite(v)]
        tru = [v for v in tru if v is not None and np.isfinite(v)]
        ax.scatter(tru, est, s=16, color=COLORS[name], alpha=0.7,
                   label=LABELS[name])
    lim = [-0.012, 0.012]
    ax.plot(lim, lim, color="k", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True $k_1$")
    ax.set_ylabel("Estimated $k_1$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "fig5_signed_k1.png", dpi=300)
    plt.close(fig)


def method_rows(results_dir, manifest, method):
    return [r for r in load_all(results_dir, manifest) if r["method"] == method]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.bench / "manifest.json").read_text("utf-8"))

    joint = method_rows(Path("results/distortcorrect"), manifest,
                        "dcs_paper_style")
    seq = method_rows(Path("results/sequential"), manifest,
                      "sequential_paper_matched")
    feedback = [
        method_rows(Path("results/ablation_feedback"), manifest,
                    f"dcs_paper_iter_{it}")
        for it in (1, 2, 5, 10)
    ]
    joint_np = method_rows(Path("results/ablation_prestitch"), manifest,
                           "dcs_paper_no_prestitch")
    seq_np = method_rows(Path("results/ablation_prestitch"), manifest,
                         "sequential_paper_no_prestitch")
    oracle_rows = method_rows(Path("results/ablation_oracle"), manifest,
                              "oracle_k1_paper")

    fig1_primary(joint, seq, args.out)
    fig2_feedback_dose(feedback, joint, args.out)
    fig3_prestitch(joint, joint_np, seq, seq_np, oracle_rows, args.out)
    fig4_accuracy_runtime([
        ("joint", distorted(joint)),
        ("joint_np", distorted(joint_np)),
        ("sequential", distorted(seq)),
        ("sequential_np", distorted(seq_np)),
        ("oracle", distorted(oracle_rows)),
    ], args.out)
    fig5_signed_k1(joint, seq, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
