# Reproduce the four ablation runs from the completed result directories.
#
#   python run_ablations.py --bench benchmark --device cuda
#
# Each ablation writes into its own results/ablation_* directory and is
# resumable through run_experiments.py.  The search and oracle ablations
# run only on the 24 nonzero-distortion cases.

import argparse
import json
import subprocess
import sys
from pathlib import Path

FEEDBACK_METHODS = [
    "dcs_paper_iter_1", "dcs_paper_iter_2",
    "dcs_paper_iter_5", "dcs_paper_iter_10",
]
PRESTITCH_METHODS = [
    "dcs_paper_no_prestitch", "sequential_paper_no_prestitch",
]
SEARCH_METHODS = [
    "sequential_paper_bound_010", "sequential_paper_bound_020",
]
ORACLE_METHODS = ["oracle_k1_paper"]


def distorted_cases(manifest):
    return ",".join(c["case"] for c in manifest["cases"]
                    if abs(c["k1_true"]) > 1e-12)


def all_cases(manifest):
    return ",".join(c["case"] for c in manifest["cases"])


def run(python, bench, out, methods, cases, device):
    cmd = [python, "-B", "run_experiments.py", "--bench", str(bench),
           "--out", str(out), "--methods", ",".join(methods),
           "--cases", cases, "--device", device, "--workers", "1"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("benchmark"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    manifest = json.loads((args.bench / "manifest.json").read_text("utf-8"))
    all_c = all_cases(manifest)
    dist_c = distorted_cases(manifest)

    run(args.python, args.bench, Path("results/ablation_feedback"),
        FEEDBACK_METHODS, all_c, args.device)
    run(args.python, args.bench, Path("results/ablation_prestitch"),
        PRESTITCH_METHODS, all_c, args.device)
    run(args.python, args.bench, Path("results/ablation_search"),
        SEARCH_METHODS, dist_c, args.device)
    run(args.python, args.bench, Path("results/ablation_oracle"),
        ORACLE_METHODS, dist_c, args.device)
    print("all ablations complete", flush=True)


if __name__ == "__main__":
    main()
