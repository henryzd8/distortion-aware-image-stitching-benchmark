"""Launch the completed standard ablation runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
JOINT_ORACLE_METHODS = ["dcs_paper_joint_oracle"]


def distorted_cases(manifest: dict[str, Any]) -> str:
    """Return the comma-separated nonzero-distortion case names."""
    return ",".join(
        c["case"] for c in manifest["cases"]
        if abs(c["k1_true"]) > 1e-12
    )


def all_cases(manifest: dict[str, Any]) -> str:
    """Return all manifest case names as a runner argument."""
    return ",".join(c["case"] for c in manifest["cases"])


def run(
    python: str,
    bench: Path,
    out: Path,
    methods: list[str],
    cases: str,
    device: str,
) -> None:
    """Run one resumable ablation group and propagate failures."""
    cmd = [python, "-B", "run_experiments.py", "--bench", str(bench),
           "--out", str(out), "--methods", ",".join(methods),
           "--cases", cases, "--device", device, "--workers", "1"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    """Parse options and run each completed ablation group."""
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
    run(args.python, args.bench, Path("results/ablation_joint_oracle"),
        JOINT_ORACLE_METHODS, dist_c, args.device)
    print("all ablations complete", flush=True)


if __name__ == "__main__":
    main()
