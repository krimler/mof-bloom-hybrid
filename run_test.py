#!/usr/bin/env python3

"""
run_test.py — Unified runner for MOF experiments

This script provides a single entry point to run the common experiments in this repo
without worrying about PYTHONPATH intricacies. It dispatches to the scripts under
`experiments/` using subprocess, passing through the relevant flags.

Supported modes:
  - nd         : Experiment 1 (synthetic relativity demo)
  - mpeg_mof   : MPEG-7 MOF–Bloom prefilter only
  - mpeg_hybrid: MPEG-7 Hybrid (MOF–Bloom + Zernike)
  - mpeg_plot  : Plot MPEG-7 bars from a results CSV
  - zernike    : Baseline descriptor (Zernike)
  - efd        : Baseline descriptor (Elliptic Fourier)
  - hu         : Baseline descriptor (Hu moments)
  - learned    : Learned medoid Bloom tree on split
  - exp3       : Corneal AR-MOF analysis (hexagon reference)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "experiments"
TABLES_DIR = REPO_ROOT / "tables"

def run(cmd, env=None):
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def main():
    parser = argparse.ArgumentParser(description="Unified runner for MOF experiments")
    parser.add_argument("--mode", required=True, choices=[
        "nd", "mpeg_mof", "mpeg_hybrid", "mpeg_plot",
        "zernike", "efd", "hu",
        "learned", "exp3"
    ])
    parser.add_argument("--root", type=str, default="data/MPEG7dataset", help="Path to MPEG-7 dataset root")
    parser.add_argument("--csv", type=str, default="data/Human_Corneal_Endothelium.csv", help="Path to corneal CSV for exp3")
    parser.add_argument("--split", type=str, default=None, help="Path to stratified split JSON (learned mode)")
    parser.add_argument("--top-k", type=int, default=60, help="Top-k for Hybrid (MOF–Bloom + Zernike)")
    parser.add_argument("--results", type=str, default=str(TABLES_DIR / "mpeg70_bloom_results_norm.csv"),
                        help="Results CSV for plotting MPEG-7 bars")
    args = parser.parse_args()

    # Ensure PYTHONPATH includes repo root so internal imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    if str(REPO_ROOT) not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = (str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]).strip(os.pathsep)

    if args.mode == "nd":
        run([sys.executable, str(EXP_DIR / "nd_demo.py")], env=env)

    elif args.mode == "mpeg_mof":
        # Bloom prefilter (normalized version)
        run([sys.executable, str(EXP_DIR / "mpeg70_bloom_prefilter_norm.py"),
             "--root", args.root, "--top-k", str(args.top_k)], env=env)

    elif args.mode == "mpeg_hybrid":
        # Same entry as normalized prefilter; Hybrid re-rank handled in script
        run([sys.executable, str(EXP_DIR / "mpeg70_bloom_prefilter_norm.py"),
             "--root", args.root, "--top-k", str(args.top_k)], env=env)

    elif args.mode == "mpeg_plot":
        run([sys.executable, str(EXP_DIR / "plots_mpeg_bloom.py"),
             "--results", args.results], env=env)

    elif args.mode == "zernike":
        run([sys.executable, str(EXP_DIR / "mpeg70_baselines_compare.py"),
             "--root", args.root, "--method", "zernike"], env=env)

    elif args.mode == "efd":
        run([sys.executable, str(EXP_DIR / "mpeg70_baselines_compare.py"),
             "--root", args.root, "--method", "efd"], env=env)

    elif args.mode == "hu":
        run([sys.executable, str(EXP_DIR / "mpeg70_baselines_compare.py"),
             "--root", args.root, "--method", "hu"], env=env)

    elif args.mode == "learned":
        if not args.split:
            print("ERROR: --split is required for learned mode", file=sys.stderr)
            sys.exit(2)
        run([sys.executable, str(EXP_DIR / "mpeg70_bloom_prefilter_learned_split.py"),
             "--root", args.root, "--split", args.split], env=env)

    elif args.mode == "exp3":
        run([sys.executable, str(EXP_DIR / "exp3_corneal_ar_mof.py"),
             "--csv", args.csv, "--outdir", "figures"], env=env)

    else:
        parser.error("Unknown mode")

if __name__ == "__main__":
    main()
