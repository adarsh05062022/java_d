"""Compute dependency-fidelity checks on real Berka data and one or more
finished GRDM synthetic runs; print a comparison table and save JSON.

Usage:
    conda run -n tabsyn python -m dep_fidelity.run \\
        --synthetic rdbdiff_workspace/berka/paper_repro/seed_0/synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database \\
        --synthetic rdbdiff_workspace/berka/paper_repro/seed_1/synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database \\
        --label paper_repro
"""

import argparse
import json
import os

import numpy as np

from dep_fidelity.checks import run_all
from dep_fidelity.load import load_real, load_synthetic

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

METRIC_KEYS = ("violation_rate", "median_lag_days", "p10_lag_days",
               "consistency_rate", "median_abs_error", "mean_abs_error",
               "implication_rate")


def print_table(real_results: list, seed_results: list) -> None:
    print(f"{'check':52s} {'n(real)':>9s} {'metric':22s} {'real':>10s} {'synth mean':>12s} {'synth sd':>10s}")
    print("-" * 120)
    for i, real in enumerate(real_results):
        if "error" in real:
            print(f"{real['check']:52s}  ERROR: {real['error']}")
            continue
        for key in METRIC_KEYS:
            if key not in real:
                continue
            vals = [s[i].get(key) for s in seed_results if key in s[i]]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            mean = np.mean(vals) if vals else float("nan")
            sd = np.std(vals) if vals else float("nan")
            print(f"{real['check']:52s} {real['n']:9d} {key:22s} {real[key]:10.4f} {mean:12.4f} {sd:10.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="append", required=True,
                         help="path to a finished run's synthetic_database dir; repeat for multiple seeds")
    parser.add_argument("--label", default="run", help="name for the saved results JSON")
    args = parser.parse_args()

    real_tables, real_epochs = load_real()
    real_results = run_all(real_tables, real_epochs)

    seed_results = []
    for syn_dir in args.synthetic:
        tables, epochs = load_synthetic(syn_dir)
        seed_results.append(run_all(tables, epochs))

    print_table(real_results, seed_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"label": args.label, "synthetic_paths": args.synthetic,
             "real": real_results, "synthetic_seeds": seed_results},
            f, indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
