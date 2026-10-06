"""Collapse downstream/results_relmlp/*.json into a per-regime table.

    conda run -n tabsyn python -m downstream.relmlp.summarize

For each (task, regime) it reports the primary metric (roc_auc) averaged over generator seeds
and training seeds, the spread, and the 5-seed ensemble. The `real - synth` gap is the number
the probe exists to measure; compare it against the XGBoost gap in downstream/results/.
"""

import glob
import json
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_relmlp")
PRIMARY = "roc_auc"
REGIME_ORDER = {"real": 0, "synth": 1, "mixed": 2}


def summarize_file(path):
    recs = json.load(open(path))["records"]
    print(f"\n{os.path.basename(path)}")
    by_regime = {}
    for r in recs:
        by_regime.setdefault(r["regime"], []).append(r)

    means = {}
    for regime in sorted(by_regime, key=lambda x: REGIME_ORDER.get(x, 9)):
        rs = by_regime[regime]
        per_seed = [x["metrics"][PRIMARY] for x in rs if x["train_seed"] != "ensemble"]
        ens = [x["metrics"][PRIMARY] for x in rs if x["train_seed"] == "ensemble"]
        m, sd = float(np.mean(per_seed)), float(np.std(per_seed))
        means[regime] = m
        ci = ""
        for x in rs:
            if "ci95" in x:
                lo, hi = x["ci95"][PRIMARY]
                if lo is not None:
                    ci = f"  ci95[{lo:.3f},{hi:.3f}]"
                break
        print(f"  {regime:8s}  {PRIMARY}={m:.4f} +/- {sd:.4f}  "
              f"(n={len(per_seed)})  ensemble={np.mean(ens):.4f}{ci}")
    if "real" in means and "synth" in means:
        print(f"  --> real - synth gap = {means['real'] - means['synth']:+.4f}")


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "*.json")))
    if not files:
        print(f"no results in {RESULTS}")
        return
    for f in files:
        summarize_file(f)


if __name__ == "__main__":
    main()
