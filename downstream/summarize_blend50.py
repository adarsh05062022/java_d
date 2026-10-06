"""Quick read-out of the real vs real_50 vs blend_50 substitution experiment.

    conda run --no-capture-output -n tabsyn python -m downstream.summarize_blend50
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ORDER = ["real", "real_50", "blend_50", "real_25", "blend_25"]
LABEL = {
    "real": "100% real (full)",
    "real_50": "50% real only",
    "blend_50": "50% real + 50% synth",
    "real_25": "25% real only",
    "blend_25": "25% real + 75% synth",
}
PAIR = {"blend_50": "real_50", "blend_25": "real_25"}


def summarise(recs, regime, metric):
    vals = [r["metrics"][metric] for r in recs if r["regime"] == regime and metric in r["metrics"]]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def gen_spread(recs, regime, metric):
    seeds = sorted({r["grdm_seed"] for r in recs if r["regime"] == regime and r["grdm_seed"] is not None})
    if len(seeds) < 2:
        return None
    means = [
        np.mean([r["metrics"][metric] for r in recs
                 if r["regime"] == regime and r["grdm_seed"] == s])
        for s in seeds
    ]
    return float(np.std(means))


for path in sorted(glob.glob(os.path.join(RESULTS, "*_blend.json"))):
    payload = json.load(open(path))
    recs = payload["records"]
    tasks = sorted({r["task"] for r in recs})
    for task in tasks:
        trecs = [r for r in recs if r["task"] == task]
        metrics = list(trecs[0]["metrics"])
        primary = "roc_auc" if "roc_auc" in metrics else ("r2" if "r2" in metrics else metrics[0])
        print(f"\n### {os.path.basename(path)}  ::  {task}   (primary = {primary})")
        base = summarise(trecs, "real", primary)
        print(f"{'regime':24s} {'n_train':>9s}  " + "  ".join(f"{m:>18s}" for m in metrics))
        for regime in ORDER:
            first = next((r for r in trecs if r["regime"] == regime), None)
            if first is None:
                continue
            cells = []
            for m in metrics:
                s = summarise(trecs, regime, m)
                cell = f"{s[0]:.4f}±{s[1]:.4f}"
                if regime != "real" and base:
                    d = summarise(trecs, regime, m)[0] - summarise(trecs, "real", m)[0]
                    cell += f" ({d:+.4f})"
                cells.append(cell)
            n = first["n_train"]
            print(f"{LABEL[regime]:24s} {n:>9,d}  " + "  ".join(f"{c:>18s}" for c in cells))
        for blend, ctrl in PAIR.items():
            b = summarise(trecs, blend, primary)
            c = summarise(trecs, ctrl, primary)
            if not (b and c):
                continue
            sp = gen_spread(trecs, blend, primary)
            sp_s = f", GRDM-seed spread ±{sp:.4f}" if sp is not None else ""
            print(f"  >>> {blend}: synthetic fill vs dropping those rows = {b[0] - c[0]:+.4f} "
                  f"{primary} ({b[0]:.4f} vs {ctrl} {c[0]:.4f}){sp_s}")
