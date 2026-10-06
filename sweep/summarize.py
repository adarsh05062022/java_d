#!/usr/bin/env python3
"""Compact per-variant retained-signal summary for the sweep status page.

Stdlib only (no conda needed). Reads downstream/results/<name>__<task>.json and prints
    <name>\t<taskA_ret>\t<taskB_ret>\t<nseeds>
for each name passed on argv. Retained = (syn_mean - floor) / (real_mean - floor),
floor 0.5 for AUROC, 0 for R2 / macro-F1. "-" when not yet scored.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "downstream", "results")

# (task, primary metric, floor, real baseline mean)
TASKS = {
    "berka_loan_default": ("roc_auc", 0.5, 0.8886),
    "berka_card_ownership": ("roc_auc", 0.5, 0.8739),
    "california_inctot": ("r2", 0.0, 0.5527),
    "california_empstat": ("f1_macro", 0.0, 0.6850),
}
ORDER = {
    "berka": ["berka_loan_default", "berka_card_ownership"],
    "california": ["california_inctot", "california_empstat"],
}


def summarise(name, task):
    path = os.path.join(RES, f"{name}__{task}.json")
    if not os.path.exists(path):
        return None, 0
    try:
        recs = json.load(open(path))["records"]
    except Exception:
        return None, 0
    metric, floor, real = TASKS[task]
    syn = [r["metrics"][metric] for r in recs if r["regime"] == "synth"]
    seeds = {r["grdm_seed"] for r in recs if r["regime"] == "synth" and r.get("grdm_seed") is not None}
    if not syn:
        return None, 0
    m = sum(syn) / len(syn)
    if real == floor:
        return None, len(seeds)
    return (m - floor) / (real - floor), len(seeds)


def main():
    for name in sys.argv[1:]:
        fam = "california" if name.startswith("california") else "berka"
        cells = []
        nseeds = 0
        for task in ORDER[fam]:
            ret, n = summarise(name, task)
            nseeds = max(nseeds, n)
            cells.append(f"{ret*100:.1f}%" if ret is not None else "-")
        print(f"{name}\t{cells[0]}\t{cells[1]}\t{nseeds}")


if __name__ == "__main__":
    main()
