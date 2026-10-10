#!/usr/bin/env python3
"""Fidelity summary of the paper_repro baseline runs, written to results/paper_repro/.

Stdlib only (no conda needed). For each dataset and seed whose evaluation has finished
(eval_result.pkl exists), reads the last "Final results" block of
logs/train/paper_repro_<dataset>_seed<N>.log, and writes:

    results/paper_repro/summary.md     per-seed scores, mean ± std, paper, delta
    results/paper_repro/summary.json   the same numbers, machine-readable
    results/paper_repro/logs/          a copy of each finished seed's log, progress bars stripped

Unfinished seeds are listed as pending and their logs are not copied, so re-running this only
changes the output when a seed finishes. Prints a one-line status for the commit message.

    python3 scripts/repro_summary.py                     # berka and california
    python3 scripts/repro_summary.py berka california ccs
"""
import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "paper_repro")
SEEDS = (0, 1, 2)
METRICS = ("Cardinality", "Column Shapes", "Intra-Table Trends", "1-HOP", "2-HOP", "3-HOP", "AVG 2-WAY")

# GRDM column of the paper's Table 1 (mean, std), and the first reproduction of August 2026
# (docs/REPRODUCTION.md §2), whose checkpoints were lost with /scratch.
PAPER = {
    "berka": {
        "Cardinality": (99.65, 0.05), "Column Shapes": (96.84, 0.14),
        "Intra-Table Trends": (98.23, 0.03), "1-HOP": (91.41, 0.11),
        "2-HOP": (95.57, 0.04), "3-HOP": (92.43, 0.69),
    },
    "california": {
        "Cardinality": (99.96, 0.01), "Column Shapes": (98.75, 0.00),
        "Intra-Table Trends": (97.35, 0.02), "1-HOP": (96.16, 0.05),
    },
    "ccs": {
        "Cardinality": (99.79, 0.03), "Column Shapes": (97.30, 0.36),
        "Intra-Table Trends": (94.82, 1.20), "1-HOP": (85.38, 3.03),
    },
}
PREVIOUS = {
    "berka": {"Cardinality": 99.65, "Column Shapes": 96.89, "Intra-Table Trends": 98.26,
              "1-HOP": 91.43, "2-HOP": 95.71, "3-HOP": 92.06, "AVG 2-WAY": 95.25},
    "california": {"Cardinality": 99.96, "Column Shapes": 98.60, "Intra-Table Trends": 97.53,
                   "1-HOP": 96.48, "AVG 2-WAY": 97.00},
    "ccs": {"Cardinality": 99.79, "Column Shapes": 97.21, "Intra-Table Trends": 94.80,
            "1-HOP": 82.61, "AVG 2-WAY": 85.93},
}


def run_dir(dataset, seed):
    return os.path.join(REPO, "rdbdiff_workspace", dataset, "paper_repro", f"seed_{seed}")


def log_path(dataset, seed):
    return os.path.join(REPO, "logs", "train", f"paper_repro_{dataset}_seed{seed}.log")


def is_finished(dataset, seed):
    return os.path.exists(os.path.join(
        run_dir(dataset, seed), "synthetic_data_sampled_structure_size_multiplier_1.0", "eval_result.pkl"))


def clean_lines(path):
    """Log lines with tqdm redraws collapsed to their final state."""
    with open(path, errors="replace") as f:
        for line in f:
            parts = [p for p in line.rstrip("\n").split("\r") if p.strip()]
            yield parts[-1] if parts else ""


def final_scores(lines):
    """Scores from the last "Final results" block; a resumed run may evaluate more than once."""
    scores = None
    for line in lines:
        if line.startswith("Final results on dataset"):
            scores = {}
        elif scores is not None:
            m = re.match(r"^([A-Za-z0-9 -]+?): ([0-9.]+)$", line)
            if m and m.group(1) in METRICS:
                scores[m.group(1)] = float(m.group(2))
    return scores


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if x is not None else "–"


def summarise(dataset):
    seeds = {}
    for seed in SEEDS:
        if not is_finished(dataset, seed) or not os.path.exists(log_path(dataset, seed)):
            continue
        lines = list(clean_lines(log_path(dataset, seed)))
        scores = final_scores(lines)
        if not scores:
            continue
        seeds[seed] = scores
        os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
        with open(os.path.join(OUT, "logs", os.path.basename(log_path(dataset, seed))), "w") as f:
            f.write("\n".join(lines) + "\n")

    metrics = [m for m in METRICS if any(m in s for s in seeds.values())] or \
        [m for m in METRICS if m in PAPER.get(dataset, {})]
    rows = {}
    for m in metrics:
        vals = [seeds[s][m] for s in SEEDS if s in seeds and m in seeds[s]]
        mean = statistics.mean(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) >= 2 else None
        paper = PAPER.get(dataset, {}).get(m)
        rows[m] = {
            "seeds": {str(s): seeds[s].get(m) for s in seeds},
            "mean": mean, "std": std,
            "paper_mean": paper[0] if paper else None, "paper_std": paper[1] if paper else None,
            "delta": mean - paper[0] if mean is not None and paper else None,
            "previous_mean": PREVIOUS.get(dataset, {}).get(m),
        }
    return {"finished_seeds": sorted(seeds), "metrics": rows}


def markdown(results):
    out = [
        "# GRDM paper reproduction: Table 1, GRDM column",
        "",
        "Written by `scripts/repro_summary.py`; do not edit by hand. Runs use "
        "`configs/<dataset>_fastsample.json`, `--exp_name paper_repro`, conda env `tabsyn`, "
        "launched by `scripts/resume_runs.sh` under `scripts/watchdog.sh`. Scores are the "
        "\"Final results\" block of each seed's log (copied to `logs/`). Δ is ours minus the paper; "
        "the last column is the first reproduction of August 2026 (`docs/REPRODUCTION.md` §2).",
    ]
    for dataset, r in results.items():
        done = r["finished_seeds"]
        out += ["", f"## {dataset} ({len(done)}/{len(SEEDS)} seeds finished)", ""]
        head = "| metric | " + " | ".join(f"seed {s}" for s in SEEDS) + " | ours | paper | Δ | Aug 2026 |"
        out += [head, "|" + "---|" * (len(SEEDS) + 5)]
        for m, row in r["metrics"].items():
            per_seed = [fmt(row["seeds"].get(str(s))) if s in done else "pending" for s in SEEDS]
            ours = fmt(row["mean"]) + (f" ± {fmt(row['std'])}" if row["std"] is not None else "")
            paper = (f"{fmt(row['paper_mean'])} ± {fmt(row['paper_std'])}"
                     if row["paper_mean"] is not None else "not reported")
            delta = f"{row['delta']:+.2f}" if row["delta"] is not None else "–"
            out.append(f"| {m} | " + " | ".join(per_seed) +
                       f" | {ours} | {paper} | {delta} | {fmt(row['previous_mean'])} |")
    return "\n".join(out) + "\n"


def main():
    datasets = sys.argv[1:] or ["berka", "california"]
    results = {d: summarise(d) for d in datasets}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(markdown(results))
    print(", ".join(f"{d} {len(r['finished_seeds'])}/{len(SEEDS)}" for d, r in results.items()))


if __name__ == "__main__":
    main()
