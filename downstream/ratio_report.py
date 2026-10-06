"""Task 1: the real->synthetic replacement ladder, consolidated into one table and one plot.

    conda run --no-capture-output -n tabsyn python -m downstream.ratio_report

Reads the already-computed regime results (no refitting) and assembles the five points where
the *total* training-set size is held fixed at the real training row count and only the
composition changes:

    100% real  ->  75R/25S  ->  50R/50S  ->  25R/75S  ->  100% synthetic

Sources: `*_blend.json` (real, real_50, blend_50, real_25, blend_25) and `*_blend75.json`
(real, synth, real_75, blend_75). The dashed "real only, rows dropped" series is the control:
the same real fraction with the synthetic half simply left out, which is what the replacement
has to beat to be worth anything.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT
from downstream.tasks import TASKS

RESULTS = os.path.join(REPO_ROOT, "downstream", "results")
FIG_DIR = os.path.join(REPO_ROOT, "downstream", "figures")

METRICS = ["roc_auc", "balanced_accuracy", "f1_macro"]

# (label, synthetic share, regime name, which result file it lives in)
LADDER = [
    ("100% real", 0.00, "real", "blend75"),
    ("75% real + 25% synth", 0.25, "blend_75", "blend75"),
    ("50% real + 50% synth", 0.50, "blend_50", "blend"),
    ("25% real + 75% synth", 0.75, "blend_25", "blend"),
    ("100% synthetic", 1.00, "synth", "blend75"),
]
# The control at each rung: keep the real fraction, drop the rest instead of replacing it.
CONTROL = {0.25: ("real_75", "blend75"), 0.50: ("real_50", "blend"), 0.75: ("real_25", "blend")}


def load(task_name):
    out = {}
    for tag, suffix in (("blend", "_blend"), ("blend75", "_blend75")):
        path = os.path.join(RESULTS, f"berka_split__{task_name}{suffix}.json")
        with open(path) as f:
            out[tag] = json.load(f)["records"]
    return out


def stat(records, regime, metric):
    vals = [r["metrics"][metric] for r in records if r["regime"] == regime]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def build(task_name):
    src = load(task_name)
    ladder, control = [], []
    for label, share, regime, tag in LADDER:
        recs = src[tag]
        row = dict(label=label, synth_share=share, regime=regime)
        first = next((r for r in recs if r["regime"] == regime), None)
        row["n_train"] = first["n_train"] if first else None
        for m in METRICS:
            s = stat(recs, regime, m)
            row[m], row[f"{m}_std"] = (s[0], s[1]) if s else (None, None)
        ladder.append(row)
        if share in CONTROL:
            creg, ctag = CONTROL[share]
            crow = dict(label=f"{int((1 - share) * 100)}% real only", synth_share=share,
                        regime=creg)
            cfirst = next((r for r in src[ctag] if r["regime"] == creg), None)
            crow["n_train"] = cfirst["n_train"] if cfirst else None
            for m in METRICS:
                s = stat(src[ctag], creg, m)
                crow[m], crow[f"{m}_std"] = (s[0], s[1]) if s else (None, None)
            control.append(crow)
    return ladder, control


def table(task_name, ladder, control):
    task = TASKS[task_name]
    base = ladder[0]["roc_auc"]
    lines = [
        f"### {task_name} — replacement ladder (total training rows held fixed)",
        "",
        "| training mix | n train | AUROC | Δ AUROC | balanced acc | macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in ladder:
        d = "—" if r["synth_share"] == 0 else f"{r['roc_auc'] - base:+.4f}"
        lines.append(
            f"| {r['label']} | {r['n_train']:,} | {r['roc_auc']:.4f} ± {r['roc_auc_std']:.4f} | {d} | "
            f"{r['balanced_accuracy']:.4f} | {r['f1_macro']:.4f} |"
        )
    lines += ["", "*Control — the same real fraction with the synthetic part simply left out:*", "",
              "| training mix | n train | AUROC | balanced acc | macro-F1 | synthetic fill buys |",
              "|---|---|---|---|---|---|"]
    by_share = {r["synth_share"]: r for r in ladder}
    for c in control:
        blend = by_share[c["synth_share"]]
        lines.append(
            f"| {c['label']} | {c['n_train']:,} | {c['roc_auc']:.4f} ± {c['roc_auc_std']:.4f} | "
            f"{c['balanced_accuracy']:.4f} | {c['f1_macro']:.4f} | "
            f"**{blend['roc_auc'] - c['roc_auc']:+.4f}** AUROC |"
        )
    return lines


def plot(payload, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(payload)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.6))
    axes = np.atleast_1d(axes)
    for ax, task_name in zip(axes, tasks):
        ladder, control = payload[task_name]
        x = [r["synth_share"] for r in ladder]
        y = [r["roc_auc"] for r in ladder]
        e = [r["roc_auc_std"] for r in ladder]
        ax.errorbar(x, y, yerr=e, fmt="o-", color="#2b6cb0", capsize=3, lw=2,
                    label="real replaced by synthetic (size fixed)")
        cx = [c["synth_share"] for c in control]
        cy = [c["roc_auc"] for c in control]
        ax.plot(cx, cy, "s--", color="#718096", label="same real fraction, rows dropped")
        ax.axhline(y[0], color="#2b6cb0", ls=":", lw=1, alpha=0.6)
        ax.axhline(0.5, color="#c53030", ls=":", lw=1, alpha=0.7)
        ax.text(0.98, 0.503, "chance", color="#c53030", fontsize=7, va="bottom", ha="right")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["100R", "75R/25S", "50R/50S", "25R/75S", "100S"], fontsize=8)
        ax.set_xlabel("training mix")
        ax.set_ylabel("AUROC")
        ax.set_title(task_name, fontsize=10)
        ax.set_ylim(0.45, 0.95)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Berka downstream utility as real training data is replaced by GRDM synthetic data",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="*", default=["berka_card_ownership", "berka_loan_default"]
    )
    args = parser.parse_args()

    payload, doc = {}, []
    for task_name in args.tasks:
        ladder, control = build(task_name)
        payload[task_name] = (ladder, control)
        doc += table(task_name, ladder, control) + [""]

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "replacement_ladder.png")
    plot(payload, fig_path)

    out_json = os.path.join(RESULTS, "replacement_ladder.json")
    with open(out_json, "w") as f:
        json.dump({t: {"ladder": l, "control": c} for t, (l, c) in payload.items()}, f, indent=2)

    print("\n".join(doc))
    print(f"wrote {fig_path}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
