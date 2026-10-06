"""Render the 2x2 cross-domain transfer matrices from `downstream.crossdomain` output.

    conda run --no-capture-output -n tabsyn python -m downstream.crossdomain_report
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT

RESULTS = os.path.join(REPO_ROOT, "downstream", "results")
FIG_DIR = os.path.join(REPO_ROOT, "downstream", "figures")
METRICS = ["roc_auc", "balanced_accuracy", "f1_macro"]
LABEL = {"real": "real", "synth": "synthetic"}


def cell(recs, tr, te, metric):
    vals = [r["metrics"][metric] for r in recs if r["train_on"] == tr and r["test_on"] == te]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def matrix(recs, metric):
    return {(tr, te): cell(recs, tr, te, metric) for tr in ("real", "synth")
            for te in ("real", "synth")}


def render(task_name, recs):
    lines = [f"### {task_name}", ""]
    seeds = sorted({r["grdm_seed"] for r in recs})
    n_tr = sorted({r["n_train"] for r in recs})
    tests = {r["test_on"]: (r["n_test"], r["n_test_pos"]) for r in recs}
    lines.append(
        f"*{len(seeds)} GRDM seeds x {len({r['xgb_seed'] for r in recs})} XGBoost seeds; "
        f"training rows matched at {min(n_tr)}–{max(n_tr)}; "
        + "; ".join(f"{LABEL[k]} test n={v[0]} ({v[1]} positive)" for k, v in sorted(tests.items()))
        + ".*"
    )
    lines.append("")
    for metric in METRICS:
        m = matrix(recs, metric)
        lines.append(f"**{metric}** — rows = trained on, columns = tested on")
        lines.append("")
        lines.append("| train \\ test | real | synthetic |")
        lines.append("|---|---|---|")
        for tr in ("real", "synth"):
            cells = []
            for te in ("real", "synth"):
                c = m[(tr, te)]
                cells.append(f"{c[0]:.4f} ± {c[1]:.4f}" if c else "n/a")
            lines.append(f"| **{LABEL[tr]}** | {cells[0]} | {cells[1]} |")
        lines.append("")
    rr, sr, ss, rs = (matrix(recs, "roc_auc")[k] for k in
                      [("real", "real"), ("synth", "real"), ("synth", "synth"), ("real", "synth")])
    if rr and sr and ss:
        floor = 0.5
        lines.append(
            f"- **Transfer (synth→real):** {sr[0]:.4f} vs real→real {rr[0]:.4f} — "
            f"{(sr[0] - floor) / (rr[0] - floor):.1%} of the real model's above-chance AUROC."
        )
        lines.append(
            f"- **Internal signal (synth→synth):** {ss[0]:.4f}. "
            + (
                "The synthetic database barely contains a learnable relationship at all, so the "
                "transfer failure is *absence* of dependency, not a different dependency."
                if ss[0] - floor < 0.4 * (rr[0] - floor)
                else "The synthetic database does contain a learnable relationship internally."
            )
        )
        if rs:
            lines.append(
                f"- **Real rules on synthetic rows (real→synth):** {rs[0]:.4f}, versus "
                f"synth→synth {ss[0]:.4f} — a real-trained model is "
                f"{'no better than' if abs(rs[0] - ss[0]) < 0.02 else 'different from'} "
                "the synthetic-trained one on synthetic data."
            )
        lines.append("")
    return lines


def plot(payload, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(payload)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 4.4))
    axes = np.atleast_1d(axes)
    for ax, task_name in zip(axes, tasks):
        m = matrix(payload[task_name], "roc_auc")
        grid = np.array([[m[("real", "real")][0], m[("real", "synth")][0]],
                         [m[("synth", "real")][0], m[("synth", "synth")][0]]])
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0.5, vmax=0.9)
        for i in range(2):
            for j in range(2):
                std = [[m[("real", "real")][1], m[("real", "synth")][1]],
                       [m[("synth", "real")][1], m[("synth", "synth")][1]]][i][j]
                ax.text(j, i, f"{grid[i, j]:.3f}\n±{std:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color="#1a202c")
        ax.set_xticks([0, 1], ["real", "synthetic"])
        ax.set_yticks([0, 1], ["real", "synthetic"])
        ax.set_xlabel("tested on")
        ax.set_ylabel("trained on")
        ax.set_title(f"{task_name}\nAUROC", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label="AUROC")
    fig.suptitle("Cross-domain transfer: does a synthetic-trained model work on the real database?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="berka_split__*_crossdomain.json")
    args = parser.parse_args()

    order = ["berka_card_ownership", "berka_loan_default"]
    payload = {}
    for path in glob.glob(os.path.join(RESULTS, args.pattern)):
        d = json.load(open(path))
        payload[d["task"]] = d["records"]
    payload = {k: payload[k] for k in order if k in payload} | {
        k: v for k, v in payload.items() if k not in order
    }

    doc = []
    for task_name, recs in payload.items():
        doc += render(task_name, recs)
    print("\n".join(doc))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "crossdomain_matrix.png")
    plot(payload, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
