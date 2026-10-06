"""Which *kind* of feature loses its relationship to the label under generation?

    conda run --no-capture-output -n tabsyn python -m downstream.feature_provenance

Reads `berka_split__feature_compare.json` (no refitting) and re-cuts it by where each feature
comes from in the schema:

  own row / 1-hop parent   a column of the labelled row itself, or of a table it points straight
                           at (loan.amount; disp.disp_type; the client and account a disp joins)
  child aggregate          a rollup over a *sibling* table reached through the parent
                           (trans_*, order_*, n_disps, the client-demographic rollups)
  district dimension       the static dimension table GRDM holds fixed

The comparison that matters is the univariate AUROC of a feature against the label, measured
inside the real database and inside the synthetic one. It needs no model, so it separates "the
generator did not reproduce this relationship" from "XGBoost chose not to use it".
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT
from downstream.feature_compare import signed_retention

RESULTS = os.path.join(REPO_ROOT, "downstream", "results")
FIG_DIR = os.path.join(REPO_ROOT, "downstream", "figures")

# A feature carries usable signal in the real data if its univariate AUROC is this far from 0.5.
# Below it, the "fraction retained" ratio divides by noise and means nothing.
SIGNAL_FLOOR = 0.05

CHILD_AGG_PREFIXES = ("trans_", "order_", "disp_")
CHILD_AGG_EXACT = {
    "n_disps", "client_year_min", "client_year_max", "client_month_mean", "client_gender_mean",
}


def provenance(feature):
    if feature.startswith("dist_"):
        return "district dimension"
    if feature in CHILD_AGG_EXACT or feature.startswith(CHILD_AGG_PREFIXES):
        # `disp_type` is the labelled row's own column; `disp_disp_type_*_frac` is a rollup.
        if feature == "disp_type":
            return "own row / 1-hop parent"
        return "child aggregate"
    return "own row / 1-hop parent"


def strength(auc):
    return abs(auc - 0.5) if auc == auc else float("nan")


def aligned_synth(r):
    """Synthetic strength measured in the real relationship's direction; negative = reversed."""
    if r["real_auc"] != r["real_auc"] or r["synth_auc"] != r["synth_auc"]:
        return float("nan")
    return (r["synth_auc"] - 0.5) * (1.0 if r["real_auc"] >= 0.5 else -1.0)


def group_stats(rows):
    groups = {}
    for r in rows:
        g = provenance(r["feature"])
        rs, ss = strength(r["real_auc"]), strength(r["synth_auc"])
        if rs != rs or ss != ss:
            continue
        groups.setdefault(g, {"all": [], "carriers": []})
        groups[g]["all"].append((r, rs, ss))
        if rs > SIGNAL_FLOOR:
            groups[g]["carriers"].append((r, rs, ss))
    out = []
    for g, d in groups.items():
        car = d["carriers"]
        kept = [signed_retention(r["real_auc"], r["synth_auc"]) for r, _, _ in car]
        out.append(
            dict(
                group=g,
                n_features=len(d["all"]),
                n_carriers=len(car),
                n_reversed=sum(1 for v in kept if v < 0),
                mean_real_strength=float(np.mean([s for _, s, _ in car])) if car else float("nan"),
                mean_synth_strength=float(np.mean([s for _, _, s in car])) if car else float("nan"),
                mean_retained=float(np.mean(kept)) if car else float("nan"),
                real_importance=float(sum(r["real_importance"] for r, _, _ in d["all"])),
                synth_importance=float(sum(r["synth_importance"] for r, _, _ in d["all"])),
            )
        )
    order = ["own row / 1-hop parent", "child aggregate", "district dimension"]
    out.sort(key=lambda r: order.index(r["group"]) if r["group"] in order else 9)
    return out


def render_groups(task_name, rows):
    stats = group_stats(rows)
    lines = [
        f"**{task_name} — by where the feature comes from in the schema**",
        "",
        "| feature family | features | signal carriers | reversed in synth | mean real \\|AUC−0.5\\| "
        "| signal retained | real-model imp | synth-model imp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        # A family with no signal carriers has nothing to retain; "—" beats a nan.
        num = lambda v, f: format(v, f) if v == v else "—"
        lines.append(
            f"| {s['group']} | {s['n_features']} | {s['n_carriers']} | {s['n_reversed']} | "
            f"{num(s['mean_real_strength'], '.3f')} | "
            f"**{num(s['mean_retained'], '.0%')}** | {s['real_importance']:.2f} | "
            f"{s['synth_importance']:.2f} |"
        )
    lines.append("")
    return lines, stats


def render_top(task_name, rows, k=12):
    """Top features by how much signal they carry in the REAL data -- the ones that matter."""
    carriers = [r for r in rows if strength(r["real_auc"]) > SIGNAL_FLOOR]
    carriers.sort(key=lambda r: -strength(r["real_auc"]))
    lines = [
        f"**{task_name} — the features that actually predict the label in real data**",
        "",
        "| feature | family | real univar AUC | synth univar AUC | signal kept | KS(marginal) | "
        "real imp | synth imp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in carriers[:k]:
        kept = signed_retention(r["real_auc"], r["synth_auc"])
        keep = f"{kept:.0%}" if kept == kept else "n/a"
        fam = {"own row / 1-hop parent": "own/1-hop", "child aggregate": "child agg",
               "district dimension": "district"}[provenance(r["feature"])]
        lines.append(
            f"| `{r['feature']}` | {fam} | {r['real_auc']:.3f} | {r['synth_auc']:.3f} | "
            f"**{keep}** | {r['ks']:.3f} | {r['real_importance']:.3f} | {r['synth_importance']:.3f} |"
        )
    lines.append("")
    return lines


def plot(payload, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(payload)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.4 * len(tasks), 5.0))
    axes = np.atleast_1d(axes)
    colors = {"own row / 1-hop parent": "#2f855a", "child aggregate": "#c05621",
              "district dimension": "#718096"}
    for ax, task_name in zip(axes, tasks):
        rows = payload[task_name]
        for g, c in colors.items():
            pts = [
                (strength(r["real_auc"]), aligned_synth(r))
                for r in rows
                if provenance(r["feature"]) == g
                and strength(r["real_auc"]) == strength(r["real_auc"])
                and aligned_synth(r) == aligned_synth(r)
            ]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, alpha=0.8, color=c,
                           label=g, edgecolors="white", linewidths=0.5)
        lim = max(
            [strength(r["real_auc"]) for r in rows if strength(r["real_auc"]) == strength(r["real_auc"])]
            + [0.1]
        ) * 1.12
        low = min([aligned_synth(r) for r in rows if aligned_synth(r) == aligned_synth(r)] + [0.0])
        ax.plot([0, lim], [0, lim], ":", color="#4a5568", lw=1)
        ax.text(lim * 0.97, lim * 0.97, "relationship preserved", fontsize=7, color="#4a5568",
                ha="right", va="bottom", rotation=45)
        ax.axhline(0, color="#a0aec0", lw=0.8)
        ax.set_xlabel("signal in REAL data  |univariate AUROC − 0.5|")
        ax.set_ylabel("signal in SYNTHETIC data, real direction (below 0 = reversed)")
        ax.set_xlim(0, lim)
        ax.set_ylim(min(low * 1.12, -0.02), lim)
        ax.set_title(task_name, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Feature→target relationships: what survives generation, by schema distance\n"
        "points on the diagonal keep their predictive relationship; on the zero line it is lost; "
        "below it, reversed",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=os.path.join(RESULTS, "berka_split__feature_compare.json"))
    args = parser.parse_args()

    with open(args.path) as f:
        payload = json.load(f)

    doc, rows_by_task = [], {}
    for task_name, d in payload.items():
        rows = d["rows"]
        rows_by_task[task_name] = rows
        s = d["summary"]
        doc.append(f"### {task_name}")
        doc.append("")
        doc.append(
            f"Importance-ranking agreement between the real-trained and synthetic-trained model: "
            f"Spearman ρ = **{s['importance_spearman']:.3f}**, top-10 overlap "
            f"**{s['top10_overlap']}/10**. Across the {s['n_signal_carriers']} features that carry "
            f"real signal, the synthetic data retains **{s['mean_signal_retained']:.1%}** of it on "
            f"average."
        )
        doc.append("")
        g, _ = render_groups(task_name, rows)
        doc += g
        doc += render_top(task_name, rows)

    print("\n".join(doc))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "feature_signal_survival.png")
    plot(rows_by_task, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
