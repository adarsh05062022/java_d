"""Task 5: is the collapse specific to relational aggregates, or is everything degraded?

    conda run --no-capture-output -n tabsyn python -m downstream.aggregate_validation

For a hand-picked set of cross-table aggregate features -- and a matched control set of own-row
/ 1-hop features from the same task -- this measures the univariate AUROC of the feature against
the downstream target, separately inside the real database and inside each synthetic one.

Three things are added over the sweep in `downstream.feature_compare`, because "verify" needs
more than a point estimate:

  * **Bootstrap CIs** on every AUROC. The claim "the relationship is gone" is only supportable if
    the synthetic CI actually covers 0.5, so that test is reported per feature.
  * **Per-seed synthetic values**, not a pooled number, so a collapse cannot be an artefact of one
    unlucky generation.
  * **A control group.** Loss confined to aggregates is a finding; loss everywhere would just be a
    broken generator, and only the control separates the two.

Signal retained is (synthetic AUROC - 0.5) / (real AUROC - 0.5). It keeps the direction: an
inverted real relationship (real AUC 0.25) is measured correctly, it goes to zero when a feature
stops saying anything about the label, and it turns negative when the synthetic relationship
points the other way. The |AUC-0.5| ratio used until 5 Oct 2026 counted such reversals as kept.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import (
    REPO_ROOT,
    coerce_discrete,
    grdm_run_dir,
    load_db,
    load_domains,
    load_meta,
    load_synthetic_db,
)
from downstream.feature_compare import signed_retention
from downstream.run import DATASETS
from downstream.tasks import TASKS

RESULTS = os.path.join(REPO_ROOT, "downstream", "results")
FIG_DIR = os.path.join(REPO_ROOT, "downstream", "figures")

# Cross-table aggregates: rollups over a sibling table reached *through* the parent account.
# Controls: columns of the labelled row itself or of a table it points straight at.
SELECTION = {
    "berka_card_ownership": {
        "aggregate": ["trans_n", "trans_balance_mean", "trans_balance_last",
                      "trans_amount_mean", "order_n", "order_amount_sum"],
        "control": ["disp_type", "year", "gender", "frequency"],
    },
    # On the loan task the trans rollups are restricted to transactions before the loan date and
    # grouped by loan. The plain rollups are kept for continuity with the card task; with the
    # corrected window (5 Oct 2026) the balance ones carry real signal, and the code-share rollups
    # below predict this label too.
    "berka_loan_default": {
        "aggregate": ["trans_n", "trans_balance_mean", "trans_balance_last",
                      "trans_amount_mean", "order_n", "order_amount_sum",
                      "trans_bank_0_frac", "order_k_symbol_3_frac", "order_k_symbol_4_frac",
                      "trans_operation_3_frac"],
        "control": ["amount_per_month", "payments", "amount", "duration"],
    },
}


def auc_ci(x, y, n_boot=400, seed=0):
    """AUROC with a percentile bootstrap CI. Returns (auc, lo, hi, n_used)."""
    x = pd.to_numeric(pd.Series(np.asarray(x)).reset_index(drop=True), errors="coerce")
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    ok = x.notna().to_numpy()
    xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
    if len(xv) < 20 or len(np.unique(yv)) < 2 or len(np.unique(xv)) < 2:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    point = float(roc_auc_score(yv, xv))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2:
            continue
        vals.append(roc_auc_score(yv[idx], xv[idx]))
    if not vals:
        return point, float("nan"), float("nan"), int(ok.sum())
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(ok.sum())


def strength(a):
    return abs(a - 0.5) if a == a else float("nan")


def analyse(task, cfg, domains, meta, real_tables, grdm_seeds):
    print(f"\n{'=' * 78}\n=== {task.name}\n{'=' * 78}")
    X_real, y_real, _ = task.build(real_tables, domains)

    syn = []
    for s in grdm_seeds:
        run_dir = grdm_run_dir(cfg["grdm_dataset"], s, variant=cfg.get("variant", ""))
        tabs = coerce_discrete(load_synthetic_db(run_dir, list(meta["tables"])), domains)
        Xs, ys, _ = task.build(tabs, domains)
        syn.append((s, Xs.reindex(columns=X_real.columns), ys))
        del tabs

    out = []
    for family, feats in (("aggregate", SELECTION[task.name]["aggregate"]),
                          ("control", SELECTION[task.name]["control"])):
        for feat in feats:
            if feat not in X_real.columns:
                print(f"  !! {feat} not in this task's feature set -- skipped")
                continue
            ar, lo_r, hi_r, n_r = auc_ci(X_real[feat], y_real)
            per_seed = []
            for s, Xs, ys in syn:
                a, lo, hi, n = auc_ci(Xs[feat], ys, seed=s)
                per_seed.append(dict(grdm_seed=s, auc=a, lo=lo, hi=hi, n=n,
                                     covers_half=bool(lo <= 0.5 <= hi) if lo == lo else None))
            aucs = [p["auc"] for p in per_seed if p["auc"] == p["auc"]]
            mean_s = float(np.mean(aucs)) if aucs else float("nan")
            sd_s = float(np.std(aucs)) if aucs else float("nan")
            ret = signed_retention(ar, mean_s)
            ks = float(ks_2samp(
                pd.to_numeric(X_real[feat], errors="coerce").dropna(),
                pd.to_numeric(pd.concat([Xs[feat] for _, Xs, _ in syn]), errors="coerce").dropna(),
            ).statistic)
            n_covers = sum(1 for p in per_seed if p["covers_half"])
            real_signal = bool(not (lo_r <= 0.5 <= hi_r)) if lo_r == lo_r else False
            rec = dict(task=task.name, feature=feat, family=family, real_signal=real_signal,
                       real_auc=ar, real_lo=lo_r, real_hi=hi_r, real_n=n_r,
                       synth_auc_mean=mean_s, synth_auc_sd=sd_s, per_seed=per_seed,
                       signal_retained=ret, ks_marginal=ks,
                       seeds_covering_half=n_covers, n_seeds=len(per_seed))
            out.append(rec)
            print(f"  {family:9s} {feat:22s} {'SIG' if real_signal else '---'} "
                  f"real {ar:.3f} [{lo_r:.3f},{hi_r:.3f}]   "
                  f"synth {mean_s:.3f}±{sd_s:.3f}   retained {ret:6.1%}   "
                  f"CI covers 0.5 in {n_covers}/{len(per_seed)} seeds")
    return out


def render(rows):
    lines = []
    for task_name in sorted({r["task"] for r in rows}, key=lambda t: t != "berka_card_ownership"):
        trs = [r for r in rows if r["task"] == task_name]
        lines += [f"### {task_name}", "",
                  "| feature | family | real AUROC [95% CI] | synth AUROC (mean±sd over 3 seeds) | "
                  "signal retained | synth CI covers 0.5 | KS(marginal) |",
                  "|---|---|---|---|---|---|---|"]
        for r in trs:
            fam = "cross-table agg" if r["family"] == "aggregate" else "own/1-hop"
            lines.append(
                f"| `{r['feature']}` | {fam} | {r['real_auc']:.3f} "
                f"[{r['real_lo']:.3f}, {r['real_hi']:.3f}] | "
                f"{r['synth_auc_mean']:.3f} ± {r['synth_auc_sd']:.3f} | "
                f"{('**%.0f%%**' % (100 * r['signal_retained'])) if r['real_signal'] else '_n/a_'} | "
                f"{r['seeds_covering_half']}/{r['n_seeds']} | {r['ks_marginal']:.3f} |"
            )
        agg = [r["signal_retained"] for r in trs
               if r["family"] == "aggregate" and r["real_signal"]]
        ctl = [r["signal_retained"] for r in trs
               if r["family"] == "control" and r["real_signal"]]
        n_agg_dead = sum(1 for r in trs if r["family"] == "aggregate" and r["real_signal"]
                         and r["seeds_covering_half"] == r["n_seeds"])
        n_ctl_dead = sum(1 for r in trs if r["family"] == "control" and r["real_signal"]
                         and r["seeds_covering_half"] == r["n_seeds"])
        lines += ["",
                  "*Averaged over features whose real AUROC is itself distinguishable from "
                  "chance -- a retention ratio over a real AUROC that already covers 0.5 divides "
                  "by noise. `_n/a_` marks the features excluded on that ground.*", "",
                  f"- cross-table aggregates retain **{np.mean(agg):.1%}** of their real signal "
                  f"({len(agg)} of {sum(1 for r in trs if r['family'] == 'aggregate')} carry real "
                  f"signal); **{n_agg_dead}/{len(agg)}** have a synthetic AUROC whose CI covers "
                  f"0.5 in every seed, i.e. no measurable relationship left",
                  f"- own-row / 1-hop controls retain **{np.mean(ctl):.1%}** ({len(ctl)} of "
                  f"{sum(1 for r in trs if r['family'] == 'control')} carry real signal); "
                  f"**{n_ctl_dead}/{len(ctl)}** lost to chance",
                  ""]
    return lines


def plot(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({r["task"] for r in rows}, key=lambda t: t != "berka_card_ownership")
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.4 * len(tasks), 5.2))
    axes = np.atleast_1d(axes)
    for ax, task_name in zip(axes, tasks):
        trs = [r for r in rows if r["task"] == task_name and r["real_signal"]]
        trs = ([r for r in trs if r["family"] == "aggregate"]
               + [r for r in trs if r["family"] == "control"])
        y = np.arange(len(trs))
        rs = [strength(r["real_auc"]) for r in trs]
        # synthetic strength in the real relationship's direction, so a reversal plots below 0
        ss = [(r["synth_auc_mean"] - 0.5) * (1.0 if r["real_auc"] >= 0.5 else -1.0) for r in trs]
        err = [[abs(strength(r["real_auc"]) - strength(r["real_lo"])) for r in trs],
               [abs(strength(r["real_hi"]) - strength(r["real_auc"])) for r in trs]]
        ax.barh(y - 0.2, rs, height=0.4, color="#1d4e89", label="real database",
                xerr=err, error_kw=dict(ecolor="#8fa8c4", lw=1, capsize=2))
        ax.barh(y + 0.2, ss, height=0.4, color="#b0561a", label="synthetic database",
                xerr=[np.zeros(len(trs)), [r["synth_auc_sd"] for r in trs]],
                error_kw=dict(ecolor="#d9a97f", lw=1, capsize=2))
        n_agg = sum(1 for r in trs if r["family"] == "aggregate")
        ax.axhline(n_agg - 0.5, color="#4a5568", lw=1, ls="--")
        ax.text(ax.get_xlim()[1] * 0.99, n_agg - 0.62, "cross-table aggregates", fontsize=8,
                ha="right", va="bottom", color="#4a5568", style="italic")
        ax.text(ax.get_xlim()[1] * 0.99, n_agg - 0.38, "own-row / 1-hop controls", fontsize=8,
                ha="right", va="top", color="#4a5568", style="italic")
        ax.set_yticks(y)
        ax.set_yticklabels([r["feature"] for r in trs], fontsize=8.5, family="monospace")
        ax.invert_yaxis()
        ax.axvline(0, color="#4a5568", lw=0.8)
        ax.set_xlabel("univariate signal about the label, real direction   (below 0 = reversed)")
        ax.set_title(task_name, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Task 5 — the same measurement on both feature families\n"
        "cross-table aggregates lose their relationship to the label; own-row features keep theirs",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="berka_split", choices=sorted(DATASETS))
    p.add_argument("--tasks", nargs="*", default=["berka_card_ownership", "berka_loan_default"])
    p.add_argument("--grdm-seeds", nargs="*", type=int, default=[0, 1, 2])
    args = p.parse_args()

    cfg = dict(DATASETS[args.dataset])
    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))
    real_tables = coerce_discrete(load_db(cfg["real_train"]), domains)

    rows = []
    for t in args.tasks:
        rows += analyse(TASKS[t], cfg, domains, meta, real_tables, args.grdm_seeds)

    out = os.path.join(RESULTS, f"{args.dataset}__aggregate_validation.json")
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "rows": rows}, f, indent=2)

    print("\n" + "\n".join(render(rows)))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig = os.path.join(FIG_DIR, "aggregate_validation.png")
    plot(rows, fig)
    print(f"wrote {fig}\nwrote {out}")


if __name__ == "__main__":
    main()
