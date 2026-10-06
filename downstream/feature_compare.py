"""What do the real-trained and synthetic-trained models actually learn?

    conda run --no-capture-output -n tabsyn python -m downstream.feature_compare

For each task this fits the same XGBoost on real training rows and on synthetic rows (matched
row counts, identical hyperparameters), then compares three things per feature:

  1. **Gain importance** in the real model vs the synthetic model -- what each model leans on.
  2. **Univariate AUROC** of the feature against the label, computed separately *inside* the real
     database and *inside* the synthetic one. This is the feature->target relationship itself,
     independent of any model: 0.5 means the feature carries no information about the label in
     that database.
  3. **KS distance** between the real and synthetic marginal of the feature.

(2) and (3) together are the diagnostic. A feature whose marginal matches (small KS) but whose
univariate AUROC falls to 0.5 in synthetic data has kept its distribution and lost its
relationship to the label -- which is exactly the failure a column-shape fidelity metric cannot
see.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import (
    REPO_ROOT,
    align_feature_dtypes,
    coerce_discrete,
    grdm_run_dir,
    load_db,
    load_domains,
    load_meta,
    load_synthetic_db,
)
from downstream.evaluate import XGB_PARAMS
from downstream.run import DATASETS
from downstream.tasks import TASKS

FIG_DIR = os.path.join(REPO_ROOT, "downstream", "figures")


def signed_retention(real_auc, synth_auc):
    """Share of a real feature->label relationship the synthetic data keeps, direction included.

    (synthetic AUROC - 0.5) / (real AUROC - 0.5): 1 = kept, 0 = gone, negative = reversed. The
    ratio of |AUROC - 0.5| used until 5 Oct 2026 counted a reversed relationship as kept.
    """
    if real_auc != real_auc or synth_auc != synth_auc or real_auc == 0.5:
        return float("nan")
    return (synth_auc - 0.5) / (real_auc - 0.5)


def gain_importance(X, y, seed, columns):
    """Normalised gain importance, zero-filled for features the model never split on."""
    params = dict(XGB_PARAMS)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)
    model = XGBClassifier(**params, random_state=seed, eval_metric="logloss")
    model.fit(X, y)
    score = model.get_booster().get_score(importance_type="gain")
    total = sum(score.values()) or 1.0
    return pd.Series({c: score.get(c, 0.0) / total for c in columns})


def univariate_auc(x, y):
    """AUROC of a single feature against the label. Returns (auc, coverage).

    Kept signed -- 0.42 and 0.58 carry the same strength but opposite direction, and a direction
    flip between real and synthetic data is itself a finding.
    """
    x = pd.to_numeric(x, errors="coerce")
    ok = x.notna().to_numpy()
    if ok.sum() < 20:
        return float("nan"), float(ok.mean())
    xv, yv = x[ok], np.asarray(y)[ok]
    if len(np.unique(yv)) < 2 or xv.nunique() < 2:
        return float("nan"), float(ok.mean())
    return float(roc_auc_score(yv, xv)), float(ok.mean())


def ks_distance(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 20 or len(b) < 20:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def response_curve(x_real, y_real, x_syn, y_syn, n_bins=8):
    """P(y=1) per bin, using real-data quantile edges for both so the bins are comparable."""
    xr = pd.to_numeric(x_real, errors="coerce")
    xs = pd.to_numeric(x_syn, errors="coerce")
    edges = np.unique(np.nanquantile(xr, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    br = pd.cut(xr, edges, labels=False, include_lowest=True)
    bs = pd.cut(xs, edges, labels=False, include_lowest=True)
    nb = len(edges) - 1
    out = {"centers": [], "real": [], "synth": [], "real_n": [], "synth_n": []}
    for b in range(nb):
        mr, ms = (br == b), (bs == b)
        out["centers"].append(b)
        out["real"].append(float(np.asarray(y_real)[mr.to_numpy()].mean()) if mr.sum() else np.nan)
        out["synth"].append(float(np.asarray(y_syn)[ms.to_numpy()].mean()) if ms.sum() else np.nan)
        out["real_n"].append(int(mr.sum()))
        out["synth_n"].append(int(ms.sum()))
    return out


def plot_task(task_name, rows, curves, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rows[:12]
    n_curves = len(curves)
    fig = plt.figure(figsize=(14, 4 + 2.6 * ((n_curves + 2) // 3)))
    gs = fig.add_gridspec(1 + (n_curves + 2) // 3, 3, hspace=0.55, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    idx = np.arange(len(top))
    ax.barh(idx - 0.2, [r["real_importance"] for r in top], height=0.4, label="real-trained model",
            color="#2b6cb0")
    ax.barh(idx + 0.2, [r["synth_importance"] for r in top], height=0.4,
            label="synthetic-trained model", color="#c05621")
    ax.set_yticks(idx)
    ax.set_yticklabels([r["feature"] for r in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("normalised gain importance")
    ax.set_title(f"{task_name}: what each model leans on (top 12 by real-model importance)",
                 fontsize=10)
    ax.legend(fontsize=8)

    for i, (feat, c) in enumerate(curves.items()):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.plot(c["centers"], c["real"], "o-", color="#2b6cb0", label="real")
        ax.plot(c["centers"], c["synth"], "s--", color="#c05621", label="synthetic")
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("feature octile (real-data edges)", fontsize=7)
        ax.set_ylabel("P(y=1)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"{task_name} — feature importance and feature→target relationships, real vs synthetic",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_task(task, cfg, domains, meta, real_tables, grdm_seeds, xgb_seeds, n_curves=6):
    print(f"\n{'=' * 78}\n=== {task.name}\n{'=' * 78}")
    X_real, y_real, cat_spec = task.build(real_tables, domains)
    columns = list(X_real.columns)

    syn_frames = []
    for seed in grdm_seeds:
        run_dir = grdm_run_dir(cfg["grdm_dataset"], seed, variant=cfg.get("variant", ""))
        syn = coerce_discrete(load_synthetic_db(run_dir, list(meta["tables"])), domains)
        Xs, ys, _ = task.build(syn, domains)
        syn_frames.append((seed, Xs.reindex(columns=columns), ys))
        del syn

    # Importance: real model once per xgb seed; synthetic model per (grdm seed, xgb seed),
    # both on matched row counts so neither side is advantaged by quantity.
    n_train = min([len(X_real)] + [len(Xs) for _, Xs, _ in syn_frames])
    real_imps, syn_imps = [], []
    for xgb_seed in xgb_seeds:
        rng = np.random.default_rng(xgb_seed)
        (Xr,), _ = align_feature_dtypes([X_real], cat_spec)
        idx = np.sort(rng.choice(len(Xr), size=n_train, replace=False)) if n_train < len(Xr) else slice(None)
        real_imps.append(gain_importance(Xr.iloc[idx], y_real.iloc[idx], xgb_seed, columns))
        for seed, Xs, ys in syn_frames:
            (Xsa,), _ = align_feature_dtypes([Xs], cat_spec)
            r2 = np.random.default_rng(1000 * (seed + 1) + xgb_seed)
            j = np.sort(r2.choice(len(Xsa), size=n_train, replace=False)) if n_train < len(Xsa) else slice(None)
            syn_imps.append(gain_importance(Xsa.iloc[j], ys.iloc[j], xgb_seed, columns))
        print(f"  fitted importance models for xgb seed {xgb_seed}")

    real_imp = pd.concat(real_imps, axis=1).mean(axis=1)
    syn_imp = pd.concat(syn_imps, axis=1).mean(axis=1)

    # Feature -> target relationship, model-free, inside each database.
    X_syn_all = pd.concat([Xs for _, Xs, _ in syn_frames], ignore_index=True)
    y_syn_all = pd.concat([pd.Series(np.asarray(ys)) for _, _, ys in syn_frames], ignore_index=True)

    rows = []
    for col in columns:
        auc_r, cov_r = univariate_auc(X_real[col], y_real)
        auc_s, cov_s = univariate_auc(X_syn_all[col], y_syn_all)
        rows.append(
            dict(
                feature=col,
                real_importance=float(real_imp[col]),
                synth_importance=float(syn_imp[col]),
                real_auc=auc_r,
                synth_auc=auc_s,
                # strength above chance, sign-free: how much the feature says about the label
                real_strength=abs(auc_r - 0.5) if auc_r == auc_r else float("nan"),
                synth_strength=abs(auc_s - 0.5) if auc_s == auc_s else float("nan"),
                # direction-aware: negative when the synthetic relationship is reversed
                signal_retained=signed_retention(auc_r, auc_s),
                ks=ks_distance(X_real[col], X_syn_all[col]),
                real_coverage=cov_r,
                synth_coverage=cov_s,
            )
        )
    rows.sort(key=lambda r: -r["real_importance"])

    rho = spearmanr(real_imp[columns], syn_imp[columns]).statistic
    top_r = {r["feature"] for r in rows[:10]}
    top_s = {r["feature"] for r in sorted(rows, key=lambda r: -r["synth_importance"])[:10]}

    # Aggregate: how much of the real feature->label signal survives, over features that carry any.
    carriers = [r for r in rows if r["real_strength"] == r["real_strength"] and r["real_strength"] > 0.05]
    kept = [r["signal_retained"] for r in carriers if r["signal_retained"] == r["signal_retained"]]
    retained = float(np.mean(kept)) if kept else float("nan")
    n_reversed = sum(1 for v in kept if v < 0)

    summary = dict(
        task=task.name,
        n_features=len(columns),
        n_train_matched=n_train,
        importance_spearman=float(rho),
        top10_overlap=len(top_r & top_s),
        n_signal_carriers=len(carriers),
        n_reversed=n_reversed,
        mean_signal_retained=retained,
    )
    print(f"  importance Spearman rho = {rho:.3f}   top-10 overlap = {len(top_r & top_s)}/10")
    print(f"  features with real univariate |AUC-0.5| > 0.05: {len(carriers)}"
          f" ({n_reversed} reversed in synthetic)")
    print(f"  mean fraction of that univariate signal retained in synthetic: {retained:.1%}")

    curves = {}
    for r in rows:
        if len(curves) >= n_curves:
            break
        c = response_curve(X_real[r["feature"]], y_real, X_syn_all[r["feature"]], y_syn_all)
        if c is not None:
            curves[r["feature"]] = c

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, f"{task.name}_feature_compare.png")
    plot_task(task.name, rows, curves, fig_path)
    print(f"  wrote {fig_path}")
    return rows, summary, curves, fig_path


def markdown_table(rows, k=15):
    lines = [
        "| feature | real-model imp | synth-model imp | real univar AUC | synth univar AUC | "
        "signal kept | KS(marginal) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows[:k]:
        keep = (
            f"{r['signal_retained']:.0%}"
            if r["signal_retained"] == r["signal_retained"] and r["real_strength"] > 0.01
            else "—"
        )
        fmt = lambda v: f"{v:.3f}" if v == v else "n/a"
        lines.append(
            f"| `{r['feature']}` | {r['real_importance']:.3f} | {r['synth_importance']:.3f} | "
            f"{fmt(r['real_auc'])} | {fmt(r['synth_auc'])} | {keep} | {fmt(r['ks'])} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="berka_split", choices=sorted(DATASETS))
    parser.add_argument(
        "--tasks", nargs="*", default=["berka_card_ownership", "berka_loan_default"]
    )
    parser.add_argument("--grdm-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--xgb-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "downstream", "results"))
    args = parser.parse_args()

    cfg = dict(DATASETS[args.dataset])
    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))
    real_tables = coerce_discrete(load_db(cfg["real_train"]), domains)

    payload = {}
    for task_name in args.tasks:
        rows, summary, curves, fig_path = run_task(
            TASKS[task_name], cfg, domains, meta, real_tables, args.grdm_seeds, args.xgb_seeds
        )
        payload[task_name] = dict(summary=summary, rows=rows, curves=curves, figure=fig_path)
        print(f"\n### {task_name} — top features by real-model importance")
        print("\n".join(markdown_table(rows)))

    out = os.path.join(args.out_dir, f"{args.dataset}__feature_compare.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
