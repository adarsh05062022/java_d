"""Which family of child-table aggregates carries the signal, and which survives generation?

S8 (multi-statistic aggregation) and S10 (exact cardinality) are bets that the denoiser is
missing specific *kinds* of neighbourhood information: how many children a row has, and how
spread out they are, as opposed to their mean. Each bet costs 20-40 GPU-hours to test by
training a generator. This tests the premise for free, on samples that already exist.

Two questions, both answered per feature family:

1. **Does the family carry signal at all?** XGBoost on real data, restricted to that family
   (plus the row's own columns, which are always available to the model).
2. **Does generation preserve the family's association with the label?** Point-biserial
   correlation between each feature and the label, computed inside real data and inside each
   synthetic sample, compared. A family whose correlations survive is one the denoiser already
   reproduces; a family whose correlations collapse is one an architecture change has to fix.
   The headline is the family's synthetic correlations projected onto the real directions,
   sum(sign(corr_real) * corr_syn) / sum(|corr_real|): 1 = reproduced, 0 = gone, negative =
   reversed, and features with a weak real association weigh little. Until 5 Oct 2026 this was a
   mean of per-feature |corr_syn| / |corr_real| ratios, which counted reversals as preserved and
   let barely-correlated features dominate; the signed mean of ratios is kept as `mean_ratio`.

    conda run --no-capture-output -n tabsyn python -m downstream.diagnose_aggregates
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import (
    REPO_ROOT, align_feature_dtypes, coerce_discrete, grdm_run_dir, load_db, load_domains,
    load_meta, load_synthetic_db,
)
from downstream.evaluate import compute_metrics, fit_predict
from downstream.run import DATASETS
from downstream.tasks import TASKS

# Feature families, matched on the names `tasks.py` builds. The order matters: a name is
# assigned to the first family it matches, so "trans_amount_sum" lands in mean/sum rather than
# in the catch-all.
FAMILIES = [
    ("count", lambda c: c.endswith("_n")),
    ("mean/sum", lambda c: "_mean" in c or "_sum" in c),
    ("spread/extremes", lambda c: any(k in c for k in ("_std", "_min", "_max", "_last", "_span"))),
    ("code shares", lambda c: c.endswith("_frac")),
]


def classify(columns):
    """Split feature names into families; everything unmatched is the row's own columns."""
    out = {name: [] for name, _ in FAMILIES}
    own = []
    for col in columns:
        for name, test in FAMILIES:
            if test(col):
                out[name].append(col)
                break
        else:
            own.append(col)
    return own, out


def point_biserial(X, y):
    """Correlation of every numeric column with a binary label, NaNs dropped pairwise."""
    y = np.asarray(y, dtype=float)
    out = {}
    for col in X.columns:
        v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(v)
        if mask.sum() < 30:
            continue
        vv, yy = v[mask], y[mask]
        if vv.std() == 0 or yy.std() == 0:
            out[col] = 0.0
        else:
            out[col] = float(np.corrcoef(vv, yy)[0, 1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*",
                        default=["berka_split", "berka_split_s2", "berka_split_s3",
                                 "berka_split_s5"])
    parser.add_argument("--tasks", nargs="*",
                        default=["berka_loan_default", "berka_card_ownership"])
    parser.add_argument("--xgb-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "downstream", "results",
                                                      "aggregate_families.json"))
    args = parser.parse_args()

    base = DATASETS["berka_split"]
    meta = load_meta(base["real_train"])
    domains = load_domains(base["domains"], list(meta["tables"]))
    real_tables = coerce_discrete(load_db(base["real_train"]), domains)
    test_tables = coerce_discrete(load_db(base["test"]), domains)

    report = {}
    for task_name in args.tasks:
        task = TASKS[task_name]
        X_real, y_real, cat_spec = task.build(real_tables, domains)
        X_test, y_test, _ = task.build(test_tables, domains)
        X_test = X_test.reindex(columns=list(X_real.columns))
        (X_real, X_test), _ = align_feature_dtypes([X_real, X_test], cat_spec)
        own, fams = classify(list(X_real.columns))

        print(f"\n=== {task_name}   {X_real.shape[1]} features, "
              f"{int(y_real.sum())}/{len(y_real)} positives")
        print(f"  own-row: {len(own)}  " +
              "  ".join(f"{k}: {len(v)}" for k, v in fams.items()))

        # --- 1. where the signal lives, on real data ------------------------------------
        print(f"\n  {'feature set':26s} {'cols':>5s}  real AUROC")
        signal = {}
        subsets = [("own row only", own)] + [
            (f"own + {name}", own + cols) for name, cols in fams.items() if cols
        ] + [("everything", list(X_real.columns))]
        for label, cols in subsets:
            scores = []
            for seed in args.xgb_seeds:
                pred, score = fit_predict(task.kind, X_real[cols], y_real, X_test[cols], seed,
                                          n_classes=task.n_classes)
                scores.append(compute_metrics(task.kind, y_test, pred, score,
                                              n_classes=task.n_classes)[task.primary])
            signal[label] = (float(np.mean(scores)), float(np.std(scores)))
            print(f"  {label:26s} {len(cols):5d}  {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # --- 2. which families survive generation ---------------------------------------
        corr_real = point_biserial(X_real, y_real)
        preserved = {}
        for variant in args.variants:
            cfg = DATASETS[variant]
            syn = coerce_discrete(
                load_synthetic_db(grdm_run_dir(cfg["grdm_dataset"], 0,
                                               variant=cfg.get("variant", "")),
                                  list(meta["tables"])), domains)
            X_syn, y_syn, _ = task.build(syn, domains)
            X_syn = X_syn.reindex(columns=list(X_real.columns))
            (X_syn,), _ = align_feature_dtypes([X_syn], cat_spec)
            corr_syn = point_biserial(X_syn, y_syn)

            row = {}
            for name, cols in list(fams.items()) + [("own row", own)]:
                shared = [c for c in cols if c in corr_real and c in corr_syn]
                if not shared:
                    continue
                r = np.array([corr_real[c] for c in shared])
                s = np.array([corr_syn[c] for c in shared])
                keep = np.abs(r) > 0.02  # families with no real association say nothing
                row[name] = dict(
                    n=len(shared),
                    real_mean_abs=float(np.abs(r).mean()),
                    syn_mean_abs=float(np.abs(s).mean()),
                    # share of the real association reproduced, on the features that have one:
                    # synthetic correlations projected onto the real directions
                    retained=float((np.sign(r[keep]) * s[keep]).sum() / np.abs(r[keep]).sum())
                    if keep.any() else None,
                    mean_ratio=float((s[keep] / r[keep]).mean()) if keep.any() else None,
                    n_reversed=int((s[keep] * r[keep] < 0).sum()),
                    n_kept=int(keep.sum()),
                )
            preserved[variant] = row

        print(f"\n  label association retained by family "
              f"(synthetic correlation along the real direction, features with |corr_real| > 0.02, "
              f"seed 0)")
        fam_names = [n for n, _ in FAMILIES] + ["own row"]
        print(f"  {'variant':22s}" + "".join(f"{n:>18s}" for n in fam_names))
        for variant, row in preserved.items():
            cells = ""
            for n in fam_names:
                v = row.get(n, {}).get("retained")
                cells += f"{'-':>18s}" if v is None else f"{v:>17.2f}x"
            print(f"  {variant:22s}{cells}")

        report[task_name] = dict(signal=signal, preserved=preserved,
                                 family_sizes={"own row": len(own),
                                               **{k: len(v) for k, v in fams.items()}})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
