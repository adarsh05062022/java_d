"""Cross-domain transfer: train on real or synthetic, test on real or synthetic.

    conda run --no-capture-output -n tabsyn python -m downstream.crossdomain --dataset berka_split

Four cells per task, sharing preprocessing, feature builders, XGBoost hyperparameters and metric
code with `downstream.run` -- only the *source* of the training and test rows changes:

    real  -> real     the standing supervised baseline
    synth -> synth    is there any learnable signal inside the synthetic database at all?
    real  -> synth    does a real-learned rule still hold on synthetic rows?
    synth -> real     THE transfer question: do synthetic-learned rules work on the real database?

`synth -> synth` needs a held-out synthetic test set. Rather than slicing the feature matrix --
which would let two dispositions of the same synthetic account straddle the split and inflate the
score -- the synthetic database is cut with the very same account-level, outcome-stratified
protocol used for the real one (`downstream.splits.berka_account_split`, 25% test, seed 0). A
synthetic test set is therefore a genuinely held-out relational database, built exactly the way
the real one was.

Training sets are subsampled to a common row count within each GRDM seed, so no cell wins on
quantity: `synth -> real` and `real -> real` see the same number of training rows.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import (
    REPO_ROOT,
    align_feature_dtypes,
    assert_fk_integrity,
    coerce_discrete,
    grdm_run_dir,
    load_db,
    load_domains,
    load_meta,
    load_synthetic_db,
)
from downstream.evaluate import compute_metrics, fit_predict
from downstream.run import DATASETS
from downstream.splits import BERKA_SPLIT_SEED, BERKA_TEST_FRAC, berka_account_split, subset_berka
from downstream.tasks import TASKS

# (train source, test source)
CELLS = [("real", "real"), ("synth", "synth"), ("real", "synth"), ("synth", "real")]


def split_synthetic(tables, meta):
    """Cut a synthetic Berka database with the real protocol: account-level, outcome-stratified."""
    train_ids, test_ids = berka_account_split(
        tables, test_frac=BERKA_TEST_FRAC, seed=BERKA_SPLIT_SEED
    )
    syn_train, syn_test = subset_berka(tables, train_ids), subset_berka(tables, test_ids)
    assert_fk_integrity(syn_train, meta, label="synthetic train")
    assert_fk_integrity(syn_test, meta, label="synthetic test")
    return syn_train, syn_test


def subsample(X, y, n, rng):
    if n >= len(X):
        return X, y
    idx = np.sort(rng.choice(len(X), size=n, replace=False))
    return X.iloc[idx], y.iloc[idx]


def run_task(task, cfg, grdm_seeds, xgb_seeds, domains, meta, real_train, real_test):
    print(f"\n{'=' * 78}\n=== {task.name} ({task.kind}) target={task.target}\n{'=' * 78}")

    X_real_tr, y_real_tr, cat_spec = task.build(real_train, domains)
    X_real_te, y_real_te, _ = task.build(real_test, domains)
    reference_columns = list(X_real_tr.columns)
    records = []

    for grdm_seed in grdm_seeds:
        run_dir = grdm_run_dir(cfg["grdm_dataset"], grdm_seed, variant=cfg.get("variant", ""))
        syn = coerce_discrete(load_synthetic_db(run_dir, list(meta["tables"])), domains)
        assert_fk_integrity(syn, meta, label=f"synthetic seed {grdm_seed}")
        syn_train, syn_test = split_synthetic(syn, meta)
        del syn

        X_syn_tr, y_syn_tr, _ = task.build(syn_train, domains)
        X_syn_te, y_syn_te, _ = task.build(syn_test, domains)

        frames = [X_real_tr, X_real_te, X_syn_tr, X_syn_te]
        frames = [f.reindex(columns=reference_columns) for f in frames]
        (Xr_tr, Xr_te, Xs_tr, Xs_te), _ = align_feature_dtypes(frames, cat_spec)

        # A cell must not win on quantity: both training sets are cut to the smaller of the two.
        n_train = min(len(Xr_tr), len(Xs_tr))
        print(
            f"\n[seed {grdm_seed}] train rows real={len(Xr_tr)} synth={len(Xs_tr)} "
            f"-> matched at {n_train}   test rows real={len(Xr_te)} synth={len(Xs_te)}"
        )
        print(
            f"           positives  real train {int(y_real_tr.sum())}/{len(y_real_tr)}  "
            f"synth train {int(y_syn_tr.sum())}/{len(y_syn_tr)}  "
            f"real test {int(y_real_te.sum())}/{len(y_real_te)}  "
            f"synth test {int(y_syn_te.sum())}/{len(y_syn_te)}"
        )

        pools = {
            "real": ((Xr_tr, y_real_tr), (Xr_te, y_real_te)),
            "synth": ((Xs_tr, y_syn_tr), (Xs_te, y_syn_te)),
        }

        for train_src, test_src in CELLS:
            (X_pool, y_pool), _ = pools[train_src]
            _, (X_te, y_te) = pools[test_src]
            if len(np.unique(y_te)) < 2:
                print(f"  {train_src:5s} -> {test_src:5s}  SKIPPED: test set has one class only")
                continue
            for xgb_seed in xgb_seeds:
                rng = np.random.default_rng(1000 * (grdm_seed + 1) + xgb_seed)
                X_tr, y_tr = subsample(X_pool, y_pool, n_train, rng)
                t0 = time.time()
                y_pred, y_score = fit_predict(
                    task.kind, X_tr, y_tr, X_te, xgb_seed, n_classes=task.n_classes
                )
                metrics = compute_metrics(
                    task.kind, y_te, y_pred, y_score, n_classes=task.n_classes
                )
                records.append(
                    dict(
                        task=task.name,
                        train_on=train_src,
                        test_on=test_src,
                        cell=f"{train_src}->{test_src}",
                        grdm_seed=grdm_seed,
                        xgb_seed=xgb_seed,
                        n_train=len(X_tr),
                        n_train_pos=int(np.asarray(y_tr).sum()),
                        n_test=len(X_te),
                        n_test_pos=int(np.asarray(y_te).sum()),
                        metrics=metrics,
                        seconds=round(time.time() - t0, 1),
                    )
                )
                print(
                    f"  {train_src:5s} -> {test_src:5s}  xgb={xgb_seed}  n={len(X_tr):6d}  "
                    f"{task.primary}={metrics[task.primary]:.4f}  "
                    f"bal_acc={metrics['balanced_accuracy']:.4f}  "
                    f"f1_macro={metrics['f1_macro']:.4f}  ({records[-1]['seconds']}s)"
                )
        del X_syn_tr, X_syn_te, syn_train, syn_test

    return records


def matrix_table(records, metric):
    """The 2x2: rows are the training source, columns the test source."""
    lines = [f"| train \\ test | real | synthetic |", "|---|---|---|"]
    for train_src in ("real", "synth"):
        cells = []
        for test_src in ("real", "synth"):
            vals = [
                r["metrics"][metric]
                for r in records
                if r["train_on"] == train_src and r["test_on"] == test_src
            ]
            cells.append(f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if vals else "n/a")
        lines.append(f"| **{train_src}** | {cells[0]} | {cells[1]} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="berka_split", choices=sorted(DATASETS))
    parser.add_argument(
        "--tasks", nargs="*", default=["berka_card_ownership", "berka_loan_default"]
    )
    parser.add_argument("--grdm-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--xgb-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "downstream", "results"))
    args = parser.parse_args()

    cfg = dict(DATASETS[args.dataset])
    cfg["_name"] = args.dataset

    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))
    real_train = coerce_discrete(load_db(cfg["real_train"]), domains)
    real_test = coerce_discrete(load_db(cfg["test"]), domains)
    assert_fk_integrity(real_train, meta, label="real train")
    assert_fk_integrity(real_test, meta, label="real test")

    print(f"dataset={args.dataset}  grdm_dataset={cfg['grdm_dataset']}")
    print(f"synthetic split: account-level, {BERKA_TEST_FRAC:.0%} test, seed {BERKA_SPLIT_SEED} "
          f"(identical protocol to the real split)")

    os.makedirs(args.out_dir, exist_ok=True)
    for task_name in args.tasks:
        task = TASKS[task_name]
        records = run_task(
            task, cfg, args.grdm_seeds, args.xgb_seeds, domains, meta, real_train, real_test
        )
        out = os.path.join(args.out_dir, f"{args.dataset}__{task_name}_crossdomain.json")
        with open(out, "w") as f:
            json.dump({"dataset": args.dataset, "task": task_name, "records": records}, f, indent=2)
        print(f"\n### {task_name} — {task.primary} (rows = trained on, columns = tested on)")
        print("\n".join(matrix_table(records, task.primary)))
        for m in ("balanced_accuracy", "f1_macro"):
            print(f"\n### {task_name} — {m}")
            print("\n".join(matrix_table(records, m)))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
