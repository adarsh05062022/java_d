"""Run the real / synthetic-only / mixed downstream comparison.

    conda run --no-capture-output -n tabsyn python -m downstream.run --dataset california
    conda run --no-capture-output -n tabsyn python -m downstream.run --dataset berka
    conda run --no-capture-output -n tabsyn python -m downstream.run --dataset berka_split

Every regime is evaluated on the same real held-out test set. The synthetic-only regime is
subsampled to the real training row count so that `real` vs `synth` compares data quality rather
than data quantity; the mixed sweep then varies quantity on top of the full real set.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

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
from downstream.evaluate import bootstrap_ci, compute_metrics, fit_predict
from downstream.tasks import TASKS

CLAVA_ROOT = "/scratch/s25017/ClavaDDPM/ClavaDDPM"

DATASETS = {
    "california": dict(
        real_train=os.path.join(REPO_ROOT, "data/california/train"),
        test=os.path.join(CLAVA_ROOT, "complex_data/california/test"),
        domains=os.path.join(REPO_ROOT, "data/california/train"),
        grdm_dataset="california",
        tasks=["california_inctot", "california_empstat"],
        leakage=False,
    ),
    # Interim Berka: the generator trained on the whole database, test rows included.
    "berka": dict(
        real_train=os.path.join(REPO_ROOT, "data/berka/split_train"),
        test=os.path.join(REPO_ROOT, "data/berka/split_test"),
        domains=os.path.join(REPO_ROOT, "data/berka/preprocessed"),
        grdm_dataset="berka",
        tasks=["berka_loan_default", "berka_card_ownership"],
        leakage=True,
    ),
    # Clean Berka: same split, but the generator only ever saw split_train.
    "berka_split": dict(
        real_train=os.path.join(REPO_ROOT, "data/berka/split_train"),
        test=os.path.join(REPO_ROOT, "data/berka/split_test"),
        domains=os.path.join(REPO_ROOT, "data/berka/preprocessed"),
        grdm_dataset="berka_split",
        tasks=["berka_loan_default", "berka_card_ownership"],
        leakage=False,
    ),
}

# Fix experiments. Each reuses a base dataset's real/test/domain paths and changes exactly one
# thing, so any delta against the base is attributable to that fix alone.
#   variant      -> re-decoded copy of an existing sample (no retrain)
#   grdm_dataset -> a separate workspace produced by its own training run
FIXES = {
    "berka_split_s1": dict(base="berka_split", variant="_decoding_match"),
    # R1 = post-hoc constraint repair (dep_fidelity/run_repair.py). Like S1 it is a
    # re-written copy of the baseline sample rather than a retrained generator, so any
    # delta here is attributable to constraint enforcement alone.
    "berka_split_repair": dict(base="berka_split", variant="_repaired"),
    # Same temporal and logical repairs, but the ledger operator rewrites only each
    # account's final balance instead of deriving the whole series. It satisfies the
    # account-level check identically while leaving the balance marginal intact, which
    # separates "constraint repair" from "the balance column was replaced".
    "berka_split_repair_finalonly": dict(base="berka_split", variant="_repaired_finalonly"),
    # R2 = Tier 3. Trained on `data/berka/split_train_reparam`, where a child event's date
    # is log1p(lag from its parent account), then decoded back to absolute dates by
    # `dep_fidelity.decode_run`. Everything else matches the baseline config, so the delta
    # against `berka_split` is attributable to the representation change alone.
    "berka_split_reparam": dict(base="berka_split", grdm_dataset="berka_split_reparam",
                                variant="_decoded"),
    "berka_split_s2": dict(base="berka_split", grdm_dataset="berka_split_s2"),
    "berka_split_s3": dict(base="berka_split", grdm_dataset="berka_split_s3"),
    "berka_split_s4": dict(base="berka_split", grdm_dataset="berka_split_s4"),
    # S5 = S2 + S3 together: deeper conditioning plus balanced seeding. S3 also cuts the `trans`
    # seed share, which is what made the 2-hop fan-out expensive in the first place.
    "berka_split_s5": dict(base="berka_split", grdm_dataset="berka_split_s5"),
    "california_s5": dict(base="california", grdm_dataset="california_s5"),
    "california_s2": dict(base="california", grdm_dataset="california_s2"),
    "california_s3": dict(base="california", grdm_dataset="california_s3"),
    "california_s4": dict(base="california", grdm_dataset="california_s4"),
    "california_s7": dict(base="california", grdm_dataset="california_s7"),
    # S6/S7 push conditioning depth to 3 hops.
    "berka_split_s6": dict(base="berka_split", grdm_dataset="berka_split_s6"),
    "berka_split_s7": dict(base="berka_split", grdm_dataset="berka_split_s7"),
    # S8-S11 change the denoiser architecture, each layered on S5.
    "berka_split_s8": dict(base="berka_split", grdm_dataset="berka_split_s8"),
    "berka_split_s9": dict(base="berka_split", grdm_dataset="berka_split_s9"),
    "berka_split_s10": dict(base="berka_split", grdm_dataset="berka_split_s10"),
    "berka_split_s11": dict(base="berka_split", grdm_dataset="berka_split_s11"),
    "berka_split_s12": dict(base="berka_split", grdm_dataset="berka_split_s12"),
    "berka_split_s13": dict(base="berka_split", grdm_dataset="berka_split_s13"),
    "california_s12": dict(base="california", grdm_dataset="california_s12"),
    "california_s13": dict(base="california", grdm_dataset="california_s13"),
    # Combined-direction sweep (2026-09): reparam date representation x S12 architecture,
    # a lighter reparam x degree-features-only, and a wider hop-2 fan-out on S12. The two
    # reparam variants are trained on split_train_reparam, so they read the _decoded copy.
    "berka_split_reparam_s12": dict(base="berka_split", grdm_dataset="berka_split_reparam_s12", variant="_decoded"),
    "berka_split_reparam_s10": dict(base="berka_split", grdm_dataset="berka_split_reparam_s10", variant="_decoded"),
    "berka_split_s12_wide": dict(base="berka_split", grdm_dataset="berka_split_s12_wide"),
    # Discrete/multinomial categorical diffusion (the principled alternative to S1/S4),
    # plus its California regression guard.
    "berka_split_discrete": dict(base="berka_split", grdm_dataset="berka_split_discrete"),
    "california_discrete": dict(base="california", grdm_dataset="california_discrete"),
    # Task 6: explicit relational aggregates (agg_trans_n, agg_trans_balance_mean,
    # agg_trans_balance_last, agg_trans_amount_mean, agg_order_n, agg_order_amount_sum) added
    # as columns on the account table's domain, otherwise identical to berka_split. Tests
    # whether exposing the child-aggregate statistics GRDM's 1-hop GNN conditioning can't see
    # on its own recovers the signal Task 5 found missing.
    "berka_split_agg": dict(base="berka_split", grdm_dataset="berka_split_agg"),
}

for _name, _spec in FIXES.items():
    DATASETS[_name] = {**DATASETS[_spec["base"]], **{k: v for k, v in _spec.items() if k != "base"}}

# (name, real fraction, synthetic rows as a fraction of the real training size)
#
# `mixed_*` keep the whole real training set and *add* synthetic rows on top, so they vary
# quantity. `real_50` / `blend_50` instead hold the total row count fixed at the real training
# size and vary composition: `real_50` trains on a random half of the real rows, `blend_50`
# replaces the dropped half with the same number of synthetic rows. Comparing those two
# isolates whether synthetic data can *substitute* for real data you no longer have.
REGIMES = [
    ("real", 1.0, 0.0),
    ("synth", 0.0, 1.0),
    ("mixed_25", 1.0, 0.25),
    ("mixed_50", 1.0, 0.50),
    ("mixed", 1.0, 1.00),
    ("real_50", 0.5, 0.0),
    ("blend_50", 0.5, 0.5),
    ("real_25", 0.25, 0.0),
    ("blend_25", 0.25, 0.75),
    ("real_75", 0.75, 0.0),
    ("blend_75", 0.75, 0.25),
]

BOOTSTRAP_MAX_TEST_ROWS = 20000


def load_tables(path, domains, meta, label):
    tables = load_db(path)
    assert_fk_integrity(tables, meta, label=label)
    return coerce_discrete(tables, domains)


def load_syn_tables(dataset_cfg, seed, domains, meta):
    run_dir = grdm_run_dir(
        dataset_cfg["grdm_dataset"], seed, variant=dataset_cfg.get("variant", "")
    )
    tables = load_synthetic_db(run_dir, list(meta["tables"]))
    assert_fk_integrity(tables, meta, label=f"synthetic seed {seed}")
    return coerce_discrete(tables, domains)


def align_columns(X, reference_columns, label):
    missing = set(reference_columns) - set(X.columns)
    extra = set(X.columns) - set(reference_columns)
    if missing or extra:
        print(f"  [warn] {label}: {len(missing)} missing / {len(extra)} extra columns; reindexing")
    return X.reindex(columns=reference_columns)


def sample_rows(X, y, n, rng):
    if n >= len(X):
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx], y.iloc[idx] if hasattr(y, "iloc") else y[idx]


def run_task(task, cfg, grdm_seeds, xgb_seeds, domains, meta, real_tables, test_tables,
             regimes=None):
    active = [r for r in REGIMES if regimes is None or r[0] in regimes]
    cfg_name = cfg["_name"]
    print(f"\n=== {task.name} ({task.kind}) target={task.target}")

    X_real, y_real, cat_spec = task.build(real_tables, domains)
    X_test, y_test, _ = task.build(test_tables, domains)

    if task.target_col:
        assert task.target_col not in X_real.columns, f"target {task.target_col} leaked into X"
    if task.name == "berka_card_ownership":
        # `card` contributes the label only: `issued` and `card_type` exist for positives alone.
        assert not [c for c in X_real.columns if c.startswith("card")], "card table leaked into X"

    X_test = align_columns(X_test, list(X_real.columns), "test")
    (X_real, X_test), ood = align_feature_dtypes([X_real, X_test], cat_spec)
    print(f"  features={X_real.shape[1]}  n_real_train={len(X_real)}  n_test={len(X_test)}")
    if task.kind == "binclass":
        print(f"  positives: train {int(y_real.sum())}/{len(y_real)}  test {int(y_test.sum())}/{len(y_test)}")

    do_bootstrap = len(X_test) <= BOOTSTRAP_MAX_TEST_ROWS
    records = []

    def evaluate(regime, X_tr, y_tr, grdm_seed, xgb_seed, n_real, n_syn):
        t0 = time.time()
        y_pred, y_score = fit_predict(
            task.kind, X_tr, y_tr, X_test, xgb_seed, n_classes=task.n_classes
        )
        metrics = compute_metrics(task.kind, y_test, y_pred, y_score, n_classes=task.n_classes)
        rec = dict(
            dataset=cfg_name,
            task=task.name,
            kind=task.kind,
            regime=regime,
            grdm_seed=grdm_seed,
            xgb_seed=xgb_seed,
            n_train=len(X_tr),
            n_train_real=n_real,
            n_train_syn=n_syn,
            metrics=metrics,
            seconds=round(time.time() - t0, 1),
        )
        if do_bootstrap and xgb_seed == xgb_seeds[0]:
            lo, hi = bootstrap_ci(
                task.kind, y_test, y_pred, y_score, task.primary, n_classes=task.n_classes
            )
            rec["ci95"] = {task.primary: [lo, hi]}
        records.append(rec)
        print(
            f"  {regime:9s} grdm={str(grdm_seed):4s} xgb={xgb_seed}  "
            f"n={len(X_tr):7d}  {task.primary}={metrics[task.primary]:.4f}  ({rec['seconds']}s)"
        )

    n_real = len(X_real)

    if any(r[0] == "real" for r in active):
        for xgb_seed in xgb_seeds:
            evaluate("real", X_real, y_real, None, xgb_seed, n_real, 0)

    if not [r for r in active if r[0] != "real"]:
        return records

    for grdm_seed in grdm_seeds:
        syn_tables = load_syn_tables(cfg, grdm_seed, domains, meta)
        X_syn, y_syn, _ = task.build(syn_tables, domains)
        X_syn = align_columns(X_syn, list(X_real.columns), f"synthetic seed {grdm_seed}")
        (X_syn,), ood_syn = align_feature_dtypes([X_syn], cat_spec)
        out_of_domain = sum(ood_syn.values())
        print(
            f"  [seed {grdm_seed}] synthetic rows={len(X_syn)}  "
            f"out-of-domain categorical cells={int(out_of_domain)}"
        )
        del syn_tables

        for regime, real_frac, syn_frac in active:
            if regime == "real":
                continue
            n_syn_target = int(round(syn_frac * n_real))
            n_real_target = int(round(real_frac * n_real))
            for xgb_seed in xgb_seeds:
                rng = np.random.default_rng(1000 * (grdm_seed + 1) + xgb_seed)
                Xs, ys = sample_rows(X_syn, y_syn, n_syn_target, rng)
                if real_frac >= 1.0:
                    Xr, yr = X_real, y_real
                elif real_frac <= 0.0:
                    Xr, yr = None, None
                else:
                    Xr, yr = sample_rows(X_real, y_real, n_real_target, rng)
                if Xr is None:
                    X_tr, y_tr, nr = Xs, ys, 0
                else:
                    X_tr = pd.concat([Xr, Xs], ignore_index=True)
                    y_tr = pd.concat(
                        [pd.Series(np.asarray(yr)), pd.Series(np.asarray(ys))],
                        ignore_index=True,
                    )
                    nr = len(Xr)
                evaluate(regime, X_tr, y_tr, grdm_seed, xgb_seed, nr, len(Xs))
        del X_syn

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--grdm-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--xgb-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "downstream", "results"))
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=None,
        choices=[r[0] for r in REGIMES],
        help="Subset of training regimes to score. Default: all of them. real+synth is the\n"
             "load-bearing comparison; the mixed sweep costs ~3x more.",
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Appended to the results filename, so a partial-regime run does not "
             "overwrite a full one.",
    )
    args = parser.parse_args()

    cfg = dict(DATASETS[args.dataset])
    cfg["_name"] = args.dataset
    xgb_seeds = args.xgb_seeds or ([0, 1, 2, 3, 4] if "berka" in args.dataset else [0, 1, 2])

    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))

    real_tables = load_tables(cfg["real_train"], domains, meta, "real train")
    test_tables = load_tables(cfg["test"], domains, meta, "test")

    root = list(meta["tables"])[0]
    for table in meta["tables"]:
        key = f"{table}_id"
        train_ids = set(real_tables[table][key])
        test_ids = set(test_tables[table][key])
        if table == root and args.dataset.startswith("berka"):
            continue  # district is a dimension table, deliberately shared
        if train_ids & test_ids:
            raise AssertionError(f"{table}: {len(train_ids & test_ids)} ids in both train and test")

    print(f"dataset={args.dataset}  leakage_caveat={cfg['leakage']}")
    print(f"real_train={cfg['real_train']}\ntest={cfg['test']}")

    os.makedirs(args.out_dir, exist_ok=True)
    for task_name in args.tasks or cfg["tasks"]:
        records = run_task(
            TASKS[task_name], cfg, args.grdm_seeds, xgb_seeds, domains, meta, real_tables,
            test_tables, regimes=args.regimes,
        )
        out = os.path.join(args.out_dir, f"{args.dataset}__{task_name}{args.out_suffix}.json")
        with open(out, "w") as f:
            json.dump(
                {"dataset": args.dataset, "leakage_caveat": cfg["leakage"], "records": records},
                f,
                indent=2,
            )
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
