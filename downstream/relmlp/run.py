"""DeepSets downstream probe -- the no-hand-aggregates counterpart to `python -m downstream.run`.

    conda run --no-capture-output -n tabsyn python -m downstream.relmlp.run --dataset berka_split

Same real held-out test set, same metric definitions (imported from downstream.evaluate), same
regime semantics as downstream/run.py. What changes: the feature builder emits raw parent+child
rows instead of hand aggregates (downstream/relmlp/features.py), and the estimator is a small
permutation-invariant net (downstream/relmlp/model.py) trained for a fixed budget and repeated
over `--train-seeds` to expose optimisation variance.

Results land in downstream/results_relmlp/ so nothing here collides with the XGBoost run.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from downstream.data import REPO_ROOT, load_domains, load_meta
from downstream.evaluate import bootstrap_ci, compute_metrics
from downstream.relmlp.features import BUILDERS, concat_rel, subsample_rel
from downstream.relmlp.train import EPOCHS, LR, WEIGHT_DECAY, fit_predict, prepare
from downstream.run import DATASETS, load_syn_tables, load_tables

DEFAULT_REGIMES = ["real", "synth"]
PRIMARY = "roc_auc"
N_CLASSES = 2
BOOTSTRAP_MAX_TEST_ROWS = 20000


def _metrics_record(cfg_name, task_name, regime, grdm_seed, train_seed, n_train, y_true,
                    y_pred, y_score, seconds, with_ci):
    m = compute_metrics("binclass", y_true, y_pred, y_score, n_classes=N_CLASSES)
    rec = dict(dataset=cfg_name, task=task_name, kind="binclass", regime=regime,
               grdm_seed=grdm_seed, train_seed=train_seed, n_train=int(n_train),
               metrics=m, seconds=round(seconds, 1))
    if with_ci:
        lo, hi = bootstrap_ci("binclass", y_true, y_pred, y_score, PRIMARY, n_classes=N_CLASSES)
        rec["ci95"] = {PRIMARY: [lo, hi]}
    return rec, m


def run_regime(records, cfg_name, task_name, regime, train_rel, test_rel, grdm_seed,
               train_seeds, args):
    tr_b, te_b, spec = prepare(train_rel, test_rel, args.device)
    y_true = test_rel.y.astype("int64")
    do_ci = len(y_true) <= BOOTSTRAP_MAX_TEST_ROWS
    scores = []
    for i, ts in enumerate(train_seeds):
        t0 = time.time()
        y_pred, y_score = fit_predict(tr_b, te_b, spec, ts, args.epochs, args.lr,
                                      args.weight_decay, args.device)
        rec, m = _metrics_record(cfg_name, task_name, regime, grdm_seed, ts, len(train_rel),
                                 y_true, y_pred, y_score, time.time() - t0,
                                 with_ci=do_ci and i == 0)
        records.append(rec)
        scores.append(y_score)
        print(f"  {regime:9s} grdm={str(grdm_seed):>4s} ts={ts}  n={len(train_rel):5d}  "
              f"{PRIMARY}={m[PRIMARY]:.4f}  ({rec['seconds']}s)")

    ens = np.mean(scores, axis=0)
    rec, m = _metrics_record(cfg_name, task_name, regime, grdm_seed, "ensemble", len(train_rel),
                             y_true, (ens >= 0.5).astype("int64"), ens, 0.0, with_ci=do_ci)
    records.append(rec)
    print(f"  {regime:9s} grdm={str(grdm_seed):>4s} ens    n={len(train_rel):5d}  "
          f"{PRIMARY}={m[PRIMARY]:.4f}  <- mean of {len(train_seeds)} seeds")


def run_task(task_name, cfg, args, domains, meta, real_tables, test_tables):
    build = BUILDERS[task_name]
    print(f"\n=== {task_name}  (DeepSets probe)")
    real_rel = build(real_tables, domains)
    test_rel = build(test_tables, domains)
    n_real = len(real_rel)
    pos_tr = int(real_rel.y.sum())
    pos_te = int(test_rel.y.sum())
    n_feat = real_rel.parent_num.shape[1] + real_rel.parent_cat.shape[1]
    print(f"  parent_features={n_feat}  child_tables={list(real_rel.children)}  "
          f"n_real_train={n_real}  n_test={len(test_rel)}")
    print(f"  positives: train {pos_tr}/{n_real}  test {pos_te}/{len(test_rel)}")

    records = []
    if "real" in args.regimes:
        run_regime(records, cfg["_name"], task_name, "real", real_rel, test_rel, None,
                   args.train_seeds, args)

    need = [r for r in args.regimes if r != "real"]
    for grdm_seed in (args.grdm_seeds if need else []):
        syn_tables = load_syn_tables(cfg, grdm_seed, domains, meta)
        syn_rel = build(syn_tables, domains)
        del syn_tables
        ood = 0
        for cb in syn_rel.children.values():
            for j, c in enumerate(cb.cat_cards):
                ood += int(((cb.cat[:, j] < 0) | (cb.cat[:, j] >= c)).sum())
        print(f"  [seed {grdm_seed}] synthetic parents={len(syn_rel)}  "
              f"out-of-domain child codes={ood}")
        for regime in need:
            if regime == "synth":
                rng = np.random.default_rng(1000 * (grdm_seed + 1))
                train_rel = subsample_rel(syn_rel, n_real, rng)
            elif regime == "mixed":
                train_rel = concat_rel(real_rel, syn_rel)
            else:
                raise ValueError(f"unsupported regime {regime!r} for the DeepSets probe")
            run_regime(records, cfg["_name"], task_name, regime, train_rel, test_rel,
                       grdm_seed, args.train_seeds, args)
        del syn_rel
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--tasks", nargs="*", default=["berka_loan_default", "berka_card_ownership"])
    p.add_argument("--grdm-seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--train-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--regimes", nargs="*", default=DEFAULT_REGIMES,
                   choices=["real", "synth", "mixed"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "downstream", "results_relmlp"))
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    cfg = dict(DATASETS[args.dataset])
    cfg["_name"] = args.dataset
    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))
    real_tables = load_tables(cfg["real_train"], domains, meta, "real train")
    test_tables = load_tables(cfg["test"], domains, meta, "test")

    print(f"dataset={args.dataset}  leakage_caveat={cfg['leakage']}  device={args.device}")
    print(f"regimes={args.regimes}  grdm_seeds={args.grdm_seeds}  train_seeds={args.train_seeds}")
    print(f"epochs={args.epochs} lr={args.lr} weight_decay={args.weight_decay}")

    os.makedirs(args.out_dir, exist_ok=True)
    for task_name in args.tasks:
        records = run_task(task_name, cfg, args, domains, meta, real_tables, test_tables)
        out = os.path.join(args.out_dir, f"{args.dataset}__{task_name}{args.out_suffix}.json")
        with open(out, "w") as f:
            json.dump({"dataset": args.dataset, "leakage_caveat": cfg["leakage"],
                       "probe": "relmlp", "records": records}, f, indent=2)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
