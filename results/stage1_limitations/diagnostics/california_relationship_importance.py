"""How much of each California downstream task depends on relational information?

Feature-set ablation with the harness's own feature builder and XGBoost settings
(downstream.tasks._california_matrix, downstream.evaluate.XGB_PARAMS), always tested on the real
TEST split:
    own        individual columns only
    own+hh     + the household's columns (1-hop parent)
    full       + the harness's household rollups over co-members (hh_size, hh_age_*, ...)
Training sources: real train, baseline GRDM seed 0, aggregate-conditioned GRDM seed 0.

Features are built on the complete database first (so rollups see every member), then
N_TRAIN individuals are subsampled for fitting. This is a diagnostic ablation, not the
11-regime protocol: absolute numbers differ slightly from the harness's full-size fits.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 RDBDIFF_XGB_JOBS=8 \
      conda run -n tabsyn python california_relationship_importance.py
"""

import time

import numpy as np

import common as C
from downstream import data as D
from downstream import evaluate as E
from downstream import tasks as T

N_TRAIN = 300_000
SEED = 0
TASKS = {"california_inctot": ("INCTOT", "regression", "r2", 0),
         "california_empstat": ("EMPSTAT", "multiclass", "f1_macro", 4)}
SOURCES = {
    "real_train": None,
    "baseline_seed0": C.CAL_CONFIGS["baseline"].format(s=0),
    "agg_seed0": C.CAL_CONFIGS["agg"].format(s=0),
}


def matrices(tb, dom, target):
    X, y, cat = T._california_matrix(tb, dom, target)
    if target == "EMPSTAT":
        y = np.rint(y.astype(float)).astype(int)
    return X, np.asarray(y, dtype=float if target == "INCTOT" else int), cat


def main():
    t0 = time.time()
    real_tb, dom, _ = C.load_real(C.CAL_REAL_TRAIN)
    test_tb, _, _ = C.load_real(C.CAL_REAL_TEST)
    hh_cols = [c for c in real_tb["household"].columns if c != "household_id"]
    res = {"n_train": N_TRAIN, "xgb_seed": SEED, "xgb_params": E.XGB_PARAMS, "results": {}}
    rng = np.random.default_rng(SEED)
    for task, (target, kind, metric, n_cls) in TASKS.items():
        Xte, yte, cat = matrices(test_tb, dom, target)
        ind_cols = [c for c in real_tb["individual"].columns
                    if c not in ("individual_id", "household_id", target)]
        sets = {"own": ind_cols, "own+hh": ind_cols + hh_cols, "full": list(Xte.columns)}
        res["results"][task] = {}
        for src, path in SOURCES.items():
            tb = real_tb if path is None else C.load_syn(path, C.CAL_REAL_TRAIN)[0]
            Xtr, ytr, _ = matrices(tb, dom, target)
            idx = rng.choice(len(Xtr), size=min(N_TRAIN, len(Xtr)), replace=False)
            Xtr, ytr = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
            res["results"][task][src] = {}
            for fs, cols in sets.items():
                spec = {c: v for c, v in cat.items() if c in cols}
                (a, b), _ = D.align_feature_dtypes([Xtr[cols], Xte[cols]], spec)
                pred, score = E.fit_predict(kind, a, ytr, b, SEED, n_cls)
                m = E.compute_metrics(kind, yte, pred, score, n_cls)[metric]
                res["results"][task][src][fs] = float(m)
                print(f"{task} {src:15s} {fs:7s} {metric}={m:.4f}  ({time.time() - t0:.0f}s)",
                      flush=True)
    C.dump(res, "california_relationship_importance.json")


if __name__ == "__main__":
    main()
