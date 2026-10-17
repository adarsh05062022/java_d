"""Where does California's household -> member (1-hop) utility gap come from?

The feature-set ablation shows household columns add +0.051 R^2 to INCTOT when the model is
trained on real data but only +0.030 when trained on baseline synthetic data. One candidate is
PUMA, a 233-code nominal geography column that GRDM diffuses as a quantile-normalised ordinal
value and rounds back, so a small error lands on an unrelated area. This script separates PUMA
from the other household columns:

  1. P(PUMA) total-variation distance and E[INCTOT | PUMA] cell-mean gap (no model);
  2. XGBoost R^2 on the real test split for own, own+PUMA, own+hh_without_PUMA, own+hh.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 RDBDIFF_XGB_JOBS=8 \
      conda run -n tabsyn python california_household_gap.py
"""

import time

import numpy as np
import pandas as pd

import common as C
from california_relationship_importance import matrices
from downstream import data as D
from downstream import evaluate as E

N_TRAIN = 300_000
SEED = 0
MIN_CELL = 50


def puma_stats(ref_ind, ref_hh, ind, hh):
    a = ref_ind.merge(ref_hh[["household_id", "PUMA"]], on="household_id")
    b = ind.merge(hh[["household_id", "PUMA"]], on="household_id")
    pa, pb = a["PUMA"].value_counts(normalize=True), b["PUMA"].value_counts(normalize=True)
    tv = 0.5 * pa.subtract(pb, fill_value=0).abs().sum()
    ma, mb = a.groupby("PUMA")["INCTOT"].agg(["mean", "size"]), b.groupby("PUMA")["INCTOT"].agg(["mean", "size"])
    j = ma.join(mb, lsuffix="_r", rsuffix="_c", how="inner")
    j = j[(j["size_r"] >= MIN_CELL) & (j["size_c"] >= MIN_CELL)]
    gap = float(np.average((j["mean_c"] - j["mean_r"]).abs(), weights=j["size_r"]))
    # how much INCTOT variance PUMA explains inside each database (correlation ratio)
    eta_a, _ = C.eta_squared(a["INCTOT"].to_numpy(float), a["PUMA"].to_numpy())
    eta_b, _ = C.eta_squared(b["INCTOT"].to_numpy(float), b["PUMA"].to_numpy())
    return {"puma_tv": float(tv), "inctot_given_puma_gap": gap,
            "eta2_inctot_on_puma_ref": eta_a, "eta2_inctot_on_puma_cand": eta_b}


def main():
    t0 = time.time()
    real, dom, _ = C.load_real(C.CAL_REAL_TRAIN)
    test, _, _ = C.load_real(C.CAL_REAL_TEST)
    res = {"no_model": {"real_test_noise_floor": puma_stats(real["individual"], real["household"],
                                                            test["individual"], test["household"])},
           "xgb_r2_inctot": {}, "n_train": N_TRAIN, "xgb_seed": SEED}
    syn = {}
    for name in ("baseline", "agg"):
        for s in C.SEEDS:
            tb, _, _ = C.load_syn(C.CAL_CONFIGS[name].format(s=s), C.CAL_REAL_TRAIN)
            res["no_model"][f"{name}_seed{s}"] = puma_stats(real["individual"], real["household"],
                                                           tb["individual"], tb["household"])
            if s == 0:
                syn[f"{name}_seed0"] = tb
    print(f"no-model part done {time.time() - t0:.0f}s", flush=True)

    Xte, yte, cat = matrices(test, dom, "INCTOT")
    ind_cols = [c for c in real["individual"].columns if c not in ("individual_id", "household_id", "INCTOT")]
    hh_cols = [c for c in real["household"].columns if c != "household_id"]
    sets = {"own": ind_cols, "own+PUMA": ind_cols + ["PUMA"],
            "own+hh_without_PUMA": ind_cols + [c for c in hh_cols if c != "PUMA"],
            "own+hh": ind_cols + hh_cols}
    rng = np.random.default_rng(SEED)
    for src, tb in {"real_train": real, **syn}.items():
        Xtr, ytr, _ = matrices(tb, dom, "INCTOT")
        idx = rng.choice(len(Xtr), size=N_TRAIN, replace=False)
        Xtr, ytr = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
        res["xgb_r2_inctot"][src] = {}
        for fs, cols in sets.items():
            spec = {c: v for c, v in cat.items() if c in cols}
            (a, b), _ = D.align_feature_dtypes([Xtr[cols], Xte[cols]], spec)
            pred, _ = E.fit_predict("regression", a, ytr, b, SEED)
            r2 = E.compute_metrics("regression", yte, pred, None)["r2"]
            res["xgb_r2_inctot"][src][fs] = float(r2)
            print(f"{src:15s} {fs:20s} r2={r2:.4f} ({time.time() - t0:.0f}s)", flush=True)
    C.dump(res, "california_household_gap.json")


if __name__ == "__main__":
    main()
