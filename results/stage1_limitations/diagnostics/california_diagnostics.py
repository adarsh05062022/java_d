"""Stage 1 California conditional-information audit (no training, no sampling).

Every metric is a discrepancy between a candidate database and the real TRAIN split (the data
the generators were fitted to). The same discrepancy computed for the real TEST split is the
noise floor: two real samples of the same population.

Dependency classes, from closest to farthest from the generated row:
  within_row     P(INCTOT | AGE band x EDUC), P(EMPSTAT | AGE band x SEX)
  parent_child   P(INCTOT | household VALUEH decile), P(INCTOT | OWNERSHP x ROOMS),
                 P(EMPSTAT | OWNERSHP)                          (1 hop)
  sibling        between-household share of variance (eps^2) and leave-one-out
                 member-vs-co-member rank correlation for AGE, INCTOT, EDUC  (2 hops via household)
  group_size     P(INCTOT | AGE band x household size)          (structure x attribute)

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 conda run -n tabsyn python california_diagnostics.py
"""

import os
import time

import numpy as np
import pandas as pd

import common as C

AGE_BINS = [-1, 9, 17, 24, 34, 44, 54, 64, 200]
MIN_CELL = 50


def prep(tb):
    ind = tb["individual"].copy()
    hh = tb["household"].copy()
    ind["age_band"] = pd.cut(ind["AGE"], AGE_BINS, labels=False)
    size = ind.groupby("household_id").size().rename("hh_size")
    ind = ind.merge(hh, on="household_id", how="left").merge(
        size, left_on="household_id", right_index=True
    )
    ind["hh_size_c"] = ind["hh_size"].clip(upper=6)
    ind["valueh_dec"] = pd.qcut(ind["VALUEH"].rank(method="first"), 10, labels=False)
    ind["rooms_c"] = ind["ROOMS"].clip(upper=8)
    return ind


def cell_mean_gap(ref, cand, keys, target):
    """Real-mass-weighted mean |E_cand[target|cell] - E_ref[target|cell]| over cells with
    >= MIN_CELL rows in both."""
    a = ref.groupby(keys)[target].agg(["mean", "size"])
    b = cand.groupby(keys)[target].agg(["mean", "size"])
    j = a.join(b, lsuffix="_r", rsuffix="_c", how="inner")
    j = j[(j["size_r"] >= MIN_CELL) & (j["size_c"] >= MIN_CELL)]
    cover = float(j["size_r"].sum() / len(ref))
    return float(np.average((j["mean_c"] - j["mean_r"]).abs(), weights=j["size_r"])), cover


def cell_tv_gap(ref, cand, keys, target):
    """Real-mass-weighted total-variation distance between P(target | cell)."""
    pa = ref.groupby(keys)[target].value_counts(normalize=True).unstack(fill_value=0)
    pb = cand.groupby(keys)[target].value_counts(normalize=True).unstack(fill_value=0)
    na, nb = ref.groupby(keys).size(), cand.groupby(keys).size()
    cells = [c for c in pa.index if c in pb.index and na[c] >= MIN_CELL and nb[c] >= MIN_CELL]
    cols = sorted(set(pa.columns) | set(pb.columns))
    pa, pb = pa.reindex(columns=cols, fill_value=0), pb.reindex(columns=cols, fill_value=0)
    tv = 0.5 * (pa.loc[cells] - pb.loc[cells]).abs().sum(axis=1)
    return float(np.average(tv, weights=na[cells])), float(na[cells].sum() / len(ref))


def eta_stats(ind):
    """Sibling coherence on households with >= 2 members."""
    multi = ind[ind["hh_size"] >= 2]
    out = {}
    for col in ("AGE", "INCTOT", "EDUC"):
        v = C.rank01(multi[col].to_numpy(float))
        g = multi["household_id"].to_numpy()
        eta, gm = C.eta_squared(v, g)
        k, n = len(gm), len(v)
        ss_t = float(((v - v.mean()) ** 2).sum())
        ms_w = (1 - eta) * ss_t / (n - k)
        out[f"eps2_{col}"] = (eta * ss_t - (k - 1) * ms_w) / ss_t
        # leave-one-out co-member mean
        s = pd.Series(v).groupby(g).transform("sum").to_numpy()
        c = pd.Series(v).groupby(g).transform("size").to_numpy()
        loo = (s - v) / (c - 1)
        out[f"loo_corr_{col}"] = float(np.corrcoef(v, loo)[0, 1])
    # role-specific pairwise dependency: head (RELATE 0) vs spouse (RELATE 1), households with
    # exactly one of each. A permutation-invariant summary of the household cannot single out
    # this pair, so it separates "group-level" from "pairwise" coordination.
    heads = ind.assign(_h=ind["RELATE"] == 0).groupby("household_id")["_h"].sum()
    out["frac_households_exactly_one_head"] = float((heads == 1).mean())
    out["frac_households_no_head"] = float((heads == 0).mean())
    # Prediction if members' head status were independent given household size: with
    # p_n = P(RELATE == 0 | size n) measured in this same database,
    # P(exactly one head | n) = n p_n (1 - p_n)^(n - 1).
    hs = ind.groupby("household_id")["hh_size"].first()
    p_n = ind.assign(_h=ind["RELATE"] == 0).groupby("hh_size")["_h"].mean()
    n = hs.to_numpy()
    p = p_n.reindex(hs.to_numpy()).to_numpy()
    out["indep_pred_exactly_one_head"] = float(np.mean(n * p * (1 - p) ** (n - 1)))
    out["indep_pred_no_head"] = float(np.mean((1 - p) ** n))
    h = ind[ind["RELATE"] == 0].drop_duplicates("household_id", keep=False)
    sp = ind[ind["RELATE"] == 1].drop_duplicates("household_id", keep=False)
    pair = h.merge(sp, on="household_id", suffixes=("_h", "_s"))
    out["n_head_spouse_pairs"] = int(len(pair))
    for col in ("AGE", "EDUC", "INCTOT"):
        out[f"head_spouse_corr_{col}"] = float(np.corrcoef(pair[f"{col}_h"], pair[f"{col}_s"])[0, 1])
    return out


def discrepancies(ref, cand):
    out = {}
    out["within_row.INCTOT|age_band,EDUC"], _ = cell_mean_gap(ref, cand, ["age_band", "EDUC"], "INCTOT")
    out["within_row.EMPSTAT|age_band,SEX"], _ = cell_tv_gap(ref, cand, ["age_band", "SEX"], "EMPSTAT")
    out["parent_child.INCTOT|VALUEH_decile"], _ = cell_mean_gap(ref, cand, ["valueh_dec"], "INCTOT")
    out["parent_child.INCTOT|OWNERSHP,ROOMS"], _ = cell_mean_gap(ref, cand, ["OWNERSHP", "rooms_c"], "INCTOT")
    out["parent_child.EMPSTAT|OWNERSHP"], _ = cell_tv_gap(ref, cand, ["OWNERSHP"], "EMPSTAT")
    out["group_size.INCTOT|age_band,hh_size"], _ = cell_mean_gap(ref, cand, ["age_band", "hh_size_c"], "INCTOT")
    out["marginal.INCTOT_mean_abs_diff"] = float(abs(cand["INCTOT"].mean() - ref["INCTOT"].mean()))
    return out


def main():
    t0 = time.time()
    real_tb, _, _ = C.load_real(C.CAL_REAL_TRAIN)
    real = prep(real_tb)
    test_tb, _, _ = C.load_real(C.CAL_REAL_TEST)
    test = prep(test_tb)
    res = {
        "real_train": {"sibling": eta_stats(real), "n_individuals": len(real)},
        "real_test_noise_floor": {"sibling": eta_stats(test),
                                  "discrepancy": discrepancies(real, test),
                                  "n_individuals": len(test)},
        "configs": {},
    }
    print(f"real done {time.time() - t0:.0f}s", flush=True)
    for name, pattern in C.CAL_CONFIGS.items():
        res["configs"][name] = {}
        for s in C.SEEDS:
            d = pattern.format(s=s)
            if not os.path.isdir(d):
                res["configs"][name][s] = {"missing": d}
                continue
            tb, _, _ = C.load_syn(d, C.CAL_REAL_TRAIN)
            cand = prep(tb)
            res["configs"][name][s] = {"path": d, "sibling": eta_stats(cand),
                                       "discrepancy": discrepancies(real, cand)}
            print(f"{name} seed {s} done {time.time() - t0:.0f}s", flush=True)
    C.dump(res, "california_diagnostics.json")


if __name__ == "__main__":
    main()
