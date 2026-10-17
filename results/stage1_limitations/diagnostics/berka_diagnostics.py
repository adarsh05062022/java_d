"""Stage 1 Berka diagnostics on existing synthetic databases (no training, no sampling).

Four measurements, each computed identically on the real full database and on every
config x seed in common.BERKA_CONFIGS:

1. structure    -- cross-relation degree correlations of the sampled structure. The structure
                   sampler draws each relation's in-degrees independently, so these test whether
                   that independence is visible in the generated graph.
2. coherence    -- sibling coherence: the between-account share of variance (eta^2) of
                   transaction / order values, and the part of it NOT explained by the account's
                   own attributes and district (residual eta^2). Siblings that are conditionally
                   independent given the parent's columns would show residual eta^2 near 0.
3. label_deps   -- univariate AUROC of every downstream feature against the task label, inside
                   each database (real-in-real vs synthetic-in-synthetic), grouped by schema
                   family, plus P(card | trans_balance_mean octile) on real octile edges.
4. saturation   -- share of rows sitting exactly at a continuous column's real min or max
                   (where the inverse quantile transform parks out-of-range model outputs),
                   in excess of the real share.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 conda run -n tabsyn python berka_diagnostics.py
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
from downstream import tasks as T

KEY_FEATURES = {
    "loan": ["trans_n", "trans_balance_mean", "trans_balance_last", "trans_amount_mean",
             "trans_bank_0_frac", "trans_operation_3_frac", "order_n", "order_k_symbol_3_frac",
             "order_k_symbol_4_frac", "amount_per_month", "payments", "amount"],
    "card": ["trans_balance_mean", "trans_balance_last", "trans_amount_mean", "trans_n",
             "order_amount_sum", "order_n", "disp_type", "year", "frequency"],
}
OWN = {
    "loan": {"loan_date", "amount", "duration", "payments", "account_date", "frequency",
             "amount_per_payment", "amount_per_month", "account_age_at_loan"},
    "card": {"disp_type", "year", "month", "client_date", "gender", "account_date",
             "frequency", "same_district"},
}


def family(task, col):
    if col in OWN[task]:
        return "own_or_1hop_parent"
    if col.startswith("dist_"):
        return "district_fixed"
    if col.startswith("trans_") or col.startswith("order_"):
        return "child_aggregate"
    return "other_rollup"  # n_disps, client_*, disp_* : 2-hop rollups over disp/client


# ------------------------------------------------------------------------------ 1 structure
def structure(tb):
    acc = tb["account"][["account_id"]].set_index("account_id")
    for child in ("trans", "order", "loan", "disp"):
        acc[f"n_{child}"] = tb[child].groupby("account_id").size()
    card_per_disp = tb["card"].groupby("disp_id").size()
    disp = tb["disp"][["disp_id", "account_id"]].copy()
    disp["n_card"] = disp["disp_id"].map(card_per_disp).fillna(0)
    acc["n_card"] = disp.groupby("account_id")["n_card"].sum()
    acc = acc.fillna(0)
    out = {"digest": C.digest(acc.sort_index().to_numpy(dtype=np.int64))}
    pairs = [("n_trans", "n_order"), ("n_trans", "n_disp"), ("n_trans", "n_loan"),
             ("n_order", "n_loan"), ("n_disp", "n_card"), ("n_trans", "n_card"),
             ("n_order", "n_card")]
    for a, b in pairs:
        out[f"spearman_{a}_{b}"] = float(acc[a].corr(acc[b], method="spearman"))
    loan_acc = acc[acc["n_loan"] > 0]
    out["p_loan_account_has_order"] = float((loan_acc["n_order"] > 0).mean())
    # disposition-level: does the parent account's transaction count predict card holding?
    disp = disp.merge(acc[["n_trans", "n_order"]], left_on="account_id", right_index=True)
    y = (disp["n_card"] > 0).astype(int)
    out["auroc_ntrans_to_card"] = float(roc_auc_score(y, disp["n_trans"]))
    out["auroc_norder_to_card"] = float(roc_auc_score(y, disp["n_order"]))
    return out


def role_constraint(tb):
    """Each real account has exactly one owner disposition. For each disp_type code k: share of
    accounts with exactly one code-k disposition, and the share predicted if the dispositions of
    an account drew their type independently given the account's disposition count."""
    d = tb["disp"][["account_id", "disp_type"]]
    n = d.groupby("account_id").size().rename("n")
    out = {}
    for k in sorted(d["disp_type"].unique()):
        cnt = d.assign(_k=d["disp_type"] == k).groupby("account_id")["_k"].sum()
        out[f"frac_accounts_exactly_one_type{k}"] = float((cnt == 1).mean())
        p_n = d.merge(n, left_on="account_id", right_index=True).assign(
            _k=lambda x: x["disp_type"] == k).groupby("n")["_k"].mean()
        nn = n.to_numpy()
        p = p_n.reindex(nn).to_numpy()
        out[f"indep_pred_exactly_one_type{k}"] = float(np.mean(nn * p * (1 - p) ** (nn - 1)))
    return out


# ------------------------------------------------------------------------------ 2 coherence
def _parent_design(tb):
    a = tb["account"].set_index("account_id")
    fr = pd.get_dummies(a["frequency"].astype(int), prefix="f", drop_first=True, dtype=float)
    d = (a["account_date"] - a["account_date"].mean()) / (a["account_date"].std() + 1e-9)
    di = pd.get_dummies(a["district_id"].astype(int), prefix="d", drop_first=True, dtype=float)
    return pd.concat([fr, d.rename("d1"), (d**2).rename("d2"), (d**3).rename("d3"), di], axis=1)


def coherence(tb, n_boot=0, rng=None):
    out = {}
    design = _parent_design(tb)
    for table, col in (("trans", "amount"), ("trans", "balance"), ("order", "amount")):
        df = tb[table][["account_id", col]].dropna()
        v = C.rank01(df[col].to_numpy(float))
        eta, gm = C.eta_squared(v, df["account_id"].to_numpy())
        X = design.loc[gm.index].to_numpy(float)
        r2 = C.weighted_r2(gm["mean"].to_numpy(float), X, gm["size"].to_numpy(float))
        key = f"{table}.{col}"
        out[f"eta2_{key}"] = eta
        # chance-corrected (epsilon^2): with ~1.4 orders per account, raw eta^2 of pure noise is
        # already ~(k-1)/(N-1) ~ 0.6, so only the corrected value is comparable across tables
        k, n_tot = len(gm), len(v)
        ss_t = float(((v - v.mean()) ** 2).sum())
        ms_w = (1.0 - eta) * ss_t / max(n_tot - k, 1)
        out[f"eps2_{key}"] = (eta * ss_t - (k - 1) * ms_w) / ss_t
        out[f"chance_eta2_{key}"] = (k - 1) / (n_tot - 1)
        out[f"parent_r2_{key}"] = r2
        out[f"resid_eta2_{key}"] = eta * (1.0 - r2)
        if n_boot:
            g = pd.DataFrame({"g": df["account_id"].to_numpy(), "v": v, "v2": v * v})
            s = g.groupby("g").agg(n=("v", "size"), s1=("v", "sum"), s2=("v2", "sum"))
            n, s1, s2 = (s[c].to_numpy(float) for c in ("n", "s1", "s2"))
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(n), len(n))
                N, S1, S2 = n[i].sum(), s1[i].sum(), s2[i].sum()
                gr = S1 / N
                boots.append(((s1[i] ** 2 / n[i]).sum() - N * gr * gr) / (S2 - N * gr * gr))
            out[f"eta2_{key}_ci95"] = [float(np.percentile(boots, 2.5)),
                                       float(np.percentile(boots, 97.5))]
    return out


# ------------------------------------------------------------------------------ 3 label deps
def _aucs(X, y, cat):
    res = {}
    for c in X.columns:
        if c in cat:
            continue
        x = pd.to_numeric(X[c], errors="coerce")
        m = x.notna().to_numpy()
        if m.sum() < 20 or len(np.unique(y[m])) < 2 or x[m].nunique() < 2:
            continue
        res[c] = float(roc_auc_score(y[m], x[m]))
    # ordinal categoricals used as controls are scored on their codes
    for c in ("disp_type", "year", "frequency"):
        if c in X.columns and c not in res:
            x = pd.to_numeric(X[c], errors="coerce")
            m = x.notna().to_numpy()
            if len(np.unique(y[m])) == 2 and x[m].nunique() > 1:
                res[c] = float(roc_auc_score(y[m], x[m]))
    return res


def label_inputs(tb, dom):
    out = {}
    for task, build in (("loan", T.build_berka_loan), ("card", T.build_berka_card)):
        X, y, cat = build(tb, dom)
        out[task] = (X, np.asarray(y), cat)
    return out


def label_deps(inputs, real_aucs=None, real_edges=None):
    out = {}
    for task, (X, y, cat) in inputs.items():
        aucs = _aucs(X, y, cat)
        out[f"{task}_base_rate"] = float(np.mean(y))
        out[f"{task}_n"] = int(len(y))
        out[f"{task}_auc"] = {k: aucs.get(k) for k in KEY_FEATURES[task]}
        out[f"{task}_auc_all"] = aucs
        if real_aucs is not None:
            fam = {}
            for c, r in real_aucs[task].items():
                if abs(r - 0.5) <= 0.05 or c not in aucs:
                    continue
                fam.setdefault(family(task, c), []).append(
                    ((aucs[c] - 0.5) / (r - 0.5), np.sign(aucs[c] - 0.5) != np.sign(r - 0.5))
                )
            out[f"{task}_retained_by_family"] = {
                f: {"n_carriers": len(v), "mean_retained": float(np.mean([a for a, _ in v])),
                    "n_reversed": int(sum(b for _, b in v))}
                for f, v in fam.items()
            }
    # P(card | trans_balance_mean octile), real octile edges
    X, y, _ = inputs["card"]
    x = X["trans_balance_mean"].to_numpy(float)
    edges = real_edges if real_edges is not None else np.nanquantile(x, np.linspace(0, 1, 9))
    b = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, 7)
    curve = [float(y[b == k].mean()) if (b == k).any() else None for k in range(8)]
    out["p_card_by_balance_octile"] = curve
    return out, edges


# ------------------------------------------------------------------------------ 4 saturation
def saturation(tb, real_tb, dom):
    out = {}
    for t, d in dom.items():
        cols = [c for c, s in d.items() if s["type"] == "continuous" and c in tb[t].columns]
        if not cols or t == "district":
            continue
        per_col = {}
        for c in cols:
            r = real_tb[t][c].to_numpy(float)
            lo, hi = np.nanmin(r), np.nanmax(r)
            tol = 1e-9 * max(hi - lo, 1.0)
            s = tb[t][c].to_numpy(float)
            frac_s = float(np.mean((np.abs(s - lo) <= tol) | (np.abs(s - hi) <= tol)))
            frac_r = float(np.mean((np.abs(r - lo) <= tol) | (np.abs(r - hi) <= tol)))
            per_col[c] = frac_s - frac_r
        out[t] = {"max_excess_at_bound": max(per_col.values()),
                  "mean_excess_at_bound": float(np.mean(list(per_col.values()))),
                  "worst_column": max(per_col, key=per_col.get)}
    return out


def model_space_out_of_range(syn_dir, threshold=6.0):
    """Share of rows with any |x| > threshold in the stored model-space sample (pre-decoding)."""
    p = os.path.join(os.path.dirname(syn_dir), "node_type_to_X_syn.pkl")
    if not os.path.exists(p):
        return None
    X = pd.read_pickle(p)
    out = {t: float(np.mean(np.any(np.abs(np.asarray(x, float)) > threshold, axis=1)))
           for t, x in X.items() if t != "district" and np.asarray(x).size}
    # the quantile transform bounds real data at +-5.2; anything far beyond is an amplified error
    out["max_abs"] = {t: float(np.abs(np.asarray(x, float)).max())
                      for t, x in X.items() if t != "district" and np.asarray(x).size}
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    real, dom, _ = C.load_real(C.BERKA_REAL)
    rin = label_inputs(real, dom)
    r_lab, edges = label_deps(rin)
    real_aucs = {k: r_lab[f"{k}_auc_all"] for k in ("loan", "card")}
    # row-bootstrap CI on the real AUROC of the key features
    ci = {}
    for task, (X, y, cat) in rin.items():
        for c in KEY_FEATURES[task]:
            x = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
            m = ~np.isnan(x)
            xs, ys, bs = x[m], y[m], []
            for _ in range(400):
                i = rng.integers(0, len(xs), len(xs))
                if len(np.unique(ys[i])) == 2:
                    bs.append(roc_auc_score(ys[i], xs[i]))
            ci[f"{task}.{c}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    results = {
        "real": {
            "structure": structure(real),
            "roles": role_constraint(real),
            "coherence": coherence(real, n_boot=200, rng=rng),
            "label_deps": {k: v for k, v in r_lab.items() if not k.endswith("_auc_all")},
            "label_auc_ci95": ci,
            "saturation": saturation(real, real, dom),
        },
        "configs": {},
    }
    print(f"real done {time.time() - t0:.0f}s", flush=True)
    for name, pattern in C.BERKA_CONFIGS.items():
        results["configs"][name] = {}
        for s in C.SEEDS:
            d = pattern.format(s=s)
            if not os.path.isdir(d):
                results["configs"][name][s] = {"missing": d}
                continue
            tb, _, _ = C.load_syn(d, C.BERKA_REAL)
            lab, _ = label_deps(label_inputs(tb, dom), real_aucs, edges)
            results["configs"][name][s] = {
                "path": d,
                "structure": structure(tb),
                "roles": role_constraint(tb),
                "coherence": coherence(tb),
                "label_deps": {k: v for k, v in lab.items() if not k.endswith("_auc_all")},
                "saturation": saturation(tb, real, dom),
                "model_space_rows_abs_gt6": model_space_out_of_range(d),
            }
            print(f"{name} seed {s} done {time.time() - t0:.0f}s", flush=True)
    C.dump(results, "berka_diagnostics.json")


if __name__ == "__main__":
    main()
