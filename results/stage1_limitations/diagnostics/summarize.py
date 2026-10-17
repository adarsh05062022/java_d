"""Render the Stage 1 report tables from the diagnostic JSON outputs.

Conventions (kept per table, never mixed): Berka tables report mean +/- population SD
(ddof=0) over 3 generator seeds; California tables report mean +/- sample SD (ddof=1),
matching each dataset's established project tables.

Run:  conda run -n tabsyn python summarize.py   (writes ../outputs/summary_tables.md)
"""

import json
import os

import numpy as np

import common as C

B = json.load(open(os.path.join(C.OUT_DIR, "berka_diagnostics.json")))
CA = json.load(open(os.path.join(C.OUT_DIR, "california_diagnostics.json")))
RI = json.load(open(os.path.join(C.OUT_DIR, "california_relationship_importance.json")))
HG = json.load(open(os.path.join(C.OUT_DIR, "california_household_gap.json")))
SA = json.load(open(os.path.join(C.OUT_DIR, "schedule_amplification.json")))

BCFG = list(B["configs"])
CCFG = list(CA["configs"])
lines = []


def ms(vals, ddof, nd=3):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return "NA"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=ddof):.{nd}f}"


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    lines.append("")


def bget(path):
    def f(d):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                return None
        return d
    return f


def brow(label, path, nd=3, real=True):
    g = bget(path)
    r = g(B["real"]) if real else None
    return [label, "—" if r is None else f"{r:.{nd}f}"] + [
        ms([g(B["configs"][c][s]) for s in B["configs"][c]], 0, nd) for c in BCFG]


lines.append("## Berka (mean ± population SD, ddof=0, n=3 generator seeds)\n")
lines.append("### Sampled structure (identical across configs per seed; digests "
             + ", ".join(B["configs"]["baseline"][s]["structure"]["digest"] for s in B["configs"]["baseline"])
             + ")\n")
table(["metric", "real"] + BCFG, [brow(k, ["structure", k]) for k in B["real"]["structure"] if k != "digest"])
lines.append("### Sibling roles: accounts with exactly one disposition of each type "
             "(type1 = OWNER), observed and predicted under independence given the account's "
             "disposition count\n")
table(["metric", "real"] + BCFG, [brow(k, ["roles", k]) for k in B["real"]["roles"]])
lines.append("### Sibling coherence (rank space). eps2 = chance-corrected between-account share; "
             "resid_eta2 = eta2 x (1 - R2 of account means on account columns + district)\n")
keys = [k for k in B["real"]["coherence"] if k.startswith(("eps2", "resid", "eta2", "parent_r2"))
        and not k.endswith("ci95")]
table(["metric", "real"] + BCFG, [brow(k, ["coherence", k]) for k in keys])
lines.append("Real eta2 95% CIs (account bootstrap, 200 reps): "
             + "; ".join(f"{k}: [{v[0]:.3f}, {v[1]:.3f}]" for k, v in B["real"]["coherence"].items()
                         if k.endswith("ci95")) + "\n")
lines.append("### Label dependencies inside each database\n")
rows = [brow("loan base rate", ["label_deps", "loan_base_rate"]),
        brow("card base rate", ["label_deps", "card_base_rate"])]
for task in ("loan", "card"):
    for f in B["real"]["label_deps"][f"{task}_auc"]:
        ci = B["real"]["label_auc_ci95"].get(f"{task}.{f}")
        r = brow(f"{task}: AUROC {f}", ["label_deps", f"{task}_auc", f])
        if ci:
            r[1] += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
        rows.append(r)
    for fam in ("own_or_1hop_parent", "child_aggregate", "other_rollup"):
        rows.append(brow(f"{task}: retained, {fam}", ["label_deps", f"{task}_retained_by_family", fam, "mean_retained"], real=False))
        rows.append(brow(f"{task}: reversed carriers, {fam}", ["label_deps", f"{task}_retained_by_family", fam, "n_reversed"], 1, real=False))
table(["metric", "real"] + BCFG, rows)
nc = {t: {f: v["n_carriers"] for f, v in B["configs"]["baseline"]["0"]["label_deps"][f"{t}_retained_by_family"].items()}
      for t in ("loan", "card")}
lines.append(f"Carriers (real |AUROC-0.5| > 0.05): {nc}\n")
oct_rows = [["real"] + [f"{x:.3f}" for x in B["real"]["label_deps"]["p_card_by_balance_octile"]]]
for c in BCFG:
    arr = np.array([[np.nan if x is None else x for x in B["configs"][c][s]["label_deps"]["p_card_by_balance_octile"]]
                    for s in B["configs"][c]], dtype=float)
    oct_rows.append([c] + [f"{x:.3f}" for x in np.nanmean(arr, axis=0)])
lines.append("### P(card | trans_balance_mean octile), real octile edges, mean over seeds\n")
table(["source"] + [f"q{i + 1}" for i in range(8)], oct_rows)
lines.append("### Sampling health: rows with any |x| > 6 in model space; excess share of rows at a continuous column's real min/max\n")
rows = []
for t in ("account", "loan", "order", "trans", "disp", "card", "client"):
    rows.append(brow(f"|x|>6 rows: {t}", ["model_space_rows_abs_gt6", t], real=False))
for t in ("account", "loan", "order", "trans", "card"):
    rows.append(brow(f"at-bound excess (worst column): {t}", ["saturation", t, "max_excess_at_bound"], real=False))
table(["metric", "real"] + BCFG, rows)


def crow(label, sec, key, nd=3):
    r, f = CA["real_train"].get(sec, {}).get(key), CA["real_test_noise_floor"].get(sec, {}).get(key)
    return [label, "—" if r is None else f"{r:.{nd}f}", "—" if f is None else f"{f:.{nd}f}"] + [
        ms([CA["configs"][c][s][sec][key] for s in CA["configs"][c]], 1, nd) for c in CCFG]


lines.append("## California (mean ± sample SD, ddof=1, n=3 generator seeds)\n")
lines.append("### Sibling / group-level dependencies (households with >= 2 members)\n")
table(["metric", "real train", "real test (floor)"] + CCFG,
      [crow(k, "sibling", k, 0 if k.startswith("n_") else 4) for k in CA["real_train"]["sibling"]])
lines.append("### Conditional-distribution discrepancy to real train (lower is better; floor = real test vs real train)\n")
table(["dependency", "real train", "floor"] + CCFG,
      [crow(k, "discrepancy", k, 4) for k in CA["real_test_noise_floor"]["discrepancy"]])
lines.append(f"### Feature-set ablation, real test split (one GRDM seed, XGBoost seed {RI['xgb_seed']}, {RI['n_train']:,} training rows) — PRELIMINARY\n")
rows = []
for task, v in RI["results"].items():
    for src, r in v.items():
        rows.append([task, src] + [f"{r[k]:.4f}" for k in ("own", "own+hh", "full")])
table(["task", "trained on", "own", "own+hh", "full (+co-member rollups)"], rows)
rows = [[src] + [f"{r[k]:.4f}" for k in ("own", "own+PUMA", "own+hh_without_PUMA", "own+hh")]
        for src, r in HG["xgb_r2_inctot"].items()]
table(["INCTOT R², trained on", "own", "own+PUMA", "own+hh without PUMA", "own+hh"], rows)
rows = [[k] + [f"{v[x]:.4f}" for x in ("puma_tv", "inctot_given_puma_gap", "eta2_inctot_on_puma_ref", "eta2_inctot_on_puma_cand")]
        for k, v in HG["no_model"].items()]
table(["candidate", "P(PUMA) TV", "E[INCTOT|PUMA] gap", "eta2(INCTOT;PUMA) real", "eta2(INCTOT;PUMA) candidate"], rows)

lines.append("## Reverse-step amplification of eps error (cosine, T=2000)\n")
table(["t", "beta_t", "sqrt(abar_t)", "d x_{t-1} / d eps_hat"],
      [[t, f"{v['beta']:.5f}", f"{v['sqrt_abar']:.4f}", f"{v['eps_to_x_amplification']:.3f}"] for t, v in SA["per_t"].items()])
table(["Berka table", "seeds / batch", "P(batch has none)", "visits per row (200k steps)", "training samples at t=1999"],
      [[t, f"{v['seeds_per_batch']:.1f}", f"{v['p_batch_has_none']:.3f}", f"{v['visits_per_row_total']:.0f}",
        f"{v['samples_at_t1999_total']:.0f}"] for t, v in SA["berka_uniform_seeding"].items()])

out = os.path.join(C.OUT_DIR, "summary_tables.md")
open(out, "w").write("# Stage 1 diagnostic tables (generated by diagnostics/summarize.py)\n\n" + "\n".join(lines))
print(f"wrote {out}")
