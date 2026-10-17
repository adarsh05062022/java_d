# Berka HRD with the entity-skip fix — 3-seed results (2026-10-17)

Leakage-free protocol (generators trained on `split_train`, scored on held-out accounts). Every arm:
3 seeds, 200k steps, same baseline normalizers + sampled structure per seed, x̂₀-clipped entity-first
sampling. Arms:

- **grdm** — leakage-free GRDM baseline (`berka_split_baseline/`).
- **h0h** — HRD ordering only (entity-first timesteps), GRDM denoiser.
- **attn** — h0h + child-count input + typed sibling attention + entity skip; **no entity state**.
- **d2skip** — attn + entity state (34-d mean/std of trans, order, loan w/o status, disp). The full method.
- d2 (broken, s1) — the earlier run whose state was never learned (`berka_hrd_state_failure.md`); for reference only.

Approval and the criteria below were recorded before results: `hrd_berka_runs/APPROVAL_skip.txt`.

## Pre-registered criteria (d2skip)

| # | criterion | result | verdict |
|---|---|---|---|
| 1 | state loss < 0.3 by 50k | 0.165 / 0.167 / 0.167 (200k: 0.107–0.110) | **met** |
| 2 | no dead account head | account loss 0.054–0.058 (h0h 0.074–0.076) | **met** |
| 3 | account column fidelity within 1 pt of GRDM | 97.31 ± 0.22 vs GRDM 91.28 ± 0.24 | **met** (+6.0) |
| 4 | order.amount ε² near real (≈ 0) | 0.386 ± 0.043 (real −0.001; broken d2 0.695) | **not met** |
| 5 | battery rel-view median retention ≥ 0.51 | 0.825 ± 0.015 | **met** |

## Task battery (median retention; mean ± SD over 3 seeds)

| view | grdm | h0h | attn | d2skip |
|---|---|---|---|---|
| own (same row) | 0.991 | 0.995 | 0.988 | 0.983 |
| up (parents) | −0.050 | −0.021 | 0.767 | 0.737 |
| sib (siblings) | −0.664 | −0.595 | 0.582 | 0.573 |
| down (children) | −0.056 | 0.191 | 0.947 | 0.953 |
| **rel (all cross-row)** | **−0.320 ± 0.109** | −0.243 ± 0.133 | 0.813 ± 0.019 | **0.825 ± 0.015** |
| all features | 0.467 ± 0.128 | 0.579 ± 0.045 | 0.959 ± 0.007 | 0.956 ± 0.009 |

rel view, share of tasks keeping ≥ 50% of the real signal: grdm 7%, h0h 16%, attn 76%, d2skip 77%;
keeping ≤ 10%: 74%, 70%, 13%, 10%.

## Fidelity (SDMetrics, ×100)

| metric | grdm | h0h | attn | d2skip |
|---|---|---|---|---|
| Column Shapes | 96.81 ± 0.08 | 97.09 ± 0.05 | 96.02 ± 0.11 | 96.18 ± 0.06 |
| Intra-Table Trends | 98.17 | 98.33 | 97.48 | 97.59 |
| 1-HOP | 90.33 ± 0.14 | 92.57 | 91.92 | 91.95 ± 0.15 |
| 2-HOP | 95.07 | 95.19 | 94.53 | 94.56 |
| 3-HOP | 91.96 | 91.48 | 91.45 | 91.45 |
| AVG 2-WAY | 94.68 ± 0.12 | 95.25 ± 0.10 | 94.57 ± 0.04 | 94.62 ± 0.17 |
| shape: account | 91.28 | 97.63 | 97.83 | 97.31 |
| shape: loan | 94.53 | 93.98 | 89.32 ± 1.38 | 91.98 ± 1.07 |
| shape: order | 97.90 | 97.45 | 94.98 ± 0.88 | 93.56 ± 0.36 |
| shape: trans | 97.47 | 97.72 | 96.21 | 96.09 |

## Downstream TSTR (XGB, 5 runs per generator seed; mean ± SD over generator seeds)

| task | real | grdm | h0h | attn | d2skip |
|---|---|---|---|---|---|
| loan default AUROC | 0.939 | 0.514 ± 0.032 | 0.579 ± 0.045 | 0.745 ± 0.047 | 0.755 ± 0.034 |
| loan default AP | 0.762 | 0.128 | 0.150 | 0.273 | 0.319 ± 0.086 |
| card ownership AUROC | 0.874 | 0.602 ± 0.009 | 0.589 ± 0.028 | 0.704 ± 0.081 | 0.654 ± 0.038 |

Loan test set has 19 positives: per-seed AUROC spread is large (d2skip 0.728–0.793).

## Mechanism metrics (vs training data)

| metric | real | grdm | attn | d2skip |
|---|---|---|---|---|
| exactly-one OWNER (indep. prediction ≈ 0.90) | 1.000 | 0.729 | 0.997 | 0.996 |
| ε² trans.amount | 0.095 | 0.009 | 0.093 | 0.100 |
| ε² trans.balance | 0.552 | 0.018 | 0.588 | 0.613 |
| ε² order.amount | −0.001 | −0.024 | 0.410 | 0.386 |
| loan child-aggregate retention | — | 0.065 | 0.534 ± 0.25 | 0.652 ± 0.29 |
| loan default rate (real 0.112) | 0.112 | 0.075 | 0.085 | 0.114 |
| loan values piled at a bound (max excess) | — | 0.008 | 0.011 | −0.004 |

## Paired d2skip − attn (same seed): does a WORKING entity state add anything?

| seed | battery rel mean-clip | battery all mean-clip | loan AUROC | card AUROC |
|---|---|---|---|---|
| 0 | +0.017 | +0.041 | +0.028 | −0.033 |
| 1 | +0.005 | +0.006 | −0.071 | −0.101 |
| 2 | +0.029 | +0.029 | +0.075 | −0.015 |
| mean | +0.017 | +0.025 | +0.011 | −0.050 |

## Findings (observed)

1. **The fix works.** State and account columns are learned; the account table goes from GRDM's worst
   table (91.3) to 97.3; no values pile at bounds.
2. **The sibling-coordination architecture is a large, general improvement over GRDM on Berka.** Cross-row
   battery retention −0.32 → +0.82 (3/3 seeds, non-overlapping); every dependency family flips from
   negative to positive; roles (exactly-one-OWNER 0.996 vs real 1.000; GRDM sits at independence 0.73);
   within-account coherence of transactions matches real (ε² 0.100 vs 0.095; 0.613 vs 0.552); loan TSTR
   0.51 → 0.75; card 0.60 → 0.65–0.70.
3. **The driver is attention (+ size input + skip), not the entity state.** attn and d2skip are tied on
   almost everything. The state gives a small battery gain whose sign is consistent over 3 seeds
   (+0.017 rel, +0.025 all; not significant with n = 3), a better loan default rate (0.114 vs 0.085, real
   0.112), better loan column fidelity (92.0 vs 89.3), and worse card TSTR (−0.05, 3/3 seeds) and order
   fidelity (93.6 vs 95.0).
4. **Remaining costs:** single-table fidelity −0.6 to −0.8 (Column Shapes) vs GRDM, concentrated in
   order (−3 to −4), loan (−2.5 to −5) and trans (−1.3); 2-HOP −0.5; AVG 2-WAY ≈ GRDM.
5. **Over-coupling of orders is NOT fixed** (ε² 0.39–0.41 vs real 0). It is present without the state
   (attn), so the earlier attribution to the random shared code was only partly right: attention itself
   makes an account's orders too similar. Trans coupling is right, so the issue is order-specific
   (few orders per account, 1–5).
6. The broken d2 s1's loan AUROC (0.861) was above every fixed seed (≤ 0.799); with 19 test positives and
   one seed this is not interpretable, and that model is invalid anyway.

## Caveats

- Berka only; this is the dataset the multi-table design was developed on. Generality needs a held-out
  database (ccs, instacart_05, f1, movie_lens available locally).
- Task battery not formally pre-registered (thresholds fixed before scoring synthetic data).
- n = 3 seeds per arm: paired differences between attn and d2skip are within noise except their sign.
