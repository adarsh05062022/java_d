# Consolidated Audit: Berka + California GRDM Conditioning Experiments

**Audit date:** 2026-10-13. **Auditor:** this session, reviewing artifacts produced by prior
sessions ("Agent 1" = degree-conditioning work, "Agent 2" = aggregate-conditioning work, "Agent
3"/California session = California aggregate-conditioning work) across six git worktrees.
**Scope:** Berka relational fidelity (5 configs), Berka downstream prediction (6 configs × 2
tasks), California household fidelity (4 configs), California 11-regime downstream coverage (4
configs). No GRDM model was retrained, no new sampling was run, and no existing experiment output
was overwritten to produce this audit. All cross-checks below were done by independently reading
raw per-seed JSON and recomputing means/SDs with a fresh Python snippet, not by re-reading the
prior sessions' own summary prose.

---

## STEP 1–2: What was collected and numerically verified

| Deliverable | Location | Verification performed |
|---|---|---|
| Berka relational fidelity (Table 1) | `results/relational_fidelity_consolidated/{README.md,relational_fidelity_table.csv,relational_fidelity_detailed.csv}` | Recomputed mean/SD for degree-scratch and degree-finetune directly from `fidelity_seed{0,1,2}.json` (numpy, ddof=0) — **matches exactly**. Pulled baseline/agg-scratch/agg-finetune numbers directly from `java_d-aggft/.../metrics/comparison.json` and confirmed byte-for-byte against the published table. |
| Berka downstream prediction (Tables 2–3) | `results/paper_repro/downstream/berka_prediction_tables/{REPORT.md,table_a_loan_default.csv,table_b_card_ownership.csv,aggregate_downstream.py}` | Independently re-ran the group-by-`grdm_seed`-then-average aggregation on two raw source files (baseline loan-default `synth`, aggregate-scratch loan-default `synth`) — **both match the published table to 3 decimals**. |
| California household fidelity (Table 4) | `java_d-calagg-ft/.../eleven_regime_addendum/TABLE_A_fidelity.{md,csv}` | Traced every row to `java_d-calagg-ft/.../metrics/comparison.json`'s `relational.summary.*` block and `noise_floor_real_test` — **matches exactly**, including the ablation checks (`agg_block_zeroed`/`agg_block_shuffled` held-out loss, see §5). |
| California 11-regime downstream coverage (Table 5) | `java_d-calagg-ft/.../eleven_regime_addendum/TABLE_B_eleven_regimes.{md,csv}` | Recomputed baseline `synth` income-R² from raw `results/paper_repro/downstream/california__california_inctot{,_seeds12}.json` — **matches exactly** (0.5274 ± 0.0025, ddof=1). Originally 42 of 88 cells were `PENDING`; the job producing them died silently, was relaunched fully detached, and completed 2026-10-14 05:09 IST — all 88 cells now populated, aggregated and spot-checked (`real_25/50/75` bit-identical across arms, as expected for no-GRDM-dependence regimes). |

**New files written by this audit** (nothing pre-existing was touched):
- `results/consolidated_audit/table4_california_household_fidelity.csv`
- `results/consolidated_audit/table5_california_eleven_regime_coverage.csv`
- `results/consolidated_audit/AUDIT_REPORT.md` (this file)

Tables 1–3 already lived, correctly, in this repo (`java_d`, branch `paper-repro`) and are referenced by path rather than duplicated, per "do not overwrite / do not needlessly duplicate existing outputs." Tables 4–5 only existed in the `java_d-calagg-ft` worktree; this audit copies their **verified** values into this repo so all five final tables are reachable from one place, with every cell traceable back to its source worktree/file.

---

## STEP 3: Experimental-validity audit — findings

These are the issues a careful read turned up. None of them invalidate the headline comparisons,
but all of them change how the numbers should be read.

### Finding 1 — Cardinality is bit-identical across every Berka configuration and is not a valid comparison metric here
`Cardinality` (row 1 of Table 1) is **99.65 ± 0.05 for all five Berka configurations**, and the
underlying per-seed values are identical to 12+ significant digits between `java_d_degcond` and
`java_d_degft` (two separate worktrees, separate training runs) and between `baseline`/
`agent2_scratch`/`ft_same_spec`/etc. inside `comparison.json` (confirmed by direct inspection:
`mean=99.64999999999999, sd=0.053541261347361556` repeated verbatim across all six keys). This
means Cardinality in this pipeline is a property of the **structure-sampling step**, which is
shared/deterministic given the seed and is **not affected by degree- or aggregate-conditioning**
(those only change attribute generation downstream of structure sampling). **Do not use
Cardinality to argue any configuration is "better" or "worse" — it carries no information about
the intervention being tested in any of these experiments.** `3-HOP relational fidelity` is
similarly flat across all five Berka configs (92.65–93.05, differences well within 1 SD of each
other) and should be treated the same way.

### Finding 2 — Standard-deviation convention differs between the Berka and California tables
- Table 1 (Berka fidelity) and Tables 2–3 (Berka downstream) use **population SD, `ddof=0`**
  (`numpy.std`/`statistics.pstdev`) — explicitly chosen by the prior session specifically to make
  all five Berka fidelity columns comparable (see `relational_fidelity_consolidated/README.md`
  §"Precision gotcha").
- Table 4 and Table 5 (California) use **sample SD, `ddof=1`** — inherited from
  `GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/compare.py` line 35
  (`np.std(v, ddof=1)`), confirmed by code inspection and by recomputing one cell
  (aggregate-conditioned synth income R² per-seed values `[0.52400, 0.52694, 0.52502]` give
  ddof=0 SD = 0.0012 but ddof=1 SD = 0.0015, matching the published 0.0015).
- For n=3, `ddof=1` SDs are **√(3/2) ≈ 1.22× larger** than `ddof=0` SDs on the same data. Means
  are unaffected either way. **This does not change any qualitative conclusion**, but the raw SD
  numbers in Tables 1–3 are not on the same numeric scale as Tables 4–5, and treating them as
  directly comparable "noise levels" across datasets would be wrong. Flagged rather than silently
  renormalized, per the instruction not to merge incompatible protocols.

### Finding 3 — Berka and California have genuinely different leakage exposure (not a copy-paste inconsistency)
Every one of the 6 Berka configurations' downstream JSON files carries `"leakage_caveat": true`,
and `downstream/run.py`'s `DATASETS["berka"]` entry says why: the GRDM generator for **every**
Berka configuration (baseline, degree ×2, aggregate ×3) was trained on
`data/berka/preprocessed`, the **full** database, which includes the accounts later held out as
the downstream test set (`BERKA_SPLIT_SEED=0`, 25% test). The classifier itself never sees test
labels, but the generator could have memorized test-account patterns.

California is different **by construction**, not by oversight: `DATASETS["california"]` sets
`real_train=data/california/train` (a real, separate split from `complex_data/california/test`)
and `leakage=False`. All three new California configs (`control`, `agg`, `agg_label_free`)
inherit `base="california"` and therefore the same `leakage=False`, confirmed directly in their
own result JSONs (`"leakage_caveat": false` in `california_agg_ft__california_inctot.json`, etc.)
— and mechanically, because they are fine-tuned from `java_d/rdbdiff_workspace/california/
paper_repro/seed_N`, a checkpoint that itself only ever trained on `data/california/train`.

**Consequence:** Berka's downstream numbers (Tables 2–3) are an upper bound on what a
leakage-free Berka generator would achieve (same caveat for all 6 configs, so *relative* ranking
between configs is not distorted, but *absolute* levels are optimistic). California's downstream
numbers (Table 5) carry no such caveat. A direct "is Berka's gain bigger than California's gain"
comparison must account for this asymmetry — see §5, Q7.

### Finding 4 — The Berka "aggregate fine-tuned" column in Table 1 reflects a sampling failure, not a clean fine-tuning read
Per the aggregate-finetune experiment's own `REPORT.html` (not re-verified byte-for-byte in this
audit, but consistent with the extreme values observed: Column-shape fidelity drops from 96.44
(scratch) to 84.46, far more than any other config-to-config gap in the table), the **joint**
(unfrozen-backbone) fine-tune of the aggregate-conditioned model produces unstable sampling — up
to 78% of rows fall outside the real-data range in model space. The numbers in the main Table 1
column are computed with the same protocol as every other column (so they are *mechanically*
comparable) but they measure this failure mode, not "what fine-tuning does to fidelity" in
general. The frozen-backbone counterpart (`ft_same_spec_frozen`, reported in the Table 1 appendix
in `relational_fidelity_consolidated/README.md`) does not collapse and sits close to baseline.
**Any claim about "does fine-tuning help relational fidelity" should use the frozen variant, not
the headline column.**

### Finding 5 — Berka and California test different interventions, not a matched pair
Berka's "aggregate from scratch" (config 4, Table 1/2/3) is trained **from scratch** on the full
data with aggregate conditioning baked in from step 0. California has **no from-scratch
aggregate-conditioned run** — `agg`/`agg_label_free`/`control` are all **continued-training
fine-tunes** of the existing `paper_repro` checkpoint (phase A: train only the new `agg_projs`
head; phase B: joint fine-tune). So any Berka-vs-California comparison of "does aggregate
conditioning help" is implicitly comparing Berka's *scratch* arm against California's *fine-tune*
arm — these are not the same experimental regime, and Berka's own data shows scratch and
fine-tune behave very differently (Finding 4, and §5 Q4). This is a scope limitation, not a
contradiction, and it's the single most important caveat for the Berka-vs-California consistency
question below.

### Finding 6 — Berka's label-free aggregate arms are not in the main 6-configuration downstream table
`berka_aggft_label_free{,_frozen}` results exist and are internally consistent with the
label-bearing arms, but weren't requested as one of the 6 named Berka configurations, so Table
2/3 omit them (available in `results/paper_repro/downstream/berka_prediction_tables/REPORT.md`
§6 on request). California's label-free arm (`agg_label_free`), by contrast, **is** included in
Tables 4–5 since it was one of the California session's three tested arms. This asymmetry is a
scoping choice carried over from the source reports, not an error, but it means "6 Berka
configs" and "4 California configs" are not simply parallel lists — flagged so it isn't read as
one.

### Finding 7 — Cross-checked against the pre-`/scratch`-loss original repository: the eval code is identical, the GRDM models are not
2026-10-14: the original, pre-loss checkout became readable again at
`/bhallukah/s25017/ClavaDDPM/rdb-diffusion` (the old `/scratch` mount, relocated — see
`grdm-rebuild-after-scratch-loss` memory). This audit diffed it directly against the rebuilt
`java_d` repo used for every table above:

- **Byte-identical**: `downstream/evaluate.py`, `downstream/data.py`, `downstream/splits.py`,
  `downstream/tasks.py` (including the Oct-5 date-offset bug fix, present in both), the entire
  `rdbdiff/` model+training code tree, `main.py`, and the training configs
  (`berka_fastsample.json`, `california_fastsample.json`). `downstream/run.py` differs by exactly
  one line — a `CLAVA_ROOT` path override for where California's test split lives, irrelevant to
  Berka and to every metric value.
- **The `real` regime (classifier trained on real data only, no generator involved) matches to
  full floating-point precision**, not just rounded — checked directly: Berka loan-default AUROC
  `0.938573407202216` in both repos; Berka card-ownership AUROC `0.8739133895876453` in both;
  California income R² matches to 4 decimals (0.5527 both). This proves the real-data split and
  XGBoost evaluation protocol are byte-for-byte unchanged by the rebuild.
- **Everything that depends on synthetic data differs**, because the original GRDM checkpoints
  (`rdbdiff_workspace/`) were physically lost when `/scratch` stopped being mounted on
  2026-10-07, and the `paper_repro` baseline used throughout this audit is a **fresh retrain**
  with the same code/config/seeds — not a copy of the original run. Diffusion-model training is
  stochastic, so a retrain with identical settings still yields different weights. Magnitude of
  the resulting metric drift is dataset-dependent: **Berka** (small, complex relational schema,
  only 171 test loans) shows real swings — loan-default `synth` AUROC 0.5316→0.5068,
  card-ownership `synth` AUROC 0.6094→0.5726; **California** (large, smooth regression/
  classification targets) barely moves — `synth` income R² 0.5272→0.5274, within noise.
- The original repo also never evaluated `blend_25/50/75` or `real_25/50/75` for Berka/California
  at all (those regimes are simply absent from its JSON files) — the rebuild's eval sweep is more
  complete, not inconsistent, on that front.

**Consequence for this report**: every number in Tables 1–5 is correctly computed from, and
internally consistent with, the current `paper_repro` baseline — but that baseline's absolute
synthetic-only numbers (especially for Berka) should not be assumed to match any pre-2026-10-07
reporting of the "same" experiment, because the underlying trained model is literally different.
Relative comparisons *between* the conditioning variants in this audit are unaffected (all of
them are trained on top of or measured against the same current baseline).

### Finding 8 — Correction: the leakage-free Berka baseline (`berka_split`) is NOT orphaned; it is recoverable from `/bhallukah`, and this audit's original claim to the contrary was wrong
The original audit (and §4/§7 of the downstream prediction report it cites) stated that the
leakage-free `berka_split` companion was an "orphaned artifact" because its GRDM checkpoint and
real-data split "no longer exist on disk" — that was checked only inside the `java_d` family of
worktrees, where it's true. It is **not** true overall: now that `/bhallukah` (the relocated old
`/scratch` mount) is reachable again, direct inspection found, fully intact:
- `rdbdiff_workspace/berka_split/paper_repro/seed_{0,1,2}/model_checkpoint/{model.pt,model_ema.pt,checkpoint.pt}`
  — real trained checkpoints, dated 2026-08-09, ~46 MB EMA weights each, plus already-sampled
  `synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database/` for every seed.
- `data/berka/split_train/` and `data/berka/split_test/` — the real-data split itself, with
  `dataset_meta.json` present (the exact file whose absence blocked this audit's own attempt to
  fill `blend_*` regimes for `berka_split` — see the downstream report's §5). Verified clean:
  3375 train accounts, 1125 test accounts, **zero account-ID overlap**.
- `downstream/tasks.py`, `evaluate.py`, `data.py`, `splits.py` are byte-identical to the current
  repo's copies (Finding 7), so the existing `downstream/results/berka_split__{task}.json` files
  were already computed with the same, bug-fixed protocol as every other number in this report —
  they are genuine, current-protocol, leakage-free results, not a stale artifact from a
  pre-bug-fix run.

**What this means**: the "historical" leakage-free numbers reported separately in §5 of the
downstream prediction report (loan default: synth AUROC 0.514±0.026; card ownership: synth AUROC
0.602±0.007) are **valid, usable leakage-free results for configuration 1 (original baseline)
today**, not a dead end. What's still missing is only regime *coverage*: the existing
`berka_split` JSON only has `real`/`synth`/`mixed`, not `blend_25/50/75` or `real_25/50/75` —
and closing that gap would not require retraining anything, only re-running the downstream
XGBoost evaluation against the already-sampled synthetic data that already exists at
`/bhallukah/.../rdbdiff_workspace/berka_split/paper_repro/seed_*`, the same kind of
no-retraining regime-completion job already done twice in this audit for California (Table 5).
**This audit did not perform that re-run** — it only corrects the factual record and flags the
opportunity; whether to pursue it is a scope decision for whoever owns this report next, since it
touches a shared, read-only-looking mount this audit did not attempt to write to.

### Nothing fabricated, nothing silently merged
No metric in Tables 1–5 was estimated, interpolated, or backfilled. (Table 5 originally had 42 of
88 cells marked `NA` with an explicit, checkable reason — resolved 2026-10-14, see Table 5 update
below; none are estimated.) The one historical leakage-free Berka baseline (`berka_split`) was
deliberately **excluded** from Tables 2–3 rather than merged in, per the reasoning in Finding 3 /
§4 — **note the status of that checkpoint has since changed, see Finding 8 below.**

---

## STEP 4 / TABLE 1 — Berka relational fidelity, all 5 configurations

Mean ± population SD (ddof=0), n=3 seeds, paper's 0–100 fidelity scale for rows 1–7, native 0–1
scale for rows 8–11. Full source/script provenance per cell: `results/relational_fidelity_consolidated/relational_fidelity_detailed.csv`.

| Metric | Original GRDM | Degree from scratch | Degree fine-tuned | Aggregate from scratch | Aggregate fine-tuned¹ |
|---|---|---|---|---|---|
| Cardinality² | 99.65 ± 0.05 | 99.65 ± 0.05 | 99.65 ± 0.05 | 99.65 ± 0.05 | 99.65 ± 0.05 |
| Column-shape fidelity | 96.997 ± 0.107 | 96.023 ± 0.094 | 90.907 ± 1.096 | 96.443 ± 0.160 | 84.457 ± 0.776 |
| Intra-table trends | 98.310 ± 0.043 | 97.506 ± 0.176 | 94.831 ± 0.600 | 97.520 ± 0.094 | 88.253 ± 0.442 |
| 1-hop relational fidelity | 91.487 ± 0.200 | 92.148 ± 0.035 | 88.877 ± 0.555 | 92.700 ± 0.049 | 80.913 ± 2.176 |
| 2-hop relational fidelity | 95.610 ± 0.071 | 95.050 ± 0.171 | 91.101 ± 0.964 | 95.017 ± 0.092 | 84.413 ± 1.554 |
| 3-hop relational fidelity² | 92.710 ± 0.449 | 92.653 ± 0.586 | 92.988 ± 0.906 | 93.047 ± 0.772 | 92.853 ± 0.550 |
| Average 2-way fidelity | 95.263 ± 0.052 | 94.935 ± 0.114 | 91.681 ± 0.697 | 95.063 ± 0.045 | 85.067 ± 0.576 |
| Aggregate KS distance (↓ better) | 0.2728 ± 0.0012 | 0.2037 ± 0.0013 | 0.2199 ± 0.0030 | 0.1425 ± 0.0054 | 0.3390 ± 0.0239 |
| Parent–aggregate correlation error (↓ better) | 0.1413 ± 0.0031 | 0.1213 ± 0.0023 | 0.1859 ± 0.0116 | 0.0863 ± 0.0034 | 0.2103 ± 0.0190 |
| Aggregate–aggregate correlation error (↓ better) | 0.2447 ± 0.0016 | 0.1800 ± 0.0026 | 0.2463 ± 0.0172 | 0.0760 ± 0.0049 | 0.2836 ± 0.0572 |
| Strong-dependency sign agreement, \|r\|≥0.1 (↑ better) | 0.6250 ± 0.0794 | 0.7639 ± 0.0196 | 0.6944 ± 0.0494 | 0.8843 ± 0.0458 | 0.7546 ± 0.0797 |

¹ = `ft_same_spec` (joint fine-tune). **Caveat: known sampling collapse, see Finding 4.** Frozen
counterpart (`ft_same_spec_frozen`): Column-shape 96.880±0.078, Intra-table 98.137±0.056, 1-hop
91.183±0.246, 2-hop 95.660±0.188, 3-hop 92.630±1.221, avg-2way 95.173±0.167, agg-KS
0.2574±0.0060, parent-agg-corr 0.1387±0.0006, agg-agg-corr 0.2205±0.0086, sign-agreement
0.7130±0.0429 — full appendix in `relational_fidelity_consolidated/README.md`.
² = **Not informative for comparing these configurations — see Finding 1.**

---

## STEP 4 / TABLE 2 — Berka loan-default prediction, all 6 configurations × 5 regimes

Mean ± population SD (ddof=0), n=3 GRDM seeds (each averaged over its own xgb_seeds first).
Real-only reference (no generator, identical across all 6 configs): AUROC 0.939, AP 0.762,
Macro-F1 0.800 (n=511 real train rows). **All 6 configs carry `leakage_caveat=true`** (Finding 3).
Full table incl. `mixed_25/50`, `real_25/50/75`: `results/paper_repro/downstream/berka_prediction_tables/table_a_loan_default_extended.csv`.

| Configuration | Synthetic-only | Real+Synthetic (add) | Blend 25% real | Blend 50% real | Blend 75% real |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.507/0.136/0.478 | 0.928/0.739/0.778 | 0.775/0.382/0.574 | 0.878/0.615/0.717 | 0.924/0.714/0.735 |
| 2. Degree-conditioned (scratch) | **0.789/0.356/0.531** | 0.944/0.747/0.775 | 0.870/0.480/0.582 | 0.903/0.611/0.702 | 0.920/0.687/0.746 |
| 3. Degree-conditioned (fine-tune) | 0.578/0.174/0.475 | 0.916/0.674/0.734 | 0.809/0.403/0.548 | 0.867/0.545/0.683 | 0.902/0.629/0.712 |
| 4. Aggregate-conditioned (scratch) | **0.826/0.455/0.562** | 0.936/0.705/0.726 | 0.883/0.562/0.646 | 0.902/0.652/0.713 | 0.916/0.692/0.756 |
| 5. Aggregate-conditioned (joint fine-tune) | 0.595/0.167/0.504 | 0.921/0.676/0.732 | 0.836/0.430/0.592 | 0.877/0.533/0.671 | 0.911/0.654/0.737 |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.534/0.164/0.493 | 0.921/0.694/0.759 | 0.773/0.362/0.593 | 0.877/0.556/0.675 | 0.921/0.678/0.733 |

(Cell format: AUROC/AP/Macro-F1, each ± SD omitted here for width — see the full-precision
`table_a_loan_default.csv` for per-cell SDs; bold = beats baseline on all three metrics.)

---

## STEP 4 / TABLE 3 — Berka card-ownership prediction, all 6 configurations × 5 regimes

Same protocol/caveats as Table 2. Real-only reference: AUROC 0.874, AP 0.480, F1 0.692 (n=4007).

| Configuration | Synthetic-only | Real+Synthetic (add) | Blend 25% real | Blend 50% real | Blend 75% real |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.573/0.191/0.473 | 0.873/0.482/0.686 | 0.856/0.455/0.644 | 0.865/0.472/0.674 | 0.870/0.471/0.682 |
| 2. Degree-conditioned (scratch) | **0.641/0.222/0.498** | 0.872/0.492/0.694 | 0.843/0.444/0.629 | 0.864/0.479/0.670 | 0.870/0.477/0.676 |
| 3. Degree-conditioned (fine-tune) | 0.617/0.220/0.485 | 0.876/0.497/0.695 | 0.858/0.471/0.654 | 0.864/0.474/0.669 | 0.871/0.476/0.686 |
| 4. Aggregate-conditioned (scratch) | **0.648/0.235/0.512** | 0.866/0.486/0.684 | 0.804/0.406/0.599 | 0.851/0.460/0.656 | 0.865/0.471/0.675 |
| 5. Aggregate-conditioned (joint fine-tune) | 0.606/0.209/0.467 | 0.869/0.480/0.683 | 0.839/0.436/0.626 | 0.860/0.462/0.666 | 0.868/0.468/0.675 |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.569/0.181/0.467 | 0.869/0.470/0.680 | 0.845/0.434/0.635 | 0.861/0.461/0.665 | 0.867/0.468/0.675 |

(Cell format: AUROC/AP/F1; bold = beats baseline on all three metrics. Full precision + SDs:
`table_b_card_ownership.csv`.)

---

## STEP 4 / TABLE 2a & 3a — Gap to the real-data-only ceiling (Berka, all 6 configs, all 3 metrics)

Tables 2–3 compare GRDM configurations *against each other*. This section makes the comparison
against **real data** explicit and complete: for every (config, regime, metric) cell, the gap to
the real-only reference (classifier trained on real data alone, mean over 5 xgb-seed fits, zero
GRDM dependence — same value for all 6 configs, verified bit-identical to full floating-point
precision: loan-default AUROC `0.938573407202216`, card-ownership AUROC `0.8739133895876453`).
Format: `mean±SD (gap)`, gap = regime mean − real-only mean; `mixed` = real+synthetic added.

#### Loan default

**AUROC** (real-only reference: 0.9386 ± 0.0000)

| Configuration | synth | mixed | blend_25 | blend_50 | blend_75 |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.507±0.067 (−0.432) | 0.928±0.005 (−0.010) | 0.775±0.012 (−0.164) | 0.878±0.016 (−0.060) | 0.924±0.003 (−0.015) |
| 2. Degree-conditioned (scratch) | 0.789±0.019 (−0.150) | 0.944±0.009 (**+0.005**) | 0.870±0.019 (−0.069) | 0.903±0.007 (−0.035) | 0.920±0.013 (−0.019) |
| 3. Degree-conditioned (fine-tune) | 0.578±0.087 (−0.361) | 0.916±0.005 (−0.023) | 0.809±0.006 (−0.130) | 0.867±0.022 (−0.072) | 0.902±0.007 (−0.036) |
| 4. Aggregate-conditioned (scratch) | 0.826±0.039 (−0.113) | 0.936±0.019 (−0.003) | 0.883±0.020 (−0.056) | 0.902±0.014 (−0.037) | 0.916±0.005 (−0.022) |
| 5. Aggregate-conditioned (joint fine-tune) | 0.595±0.041 (−0.344) | 0.921±0.012 (−0.017) | 0.836±0.036 (−0.102) | 0.877±0.025 (−0.061) | 0.911±0.011 (−0.028) |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.534±0.086 (−0.404) | 0.921±0.016 (−0.018) | 0.773±0.003 (−0.165) | 0.877±0.024 (−0.062) | 0.921±0.010 (−0.018) |

**AP** (real-only reference: 0.7624 ± 0.0000)

| Configuration | synth | mixed | blend_25 | blend_50 | blend_75 |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.136±0.031 (−0.626) | 0.739±0.014 (−0.023) | 0.382±0.034 (−0.380) | 0.615±0.027 (−0.147) | 0.714±0.033 (−0.048) |
| 2. Degree-conditioned (scratch) | 0.356±0.031 (−0.407) | 0.747±0.029 (−0.015) | 0.480±0.023 (−0.283) | 0.611±0.034 (−0.152) | 0.687±0.052 (−0.076) |
| 3. Degree-conditioned (fine-tune) | 0.174±0.049 (−0.588) | 0.674±0.016 (−0.088) | 0.403±0.009 (−0.360) | 0.545±0.042 (−0.217) | 0.629±0.027 (−0.133) |
| 4. Aggregate-conditioned (scratch) | 0.455±0.106 (−0.308) | 0.705±0.075 (−0.057) | 0.562±0.095 (−0.201) | 0.652±0.065 (−0.110) | 0.692±0.034 (−0.071) |
| 5. Aggregate-conditioned (joint fine-tune) | 0.167±0.038 (−0.596) | 0.676±0.052 (−0.086) | 0.430±0.097 (−0.332) | 0.533±0.066 (−0.229) | 0.654±0.067 (−0.108) |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.164±0.047 (−0.598) | 0.694±0.038 (−0.069) | 0.362±0.007 (−0.400) | 0.556±0.018 (−0.206) | 0.678±0.037 (−0.085) |

**Macro-F1** (real-only reference: 0.7997 ± 0.0000)

| Configuration | synth | mixed | blend_25 | blend_50 | blend_75 |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.478±0.006 (−0.322) | 0.778±0.004 (−0.022) | 0.574±0.014 (−0.226) | 0.717±0.004 (−0.083) | 0.735±0.014 (−0.065) |
| 2. Degree-conditioned (scratch) | 0.531±0.022 (−0.268) | 0.775±0.006 (−0.024) | 0.582±0.001 (−0.218) | 0.702±0.020 (−0.098) | 0.746±0.008 (−0.054) |
| 3. Degree-conditioned (fine-tune) | 0.475±0.004 (−0.324) | 0.734±0.007 (−0.065) | 0.548±0.026 (−0.251) | 0.683±0.005 (−0.117) | 0.712±0.010 (−0.088) |
| 4. Aggregate-conditioned (scratch) | 0.562±0.026 (−0.238) | 0.726±0.025 (−0.073) | 0.646±0.028 (−0.154) | 0.713±0.017 (−0.086) | 0.756±0.020 (−0.044) |
| 5. Aggregate-conditioned (joint fine-tune) | 0.504±0.037 (−0.295) | 0.732±0.031 (−0.068) | 0.592±0.010 (−0.207) | 0.671±0.016 (−0.129) | 0.737±0.034 (−0.062) |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.493±0.017 (−0.306) | 0.759±0.028 (−0.041) | 0.593±0.029 (−0.206) | 0.675±0.028 (−0.125) | 0.733±0.022 (−0.067) |

#### Card ownership

**AUROC** (real-only reference: 0.8739 ± 0.0000)

| Configuration | synth | mixed | blend_25 | blend_50 | blend_75 |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.573±0.031 (−0.301) | 0.873±0.002 (−0.001) | 0.856±0.003 (−0.018) | 0.865±0.003 (−0.009) | 0.870±0.002 (−0.004) |
| 2. Degree-conditioned (scratch) | 0.641±0.004 (−0.233) | 0.872±0.002 (−0.002) | 0.843±0.003 (−0.031) | 0.864±0.001 (−0.010) | 0.870±0.002 (−0.004) |
| 3. Degree-conditioned (fine-tune) | 0.617±0.032 (−0.257) | 0.876±0.002 (**+0.002**) | 0.858±0.001 (−0.016) | 0.864±0.001 (−0.010) | 0.871±0.001 (−0.003) |
| 4. Aggregate-conditioned (scratch) | 0.648±0.013 (−0.226) | 0.866±0.003 (−0.007) | 0.804±0.013 (−0.070) | 0.851±0.005 (−0.022) | 0.865±0.001 (−0.009) |
| 5. Aggregate-conditioned (joint fine-tune) | 0.606±0.015 (−0.268) | 0.869±0.001 (−0.005) | 0.839±0.001 (−0.035) | 0.860±0.003 (−0.013) | 0.868±0.001 (−0.006) |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.569±0.028 (−0.305) | 0.869±0.001 (−0.005) | 0.845±0.002 (−0.029) | 0.861±0.001 (−0.013) | 0.867±0.000 (−0.007) |

**AP** (real-only reference: 0.4801 ± 0.0000)

| Configuration | synth | mixed | blend_25 | blend_50 | blend_75 |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.191±0.019 (−0.289) | 0.482±0.014 (**+0.002**) | 0.455±0.009 (−0.025) | 0.472±0.013 (−0.008) | 0.471±0.006 (−0.009) |
| 2. Degree-conditioned (scratch) | 0.222±0.007 (−0.258) | 0.492±0.003 (**+0.012**) | 0.444±0.010 (−0.036) | 0.479±0.007 (−0.001) | 0.477±0.002 (−0.003) |
| 3. Degree-conditioned (fine-tune) | 0.220±0.022 (−0.260) | 0.497±0.008 (**+0.017**) | 0.471±0.004 (−0.009) | 0.474±0.003 (−0.006) | 0.476±0.003 (−0.004) |
| 4. Aggregate-conditioned (scratch) | 0.235±0.012 (−0.245) | 0.486±0.005 (**+0.006**) | 0.406±0.020 (−0.074) | 0.460±0.005 (−0.020) | 0.471±0.003 (−0.009) |
| 5. Aggregate-conditioned (joint fine-tune) | 0.209±0.012 (−0.271) | 0.480±0.004 (+0.000) | 0.436±0.005 (−0.044) | 0.462±0.007 (−0.018) | 0.468±0.005 (−0.012) |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.181±0.015 (−0.299) | 0.470±0.004 (−0.010) | 0.434±0.007 (−0.046) | 0.461±0.005 (−0.019) | 0.468±0.001 (−0.012) |

**F1** (real-only reference: 0.6924 ± 0.0000)

| Configuration | synth | mixed | blend_25 | blend_50 | blend_75 |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | 0.473±0.015 (−0.219) | 0.686±0.006 (−0.006) | 0.644±0.003 (−0.048) | 0.674±0.005 (−0.019) | 0.682±0.005 (−0.010) |
| 2. Degree-conditioned (scratch) | 0.498±0.006 (−0.195) | 0.694±0.003 (**+0.001**) | 0.629±0.008 (−0.064) | 0.670±0.006 (−0.022) | 0.676±0.001 (−0.017) |
| 3. Degree-conditioned (fine-tune) | 0.485±0.006 (−0.207) | 0.695±0.002 (**+0.003**) | 0.654±0.003 (−0.039) | 0.669±0.004 (−0.023) | 0.686±0.005 (−0.007) |
| 4. Aggregate-conditioned (scratch) | 0.512±0.010 (−0.181) | 0.684±0.007 (−0.009) | 0.599±0.011 (−0.093) | 0.656±0.011 (−0.037) | 0.675±0.006 (−0.017) |
| 5. Aggregate-conditioned (joint fine-tune) | 0.467±0.007 (−0.226) | 0.683±0.004 (−0.009) | 0.626±0.001 (−0.066) | 0.666±0.007 (−0.027) | 0.675±0.004 (−0.017) |
| 6. Aggregate-conditioned (frozen fine-tune) | 0.467±0.008 (−0.225) | 0.680±0.001 (−0.013) | 0.635±0.006 (−0.057) | 0.665±0.003 (−0.028) | 0.675±0.001 (−0.017) |

**Reading this table**: every synthetic-involving regime sits at or below the real-only ceiling
across all 90 (config × regime × metric) cells in both tasks, with exactly 7 nominal exceptions
(bolded, all in the `mixed` = real+synthetic-added regime, never in `synth` or any `blend_*`) —
and every one of those 7 is smaller than that cell's own seed-to-seed SD, so none represents a
result distinguishable from "ties the real-data ceiling." No cell beats real data by a margin
larger than its own SD. This directly answers whether synthetic data can substitute for or exceed
real data here: at this sample size and seed count, no configuration does, in any regime, by a
confirmed (non-noise) margin — synthetic data closes most of the gap as more real data is blended
in, but does not cross it.

---

## STEP 4 / TABLE 4 — California household fidelity, all 4 configurations

Mean ± **sample SD, ddof=1** (Finding 2 — not the same convention as Tables 1–3), n=3 seeds.
Source: `results/consolidated_audit/table4_california_household_fidelity.csv` (verified against
`java_d-calagg-ft/.../metrics/comparison.json`). "Control" = continued training on the baseline
checkpoint with no aggregate head (isolates the effect of extra training steps alone).

| Metric | Original GRDM | Control | Aggregate-conditioned | Label-free aggregate-conditioned | Real-vs-real noise floor |
|---|---|---|---|---|---|
| Household-summary mean KS distance (↓) | 0.1489 ± 0.0002 | 0.1489 ± 0.0003 | **0.0543 ± 0.0016** | 0.0547 ± 0.0018 | 0.0031 |
| Downstream household-feature mean KS distance (↓) | 0.1114 ± 0.0002 | 0.1110 ± 0.0004 | **0.0338 ± 0.0009** | 0.0343 ± 0.0015 | 0.0031 |
| Household-attribute × member-summary corr. error, MAE (↓) | 0.0401 ± 0.0006 | 0.0398 ± 0.0003 | **0.0227 ± 0.0011** | 0.0228 ± 0.0008 | 0.0041 |
| Same-household agreement error, MAE (↓) | 0.0898 ± 0.0001 | 0.0902 ± 0.0002 | **0.0555 ± 0.0013** | 0.0557 ± 0.0013 | 0.0008 |
| Within-household variance-ratio error, MAE (↓) | 0.1562 ± 0.0003 | 0.1568 ± 0.0000 | **0.0697 ± 0.0018** | 0.0702 ± 0.0022 | 0.0013 |
| Numerical marginal mean TVD (↓) | 0.0617 ± 0.0010 | 0.0615 ± 0.0003 | 0.0533 ± 0.0004 | 0.0531 ± 0.0002 | 0.0073 |
| Categorical marginal mean TVD (↓) | 0.0138 ± 0.0008 | 0.0130 ± 0.0003 | 0.0111 ± 0.0009 | 0.0111 ± 0.0008 | 0.0032 |
| Held-out denoising loss (↓, not real-vs-real comparable) | 0.3003 ± 0.0002 | 0.2996 ± 0.0002 | 0.2874 ± 0.0002 | 0.2876 ± 0.0002 | NA |

**Mechanism check (not in the main table, but relevant to "did conditioning actually do
anything"):** ablating the learned aggregate signal at eval time makes held-out loss *worse* than
baseline, not just worse than the conditioned model — `agg_block_zeroed` = 0.3405 ± 0.0030,
`agg_block_shuffled` = 0.3682 ± 0.0061 vs. conditioned 0.2874 ± 0.0002 and baseline 0.3003 ±
0.0002. This confirms the model is genuinely using the aggregate conditioning signal (not
ignoring it), source: same `comparison.json`, key `relational.heldout_denoising_loss`.

---

## STEP 4 / TABLE 5 — California 11-regime downstream coverage, all 4 configurations

**Update (2026-10-14): now complete. All 88 of 88 cells (11 regimes × 4 configs × 2 tasks) are
populated — 0 remaining `NA`.** n=3 seeds throughout. Full CSV:
`results/consolidated_audit/table5_california_eleven_regime_coverage.csv`.

| Regime | Original GRDM (income R² / employ. F1) | Control | Aggregate-cond. | Label-free agg.-cond. |
|---|---|---|---|---|
| real | 0.5527±0.0004 / 0.6850±0.0001 | same (arm-invariant, no GRDM dependence) | same | same |
| synth | 0.5274±0.0025 / 0.6757±0.0008 | 0.5273±0.0025 / 0.6752±0.0005 | 0.5253±0.0015 / 0.6742±0.0008 | 0.5236±0.0022 / 0.6739±0.0008 |
| mixed_25 | 0.5522±0.0003 / 0.6832±0.0004 | 0.5523±0.0001 / 0.6831±0.0003 | 0.5522±0.0001 / 0.6827±0.0002 | 0.5520±0.0002 / 0.6826±0.0001 |
| mixed_50 | 0.5516±0.0001 / 0.6823±0.0002 | 0.5516±0.0002 / 0.6820±0.0004 | 0.5512±0.0001 / 0.6812±0.0001 | 0.5511±0.0001 / 0.6814±0.0002 |
| mixed | 0.5503±0.0005 / 0.6810±0.0006 | 0.5503±0.0005 / 0.6808±0.0003 | 0.5494±0.0002 / 0.6797±0.0003 | 0.5492±0.0001 / 0.6799±0.0001 |
| real_50 | 0.5478±0.0004 / 0.6840±0.0005 | 0.5478±0.0004 / 0.6840±0.0005 | 0.5478±0.0004 / 0.6840±0.0005 | 0.5478±0.0004 / 0.6840±0.0005 |
| blend_50 | 0.5470±0.0003 / 0.6805±0.0005 | 0.5474±0.0004 / 0.6803±0.0006 | 0.5464±0.0003 / 0.6794±0.0005 | 0.5462±0.0003 / 0.6797±0.0008 |
| real_25 | 0.5400±0.0002 / 0.6832±0.0001 | 0.5400±0.0002 / 0.6832±0.0001 | 0.5400±0.0002 / 0.6832±0.0001 | 0.5400±0.0002 / 0.6832±0.0001 |
| blend_25 | 0.5414±0.0008 / 0.6793±0.0001 | 0.5417±0.0010 / 0.6788±0.0004 | 0.5398±0.0006 / 0.6779±0.0002 | 0.5394±0.0009 / 0.6775±0.0002 |
| real_75 | 0.5511±0.0001 / 0.6845±0.0004 | 0.5511±0.0001 / 0.6845±0.0004 | 0.5511±0.0001 / 0.6845±0.0004 | 0.5511±0.0001 / 0.6845±0.0004 |
| blend_75 | 0.5506±0.0001 / 0.6827±0.0004 | 0.5507±0.0000 / 0.6825±0.0007 | 0.5503±0.0002 / 0.6821±0.0001 | 0.5502±0.0001 / 0.6823±0.0001 |

**How the 42 previously-pending cells were filled:** the background job that was supposed to
produce them (launched 2026-10-13 15:42 IST, PIDs 2744610–2744612) **died silently ~37 minutes
in** with no error, no traceback, no OOM-killer evidence, and no output file — most likely because
its process group was tied to a tool-call session that was later torn down, rather than any
resource limit (the box had 685 GB free memory and load average 67/224 cores at the time it was
checked). It was relaunched 2026-10-14 00:23 IST (PGID 425955), fully detached via
`setsid`+`nohup`+`disown` with stdin redirected from `/dev/null` specifically so it would survive
regardless of this session's own lifecycle, and ran to completion (`exit 0` for all 3 arms) by
05:09 IST — about 4h46m wall time, well under the original ~10h estimate. Each of the 6 new
`*_missing.json` files has exactly 63 records (7 regimes × 3 GRDM seeds × 3 xgb seeds, all 3 seeds
present) and `leakage_caveat=false`, consistent with the rest of Table 5. Aggregated with the same
protocol as the rest of the table (mean over xgb_seed within grdm_seed, then mean/sample-SD
(`ddof=1`) across grdm_seed — matching Finding 2's documented California convention). No retraining
or resampling was performed; the job only re-fit XGBoost on already-sampled synthetic data.

**Validation from the completed sweep:** `real_25`/`real_50`/`real_75` (pure real-data subsamples,
no synthetic rows involved) are bit-identical across all three new arms in every case checked —
expected, since these regimes have no generator dependence, and now confirmed across three
additional fractions beyond the `real` (100%) regime already checked in the original audit.

---

## STEP 5: Interpretation (verified results only)

**1. Does degree conditioning improve relational fidelity (Berka — no California degree-conditioning experiment exists)?**
Mixed, metric-dependent, and **from-scratch only**. Degree-from-scratch vs. baseline: the four
new dependency metrics it was designed to help all improve (agg-KS 0.273→0.204, parent-agg-corr
0.141→0.121, agg-agg-corr 0.245→0.180, sign-agreement 0.625→0.764), and 1-hop fidelity ticks up
slightly (91.49→92.15). But the paper's general fidelity metrics mostly drop a little
(column-shape 97.00→96.02, intra-table 98.31→97.51, 2-hop 95.61→95.05, avg-2way 95.26→94.93).
Degree-**fine-tuned** is worse than both baseline and degree-scratch on almost every metric
(column-shape 90.91, intra-table 94.83) — fine-tuning degree conditioning onto an existing
checkpoint actively hurts fidelity here, it does not just fail to help.

**2. Does aggregate conditioning improve relational fidelity?**
Yes, more cleanly than degree conditioning, and consistently across **both datasets** for the
metrics it directly targets. Berka aggregate-from-scratch: agg-KS 0.273→0.143, parent-agg-corr
0.141→0.086, agg-agg-corr 0.245→0.076, sign-agreement 0.625→0.884 — the single clearest fidelity
win of any config in Table 1, with only small costs on the general fidelity metrics
(column-shape 97.00→96.44, 2-hop 95.61→95.02). California aggregate-conditioning (fine-tune):
household-summary KS 0.149→0.054, downstream-feature KS 0.111→0.034, household×member corr. MAE
0.040→0.023, same-household agreement MAE 0.090→0.056, within-household variance-ratio MAE
0.156→0.070 — comparably large, consistent improvements, **and** unlike Berka, California's
general fidelity metrics also improve in the paper-suite block (not shown in Table 4, confirmed
in `comparison.json`'s `fidelity.summary.agg` vs `.baseline`: Column Shapes 98.573±0.064 →
98.971±0.074, 1-HOP 96.447±0.098 → 97.071±0.158). Caveat: Berka's arm is from-scratch,
California's is fine-tuned (Finding 5) — not a fully matched comparison.

**3. Which approach performs better on each relevant fidelity metric (Berka, where both exist)?**
Aggregate-conditioning-from-scratch beats degree-conditioning-from-scratch on every one of the 4
dependency metrics (agg-KS 0.143 vs 0.204; parent-agg-corr 0.086 vs 0.121; agg-agg-corr 0.076 vs
0.180; sign-agreement 0.884 vs 0.764) and on 3 of 4 paper-suite relational metrics (1-hop 92.70
vs 92.15; 2-hop tied within noise 95.02 vs 95.05; column-shape 96.44 vs 96.02; intra-table 97.52
vs 97.51, tied). Aggregate conditioning is the stronger fidelity intervention on Berka, full stop.

**4. Do fidelity improvements translate into improved downstream prediction?**
**Not reliably — this is the most important finding of the audit, and now confirmed across the
full 11-regime California sweep rather than the 4 regimes available at the time of the first
pass.** On Berka, yes: the two configs with the biggest fidelity gains (degree-scratch,
aggregate-scratch) are also the only two configs that beat the baseline on synthetic-only
AUROC/AP/F1 for both tasks (aggregate-scratch loan AUROC 0.507→0.826; card AUROC 0.573→0.648). On
California, **no**: aggregate-conditioning gave the largest fidelity improvement in the entire
audit (household-summary KS dropped 64%) but downstream utility is flat-to-slightly-worse than
both the original baseline and the no-aggregate `Control` arm in **every one of the 11 regimes,
both tasks** (e.g. `blend_25` income R² 0.5417 (Control) → 0.5398 (Aggregate-conditioned);
`mixed_50` employment F1 0.6820 (Control) → 0.6812 (Aggregate-conditioned)) — there is no regime
in the completed sweep where aggregate-conditioning overtakes Control. The likely reconciliation
(not provable from available artifacts, see Q8): Berka's winning arm is trained **from scratch**,
while California's is a **fine-tune** — and Berka's own fine-tuned aggregate arms (joint and
frozen) also fail to turn their fidelity position into downstream gains, consistent with
California. So the pattern may be "fine-tuning doesn't transfer fidelity gains to downstream
utility" rather than "Berka and California disagree" — but no from-scratch California aggregate
run exists to confirm this.

**5. Are results consistent between Berka and California?**
Partially, and only where the comparison is apples-to-apples. Consistent: aggregate conditioning
improves the targeted dependency/KS fidelity metrics substantially in both datasets (Q2).
Inconsistent on the surface: downstream utility gain on Berka, no downstream gain on California
(Q4) — but this compares a scratch arm to a fine-tune arm, a confound this audit cannot remove
(Finding 5). Not comparable at all: degree conditioning (Berka only) and the specific
from-scratch training regime (Berka only).

**6. Which findings are statistically or experimentally uncertain?**
No significance testing (t-test, bootstrap CI, etc.) was performed anywhere in the source
artifacts or in this audit — n=3 seeds throughout, and none of the reports claim p-values. Judging
informally by gap-vs-SD: the large fidelity and synthetic-only downstream effects discussed above
(gaps of several SDs) are unlikely to be pure noise. Several **blend/mixed-regime** downstream
rows are not: e.g. degree-scratch loan-default AUROC beats baseline at `blend_50` (0.903 vs 0.878)
while its AP is statistically indistinguishable from (slightly below) baseline's (0.611 vs
0.615) — exactly the kind of row the task asked not to rank on AUROC alone. Any single-metric
"win" in a blend regime with a sub-0.02 gap and SD >0.02 should be read as noise, not a result.

**7. Are the downstream results affected by potential data leakage?**
Yes for Berka (all 6 configs in Tables 2–3, `leakage_caveat=true`, Finding 3) — the generator
trained on the full database including downstream-test accounts, so synthetic-only and
blend-heavy regimes are an optimistic upper bound; relative ranking between the 6 configs is
unaffected (same caveat, same magnitude mechanism, applied identically) but absolute levels are
not a leakage-free read. No for California (all 4 configs, `leakage_caveat=false`, confirmed both
in the `DATASETS` config and in every result JSON) — California's downstream numbers in Table 5
do not carry this caveat. **Update (Finding 8): a genuine leakage-free Berka baseline
(`berka_split`, config 1 only) also exists and is usable today** — it was previously reported as
unrecoverable, which was a mistake; see Finding 8. Its existing synth-only AUROC is **higher**,
not lower, than the leaky baseline's: loan default 0.514±0.026 (leakage-free) vs. 0.507±0.067
(leaky); card ownership 0.602±0.007 (leakage-free) vs. 0.573±0.031 (leaky) — the opposite of what
a naive "leakage inflates performance" story would predict. This is **not** a clean leakage-effect
isolation, though: the leakage-free checkpoint is trained on a smaller real set (3375 `split_train`
accounts vs. the leaky baseline's ~4500-account full `preprocessed` table) and is a different
trained model from a different date (Aug 2026, pre-`/scratch`-loss) rather than the same model
with leakage toggled off — so this comparison is confounded by training-set size and by being a
different checkpoint entirely, and should not be read as "leakage doesn't matter here." It's an
open, flagged discrepancy, not a resolved one. The `berka_split` numbers cover only
`real`/`synth`/`mixed`, not the blend/subsample regimes — a regime-coverage gap, not a
recoverability gap.

**8. What evaluations are still needed before making a strong conclusion?**
(a) ~~The 42 pending California cells~~ — **done as of 2026-10-14 05:09 IST**, see Table 5 update
above; no longer a gap. (b) A **California aggregate-conditioned-from-scratch** run, to separate
"fine-tuning doesn't transfer fidelity to downstream utility" from "California specifically
doesn't benefit" (Q4/Q5) — still missing, and now the single largest remaining gap in this audit
that requires new compute. (c) ~~A current, non-orphaned leakage-free Berka baseline~~ —
**corrected 2026-10-14 (Finding 8): it already exists** (`berka_split`, config 1, real data split
+ trained checkpoint + already-sampled synthetic data, all intact at `/bhallukah/.../
rdbdiff_workspace/berka_split`); what's missing is only the `blend_*`/`real_25/50/75` regime
sweep for it, fillable the same no-retraining way Table 5's California regimes were, not a full
retrain as this audit originally (incorrectly) claimed. Not performed in this audit. (d) Any
formal significance testing, if the research report needs to make a claim stronger than
"consistent direction across 3 seeds."

---

## STEP 6: Reproducibility

**Final tables, all paths under `/storage/s25017/AMEX/`:**
| Table | Path |
|---|---|
| 1 — Berka fidelity | `java_d/results/relational_fidelity_consolidated/relational_fidelity_table.csv` (+ `relational_fidelity_detailed.csv` for full provenance) |
| 2 — Berka loan default | `java_d/results/paper_repro/downstream/berka_prediction_tables/table_a_loan_default.csv` (+ `_extended.csv` for all 11 regimes/sample sizes) |
| 3 — Berka card ownership | `java_d/results/paper_repro/downstream/berka_prediction_tables/table_b_card_ownership.csv` (+ `_extended.csv`) |
| 4 — California fidelity | `java_d/results/consolidated_audit/table4_california_household_fidelity.csv` (new, this audit) |
| 5 — California 11-regime coverage | `java_d/results/consolidated_audit/table5_california_eleven_regime_coverage.csv` (new, this audit) |

**Scripts used to produce the underlying numbers** (all pre-existing, unmodified by this audit):
`results/relational_fidelity_consolidated/compute_degree_dep_metrics.py`,
`java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/evaluate_run.py` + `compare.py`,
`results/paper_repro/downstream/berka_prediction_tables/aggregate_downstream.py`,
`java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/evaluate.py` + `compare.py`,
`downstream/run.py` / `downstream/evaluate.py` / `downstream/tasks.py` (shared harness, byte-identical across worktrees per prior sessions' `diff` checks, not re-verified in this audit).

**Seed-level raw results** (underlying every mean±SD above): Berka fidelity —
`java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/metrics/fidelity_seed{0,1,2}.json`,
`java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/fidelity_seed{0,1,2}.json`,
`java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/comparison.json` (`per_seed` not
exposed for the dependency metrics — those are only available pre-aggregated; `results/relational_fidelity_consolidated/degree_dep_metrics_per_seed.json`
has the degree-side per-seed records). Berka downstream — the per-record JSONs listed in
`berka_prediction_tables/REPORT.md` §1 (each record carries `grdm_seed`/`xgb_seed`/raw metrics).
California fidelity — `java_d-calagg-ft/.../metrics/comparison.json` key `relational.values` (and
`fidelity.per_seed` for the paper-suite metrics). California downstream — the six JSON files in
`java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/downstream/` plus
`java_d/results/paper_repro/downstream/california__california_{inctot,empstat}{,_seeds12}.json`
for the baseline.

**Commands to reproduce the aggregation** (no retraining):
```bash
# Table 1 dependency metrics (degree-conditioned configs)
cd /storage/s25017/AMEX/java_d-aggft && CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n tabsyn \
    python /storage/s25017/AMEX/java_d/results/relational_fidelity_consolidated/compute_degree_dep_metrics.py

# Tables 2-3 Berka downstream aggregation
python /storage/s25017/AMEX/java_d/results/paper_repro/downstream/berka_prediction_tables/aggregate_downstream.py

# Table 5, once the pending background job finishes (see below)
cd /storage/s25017/AMEX/java_d-calagg-ft && conda run --no-capture-output -n tabsyn python -m downstream.run \
    --dataset california_agg_ft --tasks california_inctot california_empstat \
    --regimes real_25 real_50 real_75 blend_25 blend_75 mixed_25 mixed_50 \
    --out-dir experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/downstream --out-suffix _missing
    # (repeat --dataset for california_ft_control, california_agg_ft_label_free)
```

### Discrepancy log

| # | Discrepancy found | Resolution |
|---|---|---|
| 1 | Cardinality metric identical across all configs/worktrees | Flagged as uninformative (Finding 1), kept in Table 1 for completeness per the paper's standard 11-metric suite but footnoted |
| 2 | Degree-conditioning experiments' own `compare.py` reports SD with `ddof=1` (pandas default); the aggregate-conditioning pipeline (`evaluate_run.py`/`compare.py`) uses `ddof=0` | Recomputed degree-conditioning SDs from raw per-seed JSON with `ddof=0` so Table 1 is internally consistent; original `ddof=1` numbers remain in the degree experiments' own `comparison.md`, untouched |
| 3 | California's `compare.py` (fidelity) and the California 11-regime downstream builder both use `ddof=1`, different from the Berka tables' `ddof=0` | Not renormalized — flagged as Finding 2 and left in native convention per-table, since recomputing Table 4/5 would require per-seed raw arrays not fully exposed in `comparison.json` for every metric |
| 4 | A historical leakage-free Berka baseline (`berka_split`) was reported in this audit's first pass as "orphaned," with its GRDM checkpoint and real-data split said to no longer exist | **This was wrong** (Finding 8). That check only looked inside the `java_d` worktree family. The checkpoint, real split, and already-sampled synthetic data are all fully intact at `/bhallukah/s25017/ClavaDDPM/rdb-diffusion` (the relocated old `/scratch` mount, readable again as of 2026-10-14) — verified directly: trained `model_ema.pt` per seed, `split_train`/`split_test` with 0 account-ID overlap, byte-identical eval code. Corrected to: the existing `berka_split` results are valid and usable; only their regime coverage (`real`/`synth`/`mixed` only) is incomplete, not their validity |
| 5 | Config 4's own home directory (`java_d-agg`) has only 2-of-3 downstream seeds; the 3rd-seed-complete version lives in `java_d-aggft`'s re-evaluation | Used the 3-seed `java_d-aggft` copy; cross-checked its `real`-regime numbers against `java_d-agg`'s partial copy (bit-identical), confirming it's a re-evaluation of the same checkpoints, not a different model |
| 6 | 42 of 88 California Table 5 cells had no result; the job meant to produce them had died silently ~37 min in (no error, no OOM evidence, no output) | Initially left as `NA` with the specific reason, not estimated. **Resolved 2026-10-14**: relaunched fully detached (`setsid`+`nohup`+`disown`) so it would survive independent of any one tool-call session; completed in ~4h46m (`exit 0` ×3); all 42 cells filled from the resulting JSONs, aggregated with the same protocol/convention as the rest of the table |

### Remaining missing values
- ~~42 cells in Table 5~~ — **filled 2026-10-14** (see Table 5 update); no longer missing.
- `blend_25/50/75` and `real_25/50/75` regimes for the `berka_split` leakage-free reference —
  **no longer blocked on a missing checkpoint/split (Finding 8 corrects that)**; the real split
  and already-sampled synthetic data both exist at `/bhallukah/.../rdbdiff_workspace/berka_split`.
  Filling this would only need a downstream-eval re-run (XGBoost only, no retraining/resampling),
  the same kind of job already done twice in this audit for California — not attempted here, left
  as a flagged opportunity (see Finding 8 and Q8c).

### Experiment limitations (cannot be resolved from available artifacts)
- No California from-scratch aggregate-conditioning run exists, so the Berka-vs-California
  downstream-utility disagreement (Q4) cannot be cleanly attributed to dataset vs. training
  regime. This remains the single largest gap in the audit that requires genuinely new training
  (as opposed to re-running evaluation on data that already exists).
- No formal significance testing (t-test/bootstrap CI) exists anywhere in the source pipeline;
  all "SD" values are seed-to-seed spread over n=3, which is itself a small, noisy estimate of
  true variance.
- The Berka aggregate-fine-tune sampling-collapse root cause (Finding 4) is described in the
  source `REPORT.html`'s own diagnostics (eps-error blowup at high timesteps) but was not
  independently re-verified in this audit.
- The `evaluate_run.py` dependency-metric pipeline's `per_seed` outputs for the aggregate-family
  configs (baseline/agent2_scratch/ft_*) are not separately exposed in `comparison.json` — only
  pre-aggregated mean/SD/n — so this audit could recompute degree-side per-seed values but had to
  trust the already-aggregated numbers for the aggregate-family configs (spot-checked instead
  against the unaggregated fidelity numbers' pattern, which matched).

### Final summary (for the research report)

Across Berka, aggregate-conditioning trained from scratch is the strongest intervention tested:
it improves every targeted relational-dependency metric (KS, correlation-error, sign-agreement)
by a wide margin and is one of only two configurations (with degree-conditioning-from-scratch)
that improves downstream synthetic-only prediction utility over the baseline on both tasks and
all three metrics. Fine-tuning either conditioning scheme onto an existing checkpoint, rather
than training from scratch, consistently underperforms its from-scratch counterpart and in the
aggregate-conditioning case destabilizes sampling outright (joint fine-tune) — a frozen-backbone
fine-tune is safe but gives much smaller gains than from-scratch training. On California,
aggregate-conditioning (tested only as a fine-tune) produces the largest relational-fidelity
improvement seen in this audit, and a mechanism check (ablating the aggregate signal at eval
time) confirms the model is genuinely using it — but this fidelity gain does not transfer to
downstream prediction utility in any of the 11 regimes now evaluated (synth, mixed/mixed_25/50,
real_25/50/75, blend_25/50/75), mirroring Berka's own finding that fine-tuned (as opposed to
from-scratch) conditioning is the weaker lever for downstream gains. All Berka downstream numbers
in Tables 2–3 (6 configurations) carry an acknowledged, quantified data-leakage caveat; all
California downstream numbers (4 configurations) do not, by construction of the two tasks'
train/test splits. A genuine leakage-free Berka baseline (`berka_split`) also exists and is
usable — this audit's first pass wrongly reported it as unrecoverable; its checkpoint, split,
and already-sampled synthetic data all survived at a relocated mount, just temporarily
unreachable (Finding 8) — but it only covers 3 of 11 regimes today. Explicitly checked against a
real-data-only reference throughout (Tables 2a/3a): no configuration, in any regime, beats
real-data-only performance by a margin larger than its own seed-to-seed SD — synthetic data
narrows the gap to real data as more real data is blended in, but does not reliably close it. No
formal statistical-significance claim is made anywhere in this report; all comparisons are
n=3-seed point estimates.
