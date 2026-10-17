# Phase D pilot — HRD on California, seed 0 (results)

> **CORRECTION (2026-10-15 21:10).** The PUMA finding in §1, §3 and §5.4 below was **wrong**. The apparent
> restoration of η²(INCTOT; PUMA) in D1/D2 (0.073 / 0.060) is an **artefact of the runaway households**:
> 93.5% of them have PUMA pinned at code 0 or 232, and 96% of their members have INCTOT at 0 or 99.
> Without them, η² is 0.0037 / 0.0038, the same as GRDM (0.0041). The clipped resample (§7) gives the same
> answer. The pre-registered negative control ("PUMA dependency unchanged in every arm") therefore
> **holds**. The struck-through text is kept for the record.

2026-10-15 · one seed per arm · scored by `stage2/launch/score_phase_d.sh` · raw files
`metrics/phase_d/*.json`, `metrics/phase_d/downstream/*.json`, `metrics/phase_d/runaway_diag_seed0.json`

**Status of the evidence.** Each arm is **one trained model and one sample (seed 0)**. Mechanism
metrics have 3-seed SDs of ≤ 0.015 in every earlier arm, so differences of 0.1–0.5 are far outside
seed noise. Utility differences are not: GRDM's TSTR income R² has a generator-seed SD of 0.0025, so
one-seed R² differences below ~0.008 are **not established**. Primary analysis is **unclipped**, as
pre-registered.

## 1. Main table (vs original GRDM seed 0, same structure, normalizers, split and scorers)

| metric | real (train) | GRDM s0 | H0h (ordering only) | D1 (+ entity state) | D1+D2 (+ sibling attention) |
|---|---|---|---|---|---|
| **roles** exactly-one-head | 0.933 | 0.376 | 0.371 | 0.533 | **0.927** |
| … independence prediction | 0.541 | 0.376 | 0.377 | 0.536 | 0.540 |
| households with no head | 0.067 | 0.397 | 0.395 | 0.280 | **0.071** |
| **coherence** AGE ε² | 0.291 | 0.048 | 0.102 | 0.481 (over) | **0.290** |
| EDUC ε² | 0.238 | 0.037 | 0.043 | 0.428 (over) | 0.273 |
| INCTOT ε² | 0.076 | 0.024 | 0.030 | **0.356 (over)** | 0.128 (over) |
| head–spouse AGE r | 0.928 | 0.180 | 0.276 | 0.915 | **0.924** |
| head–spouse EDUC r | 0.681 | 0.084 | 0.103 | 0.693 | 0.624 |
| **within row** INCTOT \| age×EDUC gap (floor 0.158) | — | 0.217 | 0.250 | **1.017** | **1.134** |
| EMPSTAT \| age×SEX TV (floor 0.0049) | — | 0.0066 | 0.0070 | **0.030** | **0.033** |
| group size → INCTOT gap | — | 1.86 | 1.76 | 1.36 | 1.42 |
| **PUMA** η²(INCTOT; PUMA) | 0.062 | 0.004 | 0.002 | 0.073 ¹ | 0.060 ¹ |
| E[INCTOT \| PUMA] gap (floor 0.52) | — | 2.96 | 3.03 | 3.42 | 3.36 |
| **fidelity** Column Shapes | — | 98.61 | **99.13** | 98.20 | 98.32 |
| AVG 2-WAY | — | 96.96 | 97.66 | 96.03 | 96.07 |
| **stability** member rows \|x\| > 6 (model space) | — | 0.002% | 0.001% | **1.51%** | **1.37%** |
| max \|x\| | — | 269 | 3,252 | 906,735 | 1,066,434 |
| **utility** TSTR income R² (± SD over 3 XGBoost seeds) | 0.5527 | 0.5249 | 0.5201 | **0.5318** ± 0.0004 | 0.5283 ± 0.0005 |
| Δ vs GRDM s0 | | | −0.0048 | **+0.0069** | +0.0034 |
| TSTR employment macro-F1 | 0.6850 | 0.6751 | 0.6751 | 0.6697 | 0.6695 |
| Δ vs GRDM s0 | | | 0.0000 | −0.0054 | −0.0056 |
| ¹ *artefact of runaway households; 0.0037 / 0.0038 without them (see correction)* | | | | | |
| train / sample time (GPU, contended) | | — | 1.6 h / 1.1 h | 2.1 h / 1.1 h | 2.2 h / 0.9 h |

## 2. Pre-registered criteria (work order D2; checked for D1+D2, the arm they target)

| criterion | threshold | D1+D2 | pass |
|---|---|---|---|
| exactly-one-head | ≥ 0.85 | 0.927 | ✅ |
| … above independence prediction | ≥ +0.25 | +0.387 | ✅ |
| AGE ε² | 0.291 ± 0.03 | 0.290 | ✅ |
| INCTOT ε² | ≤ 0.096 | 0.128 | ❌ |
| head–spouse AGE r | ≥ 0.80 | 0.924 | ✅ |
| INCTOT \| age×EDUC gap | ≤ 0.25 | 1.134 | ❌ |
| EMPSTAT TV | ≤ 0.008 | 0.033 | ❌ |
| out-of-range rows | ≤ 2% | 1.37% | ✅ (but ~700× GRDM) |
| Column Shapes | ≥ 98.4 | 98.32 | ❌ (narrowly) |
| TSTR income R² vs GRDM, paired, 3 seeds | ≥ +0.005 | +0.0034 (1 seed) | ❌ / not testable yet |

**Score: 5 of 10 pass.** The relational criteria pass. Within-row fidelity, marginal fidelity and utility fail.

## 3. Pre-registered predictions vs outcome

| prediction (written before the run) | outcome |
|---|---|
| D1 restores co-member coherence but not roles | **Roles: confirmed.** One-head 0.533 vs its own independence prediction 0.536: D1's members remain conditionally independent given (parent, z). **Coherence: restored but heavily over-coupled** (INCTOT ε² 0.36 vs 0.076). This is worse than the bottom-up aggregates (0.113) |
| D2 restores roles | **Confirmed.** 0.927 vs real 0.933 (GRDM 0.376), and no-head households 0.071 vs real 0.067. Attention also calibrates AGE coherence (0.290) and head–spouse AGE (0.924) |
| Neither over-couples | **Failed** for D1 (all three ε²) and for D2 on INCTOT (0.128) |
| Neither reaches +0.005 income R² | **Failed for D1 on one seed** (+0.0069, ≈ 2.8 × GRDM's seed SD), not for D2 (+0.0034). Employment F1 falls by 0.005 in both. The net utility effect is mixed and unestablished |
| PUMA dependency unchanged in every arm (negative control) | **Holds** (corrected; η² 0.0037–0.0038 without runaways, and with clipping). ~~**Failed, informatively.** η²(INCTOT; PUMA) returns to the real level in D1 (0.073) and D2 (0.060), from GRDM's 0.004. But the per-area means are *worse* (gap 3.4 vs 3.0), and adding synthetic PUMA still lowers XGBoost R². The **strength** of the area–income link is restored; **which** area gets which income is not~~ |

## 4. Diagnosis of the failures (diagnostic analyses, not replacements for §1)

**4.1 Runaway households (sampling instability, new in HRD).**
* 1.4–1.5% of member rows blow up in model space (|x| up to 10⁶), **whole households at once**.
* In 100% of these households the generated entity state z is itself out of range, against ~1% of
  normal households. The household row is out of range in 96–98%.
* The risk rises steeply with household size: 0.2% (1–4 members) → 1.5% (7) → 7–8% (9 members).
* The instability therefore originates in **phase 1 (entity-level) sampling** and spreads to all
  members through z. H0h has no z and shows almost none (0.001%).
* This is the Stage 1 amplifier (β₁₉₉₉ = 0.999, unclipped x̂₀) acting on a new, wide (40-d) parent
  vector, whose tail values (std of large groups) are rare in training.

**4.2 Within-row damage is half runaway, half model.** Dropping the 0.6% of households with a
runaway row (`runaway_diag_seed0.json`):

| | INCTOT \| age×EDUC gap | EMPSTAT TV | INCTOT mean shift |
|---|---|---|---|
| D1 all / without runaways | 1.017 / 0.571 | 0.030 / 0.023 | 0.777 / 0.131 |
| D2 all / without runaways | 1.134 / 0.581 | 0.033 / 0.028 | 0.371 / 0.400 |
| GRDM s0 (reference) | 0.217 | 0.0066 | 0.071 |

The rest (gap ≈ 0.57, 2.6× GRDM) is a genuine degradation. **Interpretation (hypothesis):** members
lean on the household summary z, so a member's income follows its household's mean rather than its
own age and education. This is the same signature as the INCTOT over-coupling (ε² 0.36 / 0.13).

## 5. What the pilot establishes

1. **Explicit within-group interaction (D2) is what fixes roles.** A generated entity state alone
   (D1, the ClavaDDPM-type mechanism) does not: its members remain exactly as independent as the
   theory predicts. This is the sharp, falsifiable contrast the design was built to test, and it
   came out as predicted on the role metrics (large effect, one seed).
2. **D1 alone over-couples badly; D2 largely calibrates it** (AGE ε² 0.48 → 0.29), except for INCTOT.
3. **The new architecture has two engineering defects, both now diagnosed:** (a) entity-phase
   sampling runaways; (b) loss of within-row conditionals. Until they are fixed, the relational gains
   cost marginal and within-row fidelity, and the utility effect cannot be judged.
4. ~~**Stage 1's PUMA attribution needs revising.** The *strength* of the area → income dependence
   was lost because GRDM had no household-level income carrier. It returns with z. Its
   *area identity* is still lost. The latter is what the D5 analog-bit run tests (running; ETA ~00:15).~~
   **Corrected:** HRD does not change the PUMA → income dependency (η² 0.004 once the runaway
   households are removed, or with clipping). Stage 1's attribution stands, and D5 tests it directly.

## 6. Proposed next steps (none started; each needs approval)

| step | purpose | cost | note |
|---|---|---|---|
| **A. x̂₀-clipped resample of D1 and D2** (existing checkpoints) | does D4 clipping remove the phase-1 runaways and the runaway half of the within-row damage, without losing the role/coherence gains? | ~2 GPU-h (sampling only) | **Deviation, disclosed:** the pre-registered rule was to run this only above 2% out-of-range rows (observed 1.4–1.5%). It is proposed because §4.1 identified a specific mechanism that clipping targets. Reported as secondary |
| B. Seeds 1–2 of H0h, D1, D2 | the pre-registered expansion rule is met (D2 moves one-head by 0.55, ≫ 5× SD) | ~20 GPU-h | better after A, so the extra seeds use the sampler that will be reported |
| C. Remaining baselines: aggregate scratch (B) and K = 2 GRDM | needed before any D2 claim | ~60 GPU-h | |
| D. Over-coupling fix (e.g. label-free or mean-only z, lower `state_weight`) | the within-row loss not explained by runaways | ~6 GPU-h per variant | a new design iteration, so a separate ablation, not a silent change |

## 7. Step 1 — x̂₀-clipped resample of D1 and D2 (approved 2026-10-15 ~20:05; secondary analysis)

**Setup.** Same checkpoints, same structure, **same noise seed** as the primary samples, so each
clipped/unclipped pair differs only in the clamp of x̂₀ to ±5.1993 (model-space training range) in
both sampling phases (`sample.py --clip_x0 5.199337582290661 --tag _clip`). D1 sampled in 26.5 min,
D2 in about 30 min. Out-of-range rows are 0 **by construction** and are not used as evidence. Full
table: `metrics/phase_d/clip_comparison.{md,json}`.

| metric | GRDM s0 | D1 | D1 clipped | D1+D2 | **D1+D2 clipped** | real |
|---|---|---|---|---|---|---|
| exactly-one-head | 0.376 | 0.533 | 0.536 | 0.927 | **0.932** | 0.933 |
| AGE ε² | 0.048 | 0.481 | 0.455 | 0.290 | **0.284** | 0.291 |
| INCTOT ε² | 0.024 | 0.356 | 0.334 | 0.128 | 0.118 | 0.076 |
| head–spouse AGE r | 0.180 | 0.915 | 0.914 | 0.924 | **0.924** | 0.928 |
| INCTOT \| age×EDUC gap | 0.217 | 1.017 | 0.586 | 1.134 | **0.591** | (floor 0.158) |
| EMPSTAT TV | 0.0066 | 0.030 | 0.024 | 0.033 | 0.029 | (floor 0.0049) |
| η²(INCTOT; PUMA) | 0.004 | 0.073 ¹ | 0.004 | 0.060 ¹ | 0.004 | 0.062 |
| Column Shapes | 98.61 | 98.20 | 98.34 | 98.32 | **98.44** | — |
| AVG 2-WAY | 96.96 | 96.03 | 96.41 | 96.07 | 96.49 | — |
| households with z at the ±5.19 bound | — | 1.46% | 0.50% | 1.77% | 0.67% | — |
| TSTR income R² | 0.5249 | 0.5318 | **0.5329** | 0.5283 | 0.5295 | 0.5527 |
| TSTR employment F1 | 0.6751 | 0.6697 | 0.6700 | 0.6695 | 0.6693 | 0.6850 |

**Findings.**
1. **Clipping removes the runaway half of the damage and nothing else.**
   * The within-row income gap halves (1.13 → 0.59 for D2), matching the "drop runaway households"
     diagnostic (0.58).
   * Column Shapes rises to 98.44 for D2, now passing the ≥ 98.4 criterion. AVG 2-WAY gains +0.4.
2. **The relational gains are untouched.** D2 keeps roles (0.932), AGE coherence (0.284) and
   head–spouse correlation (0.924). Clipping is not what creates the dependence.
3. **The second failure is confirmed as a modelling problem.**
   * With no runaways left, the within-row income gap is still 0.59 (threshold 0.25; GRDM 0.217) and
     EMPSTAT TV is 0.029 (threshold 0.008; GRDM 0.0066).
   * INCTOT ε² stays over-coupled (0.118 vs 0.076).
   * This is the "members copy the household summary" problem. It needs a design change (step 2),
     not a sampler change.
4. **The income TSTR gain survives clipping** (D1 +0.0080, D2 +0.0046 vs GRDM s0, one seed each; GRDM
   seed SD 0.0025). The employment loss also survives (−0.005). Utility remains mixed and unestablished.
5. **Pre-registered criteria for D1+D2 clipped: 6 of 10 pass.**
   * Pass: one-head, its margin over independence, AGE ε², head–spouse, out-of-range (trivially),
     and Column Shapes.
   * Fail: INCTOT ε², the within-row income gap and EMPSTAT TV.
   * Not testable: paired utility over 3 seeds.

**Decision implication.** Adopt x̂₀ clipping as the HRD sampler for all further runs (it costs nothing
and only removes artefacts). This is a disclosed change from the pilot's primary (unclipped) analysis.
The next blocker is the over-coupling / within-row loss (step 2).

## 8. D5 — PUMA as analog bits (GRDM, otherwise unchanged; seed 0; approved 2026-10-15 14:28)

**Setup.**
* `stage2/d5/puma_bits.py` replaces PUMA by 8 binary columns. The unchanged GRDM `main.py` runs with
  the paper_repro settings (100k steps, unclipped, as GRDM).
* The sampled structure and the `individual` normalizer are byte-identical to GRDM seed 0.
* Decoding: 338 households (0.06%) had an invalid bit pattern and were mapped to the nearest valid
  code. All 233 codes occur.
* Scores come from the decoded database in the standard schema. GRDM's own log reports fidelity on
  the bit schema (Column Shapes 99.1), which is **not comparable** and is not used.

| metric | real (test floor) | GRDM s0 | D5 (PUMA bits) |
|---|---|---|---|
| P(PUMA) TV | 0.029 | 0.093 | **0.051** |
| E[INCTOT \| PUMA] gap | 0.52 | 2.96 | **2.57** |
| η²(INCTOT; PUMA) (real 0.062) | 0.062 | 0.0041 | 0.0066 |
| XGBoost R²: own → own+PUMA | 0.482 → 0.493 (real train, Stage 1) | 0.479 → 0.467 (**−0.012**) | 0.478 → 0.477 (**−0.001**) |
| XGBoost R²: own+hh | 0.533 (real train, Stage 1) | 0.507 | 0.506 |
| Column Shapes / AVG 2-WAY | — | 98.61 / 96.96 | 98.70 / 97.38 |
| INCTOT \| age×EDUC gap | (floor 0.158) | 0.217 | 0.284 |
| INCTOT \| VALUEH decile gap | (floor 0.120) | 1.10 | 1.42 |
| roles / coherence | 0.933 / 0.291 | 0.376 / 0.048 | 0.377 / 0.044 (unchanged) |
| rows \|x\| > 6 | — | 0.002% | 0.004% |
| TSTR income R² / employment F1 | 0.5527 / 0.6850 | 0.5249 / 0.6751 | 0.5228 / 0.6750 |

**Findings (one seed).**
1. **The representation fixes the harmful part.** Synthetic PUMA no longer misleads a model trained
   on it: adding it changes R² by −0.001 instead of −0.012. The PUMA marginal improves (TV
   0.093 → 0.051), and the per-area income means improve by 13%.
2. **It does not restore the area → income dependence.** η² is 0.0066 vs real 0.062, about 10×
   short. The dependence is weak, not just scrambled.
3. **No utility gain.** TSTR income R² −0.0021 (within GRDM's seed SD of 0.0025). Employment is unchanged.
   Two fidelity gaps get worse: income | age×EDUC 0.28 vs 0.22, and income | VALUEH 1.42 vs 1.10.
4. **Revises Stage 1 F9.** The ordinal encoding explains why synthetic PUMA is *harmful* (wrong area
   identity). It does not explain why the area → income link is *weak*. That part persists with a
   proper nominal representation. A plausible cause is again information flow: income is a member
   column, and members see the household's PUMA only through one noisy message-passing layer. That is
   untested. Combining bits with HRD's household-first ordering is the natural test.

## 9. Step 2 — what goes into the entity state (approved 2026-10-15 ~21:40; seed 0; x̂₀-clipped sampling)

Two variants of D1+D2, each differing from it **only** in the entity-state target (`state_exclude_columns`
/ `state_stats`; `stage2/hrd/configs/hrd_d2_{nolabel,meanonly}.json`):

* **no-label z:** mean and std of 13 member columns, without INCTOT and EMPSTAT (26 dims).
* **mean-only z:** means of all 15 member columns (15 dims).

Training took 1.6 h and 2.0 h; sampling 49 and 35 min. The table comes from
`stage2/eval/compare_arms.py` → `metrics/phase_d/step2_comparison.{md,json}`.

| metric | GRDM | D2 (pilot, unclipped) | D2 clipped | D2 no-label z | D2 mean-only z |
|---|---|---|---|---|---|
| exactly-one-head (real 0.933) | 0.3757 | 0.9272 | 0.9318 | 0.9418 | 0.9392 |
| indep. prediction | 0.3755 | 0.5398 | 0.5406 | 0.5522 | 0.5503 |
| AGE ε² (real 0.291) | 0.0483 | 0.2897 | 0.2835 | 0.2971 | 0.3014 |
| EDUC ε² (real 0.238) | 0.0374 | 0.2729 | 0.2648 | 0.2632 | 0.2578 |
| INCTOT ε² (real 0.076) | 0.0243 | 0.1282 | 0.1180 | 0.0865 | 0.1034 |
| head–spouse AGE r (real 0.928) | 0.1801 | 0.9240 | 0.9237 | 0.9226 | 0.9159 |
| INCTOT|age×EDUC gap (≤0.25) | 0.2171 | 1.1339 | 0.5912 | 0.5987 | 0.7279 |
| EMPSTAT TV (≤0.008) | 0.0066 | 0.0330 | 0.0289 | 0.0266 | 0.0262 |
| INCTOT mean shift | 0.0709 | 0.3707 | 0.4930 | 0.4372 | 0.6556 |
| group size → INCTOT gap | 1.8599 | 1.4216 | 0.7679 | 0.7665 | 1.0294 |
| η²(INCTOT; PUMA) (real 0.062) | 0.0041 | 0.0601 | 0.0038 | 0.0035 | 0.0056 |
| Column Shapes (≥98.4) | 98.61 | 98.32 | 98.44 | 98.46 | 98.33 |
| AVG 2-WAY | 96.96 | 96.07 | 96.49 | 96.53 | 96.56 |
| households with z at bound | — | 0.0177 | 0.0067 | 0.0054 | 0.0040 |
| TSTR income R² | 0.5249 | 0.5283 | 0.5295 | 0.5243 | 0.5278 |
| TSTR employment F1 | 0.6751 | 0.6695 | 0.6693 | 0.6742 | 0.6729 |
| criteria passed (of 7 mechanism) | 3 | 4 | 4 | 5 | 4 |

**Findings (one seed each).**
1. **Excluding the target columns from z removes the income over-coupling.**
   * INCTOT ε² falls 0.118 → 0.087 (passes ≤ 0.096; real 0.076).
   * Roles (0.942), AGE coherence (0.297) and head–spouse (0.923) are kept.
   * Column Shapes is 98.46 (pass). Employment F1 recovers to 0.6742 (GRDM 0.6751).
   * **5 of 7 mechanism criteria pass, the best arm so far.**
2. **The "members copy the household's income summary" hypothesis for the within-row loss is rejected.**
   The INCTOT | age×EDUC gap is unchanged (0.591 → 0.599), and EMPSTAT TV is essentially unchanged.
3. **The earlier income-TSTR gain disappears** with no-label z (0.5243 vs GRDM 0.5249). The D1/D2 gain
   (+0.003 to +0.008) therefore came with INCTOT in z, i.e. with over-coupled household income. It is
   not evidence of better income modelling.
4. **The mean-only z is worse** than the full z on the within-row gap (0.73) and on Column Shapes (98.33).
   Dropping the dispersion statistics is not the fix.
5. **The within-row gap tracks the global INCTOT mean shift** across all arms (shift 0.07 → gap 0.22 for
   GRDM; 0.44–0.66 → 0.60–0.73 for HRD). A diagnostic separating "marginal bias" from "lost
   conditionals" is in §10.

## 10. Diagnosis of the remaining within-row loss (2026-10-16; diagnostics, no new training)

**(a) Marginal bias or lost conditionals?** `stage2/eval/inctot_shift_diag.py` →
`metrics/phase_d/inctot_shift_diag.json`. It measures the Stage 1 INCTOT | age×EDUC cell-mean gap three
ways: as is; after removing the global mean shift; and after mapping synthetic INCTOT onto the real
marginal by rank, which removes any marginal difference and keeps the within-synthetic ordering.

| arm | gap | mean shift removed | marginal matched | EMPSTAT conditional TV | EMPSTAT marginal TV |
|---|---|---|---|---|---|
| real test (floor) | 0.158 | 0.161 | 0.159 | 0.0049 | 0.0014 |
| GRDM s0 | 0.217 | 0.251 | 0.233 | 0.0066 | 0.0053 |
| H0h | 0.250 | 0.278 | 0.253 | 0.0067 | 0.0023 |
| D2 clipped | 0.591 | 0.647 | 0.439 | 0.0289 | 0.0158 |
| D2 no-label z | 0.599 | 0.636 | 0.392 | 0.0266 | 0.0170 |
| D2 mean-only z | 0.728 | 0.811 | 0.435 | 0.0262 | 0.0182 |

Removing the mean shift does not help. Matching the marginal removes about a third of the gap. The
rest (0.39–0.44, about 2× GRDM) is a **genuine loss of the age × education → income conditionals**.
The EMPSTAT marginal is also off by about 3× GRDM. H0h shows none of this, so the loss comes from the
entity state / attention, not from entity-first ordering.

**(b) Which stage? Phase 2 run on the real households.**
`sample.py --teacher_force_parents --tag _tfparent` uses the real training households, their real
no-label entity state, and the real structure, and generates the members only (11 min; diagnostic,
training data only).

| hrd_d2_nolabel | INCTOT \| age×EDUC gap | EMPSTAT TV | INCTOT mean shift |
|---|---|---|---|
| generated households (normal sampling) | 0.599 | 0.0266 | 0.437 |
| **real households (teacher-forced)** | **0.340** | **0.0156** | 0.189 |
| GRDM s0 | 0.217 | 0.0066 | 0.071 |

About half of the excess comes from the generated entity level (phase 1). The other half is in the
member model itself, which is worse than GRDM even given perfect households.

**Step 3 (approved 2026-10-16 ~11:45):** entity-state dropout 0.3 in training
(`hrd_d2_nolabel_sdrop`), aimed at the member-model half. Results in §11.

## 11. Step 3 — entity-state dropout 0.3 (`hrd_d2_nolabel_sdrop`, seed 0): negative result

Table: `metrics/phase_d/step3_comparison.{md,json}`.

| | GRDM s0 | D2 no-label z | + state dropout 0.3 |
|---|---|---|---|
| exactly-one-head / AGE ε² / head–spouse r | 0.376 / 0.048 / 0.18 | 0.942 / 0.297 / 0.923 | 0.939 / 0.305 / 0.924 |
| INCTOT ε² (≤ 0.096) | 0.024 | 0.087 | 0.105 ❌ |
| INCTOT \| age×EDUC gap, generated households | 0.217 | 0.599 | **0.849** |
| … with real households (teacher-forced phase 2) | — | 0.340 | **0.344** |
| EMPSTAT TV, generated / teacher-forced | 0.0066 / — | 0.027 / 0.016 | 0.039 / 0.018 |
| households with z at the ±5.19 bound | — | 0.5% | **11.0%** |
| Column Shapes | 98.61 | 98.46 | 98.10 |
| TSTR income R² / employment F1 | 0.5249 / 0.6751 | 0.5243 / 0.6742 | 0.5232 / 0.6744 |

**Reading.**
1. Dropout did **not** improve the member stage it targeted: given real households, the gap is 0.344 vs 0.340.
2. It **destabilised entity-level sampling**: 11% of generated states hit the clip bound.
3. Net result: worse on generated data.

The member-stage within-row loss is therefore not caused by members over-relying on z. Remaining
candidates (untested):
* the hierarchical training distribution: members rarely see clean parents (t_P = 0) during
  training but always do in phase 2;
* member-head capacity shared with the sibling attention.

The first is cheap to test: oversample t_P = 0 for member groups in training.
