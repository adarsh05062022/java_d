# Stage 2 — Phase B diagnostic results

2026-10-15 · all new outputs under `results/stage2/` · code in `java_d-stage2/stage2/{e0,eval}` ·
machine-readable: **`metrics/e0_results.csv`** (one row per arm, built by `stage2/eval/e0_table.py`),
`e0a/out_of_range.json`, `e0a/*/x0clip_stats.json`, `metrics/*.json`, `e0{a,b}/downstream/*.json`,
`table5/TABLE5_eleven_regimes.{csv,md}`

Labels: **VERIFIED** (computed from on-disk outputs, method stated), **PRELIMINARY** (one generator
seed / one sample: every E0 number below is PRELIMINARY in this sense), **NOT RUN**.

**Statistical caveat for all of E0.** Each arm is **one sample from one checkpoint (seed 0)**. The
"± SD" is over XGBoost seeds only, so it measures classifier noise, not generator noise. Stage 1
measured the original GRDM's loan TSTR AUROC SD across generator seeds at 0.067, and the loan test set
has **19 positives**. Loan-AUROC differences below ~0.1 between single samples are therefore not
interpretable. Card (220 positives) and fidelity/mechanism metrics are much tighter.

---

## B1. California Table 5 — **complete (88/88 cells)**

* The relaunched job (`run_missing_regimes_v2.sh`, started 2026-10-14 00:23) finished at 05:09, with all 3 arms
  exit 0 (`logs/missing_regimes_master_v2.log`).
* The 6 new files each have 63 records (7 regimes × 3 generator seeds × 3 XGBoost seeds). There are no duplicate keys
  and no non-finite values, and every record matches its log line to 4 decimals (VERIFIED, all 6 files).
* `real_*` regimes reproduce the baseline exactly, as they must (no generator involved).
* Full table: `table5/TABLE5_eleven_regimes.{csv,md}` (built by `table5/build_table5_complete.py`).
  The pre-rebuild table is kept in `table5_preserved/`.

**Paired differences vs original GRDM** (same generator seed, mean over 3 XGBoost seeds; mean over 3
seeds; count of negative seeds):

| arm | metric | synth | blend_25 | blend_50 | blend_75 | mixed_25 | mixed_50 | mixed |
|---|---|---|---|---|---|---|---|---|
| control (+20k steps) | income R² | −0.0001 (2/3) | +0.0003 (2/3) | +0.0004 (0/3) | +0.0001 (1/3) | +0.0001 (1/3) | +0.0000 (2/3) | +0.0000 (2/3) |
| aggregate FT | income R² | −0.0021 (3/3) | −0.0016 (3/3) | −0.0006 (3/3) | −0.0003 (3/3) | −0.0001 (2/3) | −0.0004 (3/3) | −0.0009 (3/3) |
| label-free aggregate FT | income R² | −0.0038 (3/3) | −0.0021 (3/3) | −0.0008 (3/3) | −0.0004 (3/3) | −0.0002 (3/3) | −0.0005 (3/3) | −0.0010 (3/3) |
| control | employment F1 | −0.0005 (2/3) | −0.0005 (3/3) | −0.0002 (3/3) | −0.0002 (2/3) | −0.0001 (2/3) | −0.0003 (3/3) | −0.0002 (2/3) |
| aggregate FT | employment F1 | −0.0016 (3/3) | −0.0014 (3/3) | −0.0011 (3/3) | −0.0006 (3/3) | −0.0005 (3/3) | −0.0011 (3/3) | −0.0013 (3/3) |
| label-free aggregate FT | employment F1 | −0.0018 (3/3) | −0.0018 (3/3) | −0.0009 (3/3) | −0.0003 (2/3) | −0.0006 (3/3) | −0.0009 (3/3) | −0.0011 (3/3) |

**Implication (VERIFIED).**
* The two aggregate-conditioned fine-tunes are below the original in **82 of 84** seed-paired cells,
  across every synthetic-data regime. The deficit is small (≤ 0.004) but consistent.
* The matched control (same extra training, no aggregates) is indistinguishable from the original
  (signs mixed, |Δ| ≤ 0.0005).
* So the aggregate pathway, as fine-tuned, slightly **hurts** California utility, despite raising
  household-level fidelity (summary KS 0.149 → 0.054).
* This completes and confirms the Stage 1 finding F8: relational fidelity up, utility not up.

## B2. E0-a — x̂₀ clipping at sampling (Berka, existing checkpoints, seed 0, weights untouched)

**Design.**
* `stage2/e0/x0clip.py` clamps x̂₀ to the training range **in model (quantile-normal) space**, inside
  `GRDM.relational_reverse_step` only. That range is exactly [−5.1993, 5.1993] for every column.
* Decoding is unchanged.
* Structure, normalizers and sampler code are each experiment's own (structure byte-copied from paper_repro seed 0).
* The sampling noise draw differs from the original runs, so clipped vs unclipped also differ by one
  sample's randomness.
* **Compatibility note:** TabDDPM's `clip_denoised` argument is ignored by `gaussian_p_mean_variance`, so it
  could not just be switched on. A runtime patch was used instead (unit-tested: bit-identical when off;
  analytic clamped step when on). No `rdbdiff/` file was edited.

| arm (seed 0) | Column Shapes | AVG 2-WAY | loan rows \|x\|>6 | trans rows \|x\|>6 | loan default rate (real 0.111) | exactly-one-OWNER / indep. pred. | trans.balance ε² (real 0.553) | TSTR loan AUROC / AP | TSTR card AUROC / AP |
|---|---|---|---|---|---|---|---|---|---|
| original, unclipped | 96.85 | 95.19 | 1.9% | 0.07% | 0.050 | 0.722 / 0.721 | 0.020 | 0.583 / 0.178 | 0.591 / 0.200 |
| original, **clipped** | 96.86 | 95.26 | 0 | 0 | 0.056 | 0.726 / 0.727 | 0.019 | 0.466 / 0.123 | 0.603 / 0.205 |
| aggregate joint FT, unclipped | **84.43** | **84.90** | **44.9%** | **65.6%** | **0.359** | 0.806 / 0.812 | 0.941 | 0.547 / 0.137 | 0.626 / 0.226 |
| aggregate joint FT, **clipped** | **96.41** | **95.11** | 0 | 0 | **0.062** | 0.814 / 0.809 | 0.797 | 0.563 / 0.145 | 0.609 / 0.203 |
| degree FT, unclipped | **90.77** | **91.31** | **49.3%** | **16.7%** | **0.451** | 0.900 / 0.900 | 0.283 | 0.682 / 0.237 | 0.595 / 0.204 |
| degree FT, **clipped** | **96.80** | **95.39** | 0 | 0 | **0.069** | 0.899 / 0.900 | 0.098 | 0.638 / 0.200 | 0.604 / 0.208 |

Runtime: 3.6 h (original), 4.5 h (aggregate FT), 4.2 h (degree FT) on contended GPUs. No sampling failures.

**Findings.**
1. **x̂₀ clipping alone removes the fine-tuning collapse (PRELIMINARY, one seed, but large effects).**
   * Column Shapes: aggregate FT 84.4 → 96.4; degree FT 90.8 → 96.8. The original is 96.85, with a
     3-seed SD of 0.11.
   * Loan default rate: 0.36 → 0.06 and 0.45 → 0.07.
   * The out-of-range counts are 0 *by construction* (the final step returns the clamped x̂₀), so they
     are not evidence. The fidelity and marginal recovery is the evidence.
   * Stage 1's falsification test for the amplifier hypothesis was "if out-of-range rows stay above
     20%". It is met trivially and is replaced here by the fidelity criterion. **Verdict:** the
     first-steps amplification with unclipped x̂₀ is *sufficient* to produce the collapse. The
     fine-tuned weights themselves are close to usable.
2. **The gate for adopting clipping is passed on fidelity.**
   * The original's fidelity changes by +0.01 (Column Shapes) and +0.07 (AVG 2-WAY), within seed noise.
   * Its mechanism metrics are unchanged.
   * Its loan TSTR fell 0.583 → 0.466. This is a single sample with 19 test positives, against a
     generator-seed SD of 0.067: **inconclusive, flagged**. Card moved +0.012.
3. **Clipping does not make the fine-tunes learn relational structure.**
   * Both clipped fine-tunes stay at their independence prediction for OWNER (0.814 vs 0.809;
     0.899 vs 0.900).
   * The aggregate FT still over-couples balances (ε² 0.80 vs 0.55).
   * Neither beats the original on card or loan TSTR by more than noise.
   * Their failure was therefore **two** problems: sampling collapse (fixed by clipping) and a
     fine-tuning recipe that did not learn sibling coupling (unchanged; consistent with the Stage 1
     zeroed-aggregate probe).
4. **The clipping is active far beyond the first step.**
   * The share of rows whose x̂₀ was clipped is 100% at t = 1999 for every model (ᾱ₁₉₉₉ ≈ 0 makes
     x̂₀ huge by construction), and 20–80% at t < 1000 (`x0clip_stats.json`).
   * The late-step share is inflated by discrete columns whose training values sit exactly at ±5.1993.
   * Clipping is therefore a substantive change to the sampler, not a rare safety net. This is why
     it is treated as a separate track (D4) and not folded silently into HRD.

## B3. E0-b — sampling on the real structure (diagnostic only)

**Leakage statement.**
* **California:** `--fixed_structure` uses the **training split's** households. This holds no test
  information, but it is also not generatable at test time without the real data.
* **Berka:** the generators were trained on the full database, and the real structure includes the
  downstream test accounts. For card ownership, the synthetic table's **card labels are the real ones**
  (card existence is structure).
* **Berka E0-b numbers are not leakage-free synthetic-data performance** and appear only next to
  other Berka-leaky numbers.

| arm (seed 0) | structure | Col. Shapes | AVG 2-WAY | exactly-one-OWNER / indep. | balance ε² | loan child-agg retained | card child-agg retained | TSTR loan AUROC / AP | TSTR card AUROC / AP |
|---|---|---|---|---|---|---|---|---|---|
| original GRDM | sampled | 96.85 | 95.19 | 0.722 / 0.721 | 0.020 | 0.21 | 0.046 | 0.583 / 0.178 | 0.591 / 0.200 |
| original GRDM | **real** | — ¹ | — ¹ | 0.740 / 0.742 | 0.023 | 0.06 | 0.036 | 0.648 / 0.226 | 0.616 / 0.209 |
| aggregate scratch | sampled | 96.52 | 95.06 | 0.985 / 0.893 | 0.692 | 0.92 | 0.053 | 0.778 / 0.325 | 0.635 / 0.230 |
| aggregate scratch | **real** | 96.94 | 96.09 | 0.996 / 0.901 | 0.676 | 0.91 | 0.118 | **0.848 / 0.599** | **0.678 / 0.241** |
| *real data* | — | — | — | 1.000 / 0.903 | 0.553 | 1 | 1 | 0.939 / 0.762 | 0.874 / 0.480 |

¹ The original's real-structure sample was reused from `java_d_degft/.../diagnostic_real_structure/baseline`
(per-column scores only; Column Shapes is not directly comparable).

| California original GRDM, seed 0 | Col. Shapes | AVG 2-WAY | exactly-one-head / indep. | AGE ε² (real 0.291) | INCTOT ε² | head–spouse AGE r | INCTOT\|age×EDUC gap | TSTR income R² | TSTR employment F1 |
|---|---|---|---|---|---|---|---|---|---|
| sampled structure | 98.61 | 96.96 | 0.376 / 0.376 | 0.048 | 0.024 | 0.18 | 0.217 | 0.5249 | 0.6751 |
| **real (training) structure** | 98.57 | 96.95 | 0.378 / 0.376 | 0.046 | 0.024 | 0.17 | 0.218 | 0.5242 | 0.6761 |

Employment for the real-structure sample was scored on 2026-10-15
(`logs/ds_cal_original_realstruct_empstat.log`). The first scoring run had written income only.

**Findings.**
1. **Sibling dependence is not a structure problem (VERIFIED on 2 models × 2 datasets, one seed each).**
   * With the real structure, the original stays exactly at its independence prediction (Berka OWNER
     0.740 vs 0.742; California one-head 0.378 vs 0.376).
   * The aggregate-scratch model keeps its dependence with or without the real structure (0.996 vs 0.901).
   * The attribute-side architecture decides sibling coupling.
2. **California: structure is irrelevant at this resolution.** Every fidelity, mechanism and utility
   number moves by ≤ 0.002. California has a single relation, so "structure" is just household
   sizes, and GRDM's histogram sampler reproduces that marginal. **D3 is not justified for California.**
3. **Berka: structure adds utility on top of the attribute model, in the same direction for both
   models** (loan +0.065 / +0.070 AUROC; card +0.025 / +0.043).
   * Under the leaky protocol and one sample each, the loan differences are within noise. The card
     differences are 1–4× the XGBoost SD.
   * The card child-aggregate signal rises only to 0.118 even with the real structure **and**
     aggregate coupling (real AUROC 0.874; best synthetic 0.678).
   * The card dependency needs a two-level path (transactions → account → disposition → card) that
     no tested model has. Real structure alone cannot supply it.
   * D3 (attribute-conditioned structure) is **plausibly justified for Berka**, but only after a
     leakage-free baseline exists.

## B4. Leakage audit of the Berka protocol (VERIFIED-code)

| check | finding |
|---|---|
| split | `downstream/splits.py`: account-level split, test fraction 0.25, seed 0. `make_berka_split.py` asserts disjoint accounts and clients and no lost rows. The materialised split is byte-identical in `java_d-agg`, `java_d-aggft`, `java_d_degcond`, `java_d_degft` (`diff -rq`); it is **absent** from `java_d/data/berka/` |
| sizes / positives | loan default (status ∈ {B, D}): train 57/511, **test 19/171**. Card ownership: train 672/4,007, test 220/1,362 |
| generator training data | **every existing Berka generator** used `data/berka/preprocessed`, the full database incl. the 1,125 test accounts → **leaky** |
| model selection | final EMA weights only (`best_epoch` logged, unused); XGBoost fixed hyperparameters, no early stopping or tuning on test → no test-label use in selection |
| harness checks | `downstream/run.py` asserts no id overlap between **real** train and test. It does **not** check synthetic tables, so a real-structure synthetic database (E0-b) can contain test-account ids, as noted |
| leakage-free path | `configs/berka_split_fastsample.json` trains on `data/berka/split_train` and is compatible with the current code. **No such generator exists** (historical ones lost with `/scratch`). Retraining costs ≈ 19–38 h per seed on this node; **not started (needs approval)** |
| labelling | every Berka number in Stage 2 is labelled leaky. Old (leaky) and future (split_train) protocols must be reported in separate tables |

---

## Confirmed findings vs unresolved hypotheses

| item | status | evidence |
|---|---|---|
| H1 no generated group state | **confirmed** (code + data) | `GNNDenoiser` has no latent; independence prediction matched to ≤ 0.002 in every E0 arm without aggregates |
| H2 structure independent of attributes and of other relations | **confirmed** (code + data) | `random_graph_sampler.py:106-118, 245-255`; ρ ≈ 0 |
| H2 causes the sibling / role loss | **rejected** | E0-b: real structure leaves independence intact (2 datasets) |
| H2 costs Berka utility | **supported, not established** (leaky, 1 seed) | +0.025–0.07 AUROC with real structure, same sign for 2 models |
| H2 costs California utility | **rejected** | ≤ 0.001 R² |
| H3 no within-step sibling path | **confirmed** (code); cross-step relay negligible (data) | K = 1 |
| H4 amplifier exists (β = 0.999, unclipped x̂₀) | **confirmed** (code) | `model.py:347,353`; `gaussian_multinomial_diffsuion.py:46-62, 233-281` |
| H4 amplifier is sufficient for the FT collapse | **supported** (1 seed, large effect) | E0-a: Column Shapes 84.4 → 96.4, 90.8 → 96.8; loan rate 0.36/0.45 → 0.06/0.07 |
| clipping is neutral for the original | **fidelity yes; loan TSTR unresolved** | +0.01 Col. Shapes; loan AUROC −0.12 on one sample (19 positives) |
| FT recipes learned sibling coupling | **rejected** | clipped FTs still at independence |
| H5 PUMA as Gaussian ordinal | **confirmed** (code + domain file) | 233-code discrete, quantile-transformed |
| H5 is the full explanation of the household → member gap | **rejected** (Stage 1: ~40%) | |
| aggregate FT improves California utility | **rejected** (Table 5) | 82/84 paired cells negative, ≤ 0.004 |
| leaky Berka protocol | **confirmed** for every existing Berka generator | configs and logs |

## Reproducible commands

```bash
cd /storage/s25017/AMEX/java_d-stage2
CUDA_VISIBLE_DEVICES= conda run -n tabsyn python stage2/tests/run_tests.py          # 14 tests
# E0-a / E0-b sampling (one GPU each; ~3.6-4.5 h each on the contended node)
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 conda run --no-capture-output -n tabsyn python -u stage2/e0/run_e0a.py --arm original --mode column
#   ... --arm degft / --arm aggft ;  stage2/e0/run_e0b.py --arm cal_original / --arm berka_aggscratch
# scoring (CPU only): every downstream + mechanism job for E0
stage2/launch/score_e0.sh
conda run -n tabsyn python stage2/e0/out_of_range.py
python3 stage2/eval/e0_table.py        # -> results/stage2/metrics/e0_results.csv
# Table 5 rebuild
python3 /storage/s25017/AMEX/java_d/results/stage2/table5/build_table5_complete.py
```

Seeds:
* Generator checkpoints: seed 0. The aggregate FT was trained with seed 1000, and its E0-a resample runs `main.py --seed 1000`.
* The degree-FT resample uses `manual_seed(2000)`, its own sampling seed.
* The original resample runs `main.py --seed 0`.
* XGBoost seeds: 0–4 (Berka) and 0–2 (California). The subsampling RNG is the harness's own.
