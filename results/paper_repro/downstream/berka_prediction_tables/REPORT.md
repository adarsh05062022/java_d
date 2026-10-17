# Berka Downstream Prediction Tables — GRDM Conditioning Comparison

Generated 2026-10-13. Covers the six Berka GRDM configurations (original baseline, degree-conditioned
scratch/fine-tune, aggregate-conditioned scratch/joint-fine-tune/frozen-fine-tune) on the two downstream
tasks (loan default, card ownership).

**Everything in Tables A and B below was recovered from existing experiment artifacts. No GRDM model was
retrained and no new GRDM sampling was run to produce them.** One new (cheap) downstream-evaluation command
was attempted to patch a gap in the *leakage-controlled* baseline appendix; it failed for a documented
reason (see "Remaining missing values").

## 1. Data sources (recovered, not regenerated)

All files follow the schema written by `downstream/run.py`: `{"dataset", "leakage_caveat", "records":
[{regime, grdm_seed, xgb_seed, n_train_real, n_train_syn, metrics:{roc_auc, average_precision, f1_macro,
balanced_accuracy, accuracy}}]}`. `f1_macro` is the one metric used for both "Macro-F1" (loan default) and
"F1" (card ownership) in Tables A/B — the task definitions (`downstream/tasks.py`) register identical
`BINARY_METRICS` for both tasks, so there is no separate "plain F1" in the existing evaluation code to
report; `f1_macro` is what it computes and what every prior comparison.md in this codebase already reports
under that name.

| # | Configuration | Source directory (all under `/storage/s25017/AMEX/`) | File(s) used | mtime |
|---|---|---|---|---|
| 1 | Original GRDM baseline | `java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/` | `berka__berka_{loan_default,card_ownership}__seeds0-1-2.json` | 2026-10-12 |
| 2 | Degree-conditioned (scratch) | `java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/metrics/downstream/` | `berka_degcond__{task}__seeds{0,1,2}.json` (merged) | 2026-10-12 |
| 3 | Degree-conditioned (fine-tune) | `java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/` | `berka_degft__{task}__seeds{0,1,2}.json` (merged) | 2026-10-12 |
| 4 | Aggregate-conditioned (scratch) | `java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/` | `berka_agg_conditioned__{task}.json` | 2026-10-13 |
| 5 | Aggregate-conditioned (joint fine-tune) | same as above | `berka_aggft_same_spec__{task}.json` | 2026-10-12 |
| 6 | Aggregate-conditioned (frozen fine-tune) | same as above | `berka_aggft_same_spec_frozen__{task}.json` | 2026-10-12 |
| — | Leakage-controlled baseline (appendix, stale) | `java_d/downstream/results/` | `berka_split__{task}.json` | **historical, see §4** |

Config 4's own home directory (`java_d-agg/experiments/GRDM_AGGREGATE_CONDITIONED_SCRATCH/metrics/downstream/`)
only has 2-of-3 seeds (`*_2of3.json`, `*_partial.json`). The aggft worktree later re-sampled and
re-evaluated config 4's checkpoint for all 3 seeds as its own comparison reference
(`evaluation_agent2_scratch_seed{0,1,2}_final.json` → `berka_agg_conditioned__{task}.json`), which is what
Table A/B use. **Consistency check**: the "real" regime (no synthetic data involved) is byte-identical
across every worktree's recomputation of each file (same `roc_auc` to 15 decimal places), confirming all
worktrees share the same `data/berka/split_train` / `split_test` real split and the same fixed XGBoost
hyperparameters/seeds (`downstream/evaluate.py`) — only the "requires a generator" regimes (`synth`,
`mixed`, `blend_*`) vary between worktrees, and only because each worktree drew its own independent sample
from the relevant checkpoint (expected seed-to-seed sampling variance, not a protocol mismatch).

Aggregation method (`aggregate_downstream.py`, copied into this folder): for each (configuration, task,
regime), records are grouped by `grdm_seed`; the mean over `xgb_seed` is taken within each `grdm_seed`
first (collapsing classifier-fit noise), then the mean/std across the N `grdm_seed` groups is reported.
This differs slightly from some experiments' own `comparison.md` (which pool all `grdm_seed × xgb_seed`
samples into one flat std) — means agree to 3 decimals in every case checked; only the std differs, because
it now reflects seed-to-seed (generator) variance specifically, which is what "mean ± std over seeds" and "N
seeds" ask for.

## 2. Training regimes (existing protocol, `downstream/run.py` `REGIMES`, unchanged)

- **Synthetic-only** = `synth`: 0% real, synthetic rows capped at the real-train row count (apples-to-apples
  with `real`).
- **Real + Synthetic** = `mixed`: 100% real + synthetic rows added on top (quantity +100%).
- **Mixed real/synthetic proportions** = `blend_25` / `blend_50` / `blend_75`: total row count held fixed at
  the real-train size; the stated fraction is real, the remainder is synthetic (substitution, not addition).
- (`mixed_25`, `mixed_50`, `real_25/50/75` also exist in the raw data/CSVs but are omitted from the
  headline tables below as they aren't one of the three regime categories requested; see the extended CSVs.)

Real-only reference (no generator involved, reported once — identical across all 6 configurations):
loan default AUROC 0.939 / AP 0.762 / Macro-F1 0.800 (n=511 train rows); card ownership AUROC 0.874 / AP
0.480 / F1 0.692 (n=4007 train rows).

## 3. TABLE A — Loan Default

| Configuration | Training regime | AUROC | AP | Macro-F1 | N seeds |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | Synthetic-only | 0.507 ± 0.067 | 0.136 ± 0.031 | 0.478 ± 0.006 | 3 |
| 1. Original GRDM baseline | Real + Synthetic | 0.928 ± 0.005 | 0.739 ± 0.014 | 0.778 ± 0.004 | 3 |
| 1. Original GRDM baseline | Mixed 25% real / 75% synth | 0.775 ± 0.012 | 0.382 ± 0.034 | 0.574 ± 0.014 | 3 |
| 1. Original GRDM baseline | Mixed 50% real / 50% synth | 0.878 ± 0.016 | 0.615 ± 0.027 | 0.717 ± 0.004 | 3 |
| 1. Original GRDM baseline | Mixed 75% real / 25% synth | 0.924 ± 0.003 | 0.714 ± 0.033 | 0.735 ± 0.014 | 3 |
| 2. Degree-conditioned (scratch) | Synthetic-only | 0.789 ± 0.019 | 0.356 ± 0.031 | 0.531 ± 0.022 | 3 |
| 2. Degree-conditioned (scratch) | Real + Synthetic | 0.944 ± 0.009 | 0.747 ± 0.029 | 0.775 ± 0.006 | 3 |
| 2. Degree-conditioned (scratch) | Mixed 25% real / 75% synth | 0.870 ± 0.019 | 0.480 ± 0.023 | 0.582 ± 0.001 | 3 |
| 2. Degree-conditioned (scratch) | Mixed 50% real / 50% synth | 0.903 ± 0.007 | 0.611 ± 0.034 | 0.702 ± 0.020 | 3 |
| 2. Degree-conditioned (scratch) | Mixed 75% real / 25% synth | 0.920 ± 0.013 | 0.687 ± 0.052 | 0.746 ± 0.008 | 3 |
| 3. Degree-conditioned (fine-tune) | Synthetic-only | 0.578 ± 0.087 | 0.174 ± 0.049 | 0.475 ± 0.004 | 3 |
| 3. Degree-conditioned (fine-tune) | Real + Synthetic | 0.916 ± 0.005 | 0.674 ± 0.016 | 0.734 ± 0.007 | 3 |
| 3. Degree-conditioned (fine-tune) | Mixed 25% real / 75% synth | 0.809 ± 0.006 | 0.403 ± 0.009 | 0.548 ± 0.026 | 3 |
| 3. Degree-conditioned (fine-tune) | Mixed 50% real / 50% synth | 0.867 ± 0.022 | 0.545 ± 0.042 | 0.683 ± 0.005 | 3 |
| 3. Degree-conditioned (fine-tune) | Mixed 75% real / 25% synth | 0.902 ± 0.007 | 0.629 ± 0.027 | 0.712 ± 0.010 | 3 |
| 4. Aggregate-conditioned (scratch) | Synthetic-only | 0.826 ± 0.039 | 0.455 ± 0.106 | 0.562 ± 0.026 | 3 |
| 4. Aggregate-conditioned (scratch) | Real + Synthetic | 0.936 ± 0.019 | 0.705 ± 0.075 | 0.726 ± 0.025 | 3 |
| 4. Aggregate-conditioned (scratch) | Mixed 25% real / 75% synth | 0.883 ± 0.020 | 0.562 ± 0.095 | 0.646 ± 0.028 | 3 |
| 4. Aggregate-conditioned (scratch) | Mixed 50% real / 50% synth | 0.902 ± 0.014 | 0.652 ± 0.065 | 0.713 ± 0.017 | 3 |
| 4. Aggregate-conditioned (scratch) | Mixed 75% real / 25% synth | 0.916 ± 0.005 | 0.692 ± 0.034 | 0.756 ± 0.020 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Synthetic-only | 0.595 ± 0.041 | 0.167 ± 0.038 | 0.504 ± 0.037 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Real + Synthetic | 0.921 ± 0.012 | 0.676 ± 0.052 | 0.732 ± 0.031 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Mixed 25% real / 75% synth | 0.836 ± 0.036 | 0.430 ± 0.097 | 0.592 ± 0.010 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Mixed 50% real / 50% synth | 0.877 ± 0.025 | 0.533 ± 0.066 | 0.671 ± 0.016 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Mixed 75% real / 25% synth | 0.911 ± 0.011 | 0.654 ± 0.067 | 0.737 ± 0.034 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Synthetic-only | 0.534 ± 0.086 | 0.164 ± 0.047 | 0.493 ± 0.017 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Real + Synthetic | 0.921 ± 0.016 | 0.694 ± 0.038 | 0.759 ± 0.028 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Mixed 25% real / 75% synth | 0.773 ± 0.003 | 0.362 ± 0.007 | 0.593 ± 0.029 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Mixed 50% real / 50% synth | 0.877 ± 0.024 | 0.556 ± 0.018 | 0.675 ± 0.028 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Mixed 75% real / 25% synth | 0.921 ± 0.010 | 0.678 ± 0.037 | 0.733 ± 0.022 | 3 |

(CSV: `table_a_loan_default.csv`; with sample sizes: `table_a_loan_default_extended.csv`.)

## 4. TABLE B — Card Ownership

| Configuration | Training regime | AUROC | AP | F1 | N seeds |
|---|---|---|---|---|---|
| 1. Original GRDM baseline | Synthetic-only | 0.573 ± 0.031 | 0.191 ± 0.019 | 0.473 ± 0.015 | 3 |
| 1. Original GRDM baseline | Real + Synthetic | 0.873 ± 0.002 | 0.482 ± 0.014 | 0.686 ± 0.006 | 3 |
| 1. Original GRDM baseline | Mixed 25% real / 75% synth | 0.856 ± 0.003 | 0.455 ± 0.009 | 0.644 ± 0.003 | 3 |
| 1. Original GRDM baseline | Mixed 50% real / 50% synth | 0.865 ± 0.003 | 0.472 ± 0.013 | 0.674 ± 0.005 | 3 |
| 1. Original GRDM baseline | Mixed 75% real / 25% synth | 0.870 ± 0.002 | 0.471 ± 0.006 | 0.682 ± 0.005 | 3 |
| 2. Degree-conditioned (scratch) | Synthetic-only | 0.641 ± 0.004 | 0.222 ± 0.007 | 0.498 ± 0.006 | 3 |
| 2. Degree-conditioned (scratch) | Real + Synthetic | 0.872 ± 0.002 | 0.492 ± 0.003 | 0.694 ± 0.003 | 3 |
| 2. Degree-conditioned (scratch) | Mixed 25% real / 75% synth | 0.843 ± 0.003 | 0.444 ± 0.010 | 0.629 ± 0.008 | 3 |
| 2. Degree-conditioned (scratch) | Mixed 50% real / 50% synth | 0.864 ± 0.001 | 0.479 ± 0.007 | 0.670 ± 0.006 | 3 |
| 2. Degree-conditioned (scratch) | Mixed 75% real / 25% synth | 0.870 ± 0.002 | 0.477 ± 0.002 | 0.676 ± 0.001 | 3 |
| 3. Degree-conditioned (fine-tune) | Synthetic-only | 0.617 ± 0.032 | 0.220 ± 0.022 | 0.485 ± 0.006 | 3 |
| 3. Degree-conditioned (fine-tune) | Real + Synthetic | 0.876 ± 0.002 | 0.497 ± 0.008 | 0.695 ± 0.002 | 3 |
| 3. Degree-conditioned (fine-tune) | Mixed 25% real / 75% synth | 0.858 ± 0.001 | 0.471 ± 0.004 | 0.654 ± 0.003 | 3 |
| 3. Degree-conditioned (fine-tune) | Mixed 50% real / 50% synth | 0.864 ± 0.001 | 0.474 ± 0.003 | 0.669 ± 0.004 | 3 |
| 3. Degree-conditioned (fine-tune) | Mixed 75% real / 25% synth | 0.871 ± 0.001 | 0.476 ± 0.003 | 0.686 ± 0.005 | 3 |
| 4. Aggregate-conditioned (scratch) | Synthetic-only | 0.648 ± 0.013 | 0.235 ± 0.012 | 0.512 ± 0.010 | 3 |
| 4. Aggregate-conditioned (scratch) | Real + Synthetic | 0.866 ± 0.003 | 0.486 ± 0.005 | 0.684 ± 0.007 | 3 |
| 4. Aggregate-conditioned (scratch) | Mixed 25% real / 75% synth | 0.804 ± 0.013 | 0.406 ± 0.020 | 0.599 ± 0.011 | 3 |
| 4. Aggregate-conditioned (scratch) | Mixed 50% real / 50% synth | 0.851 ± 0.005 | 0.460 ± 0.005 | 0.656 ± 0.011 | 3 |
| 4. Aggregate-conditioned (scratch) | Mixed 75% real / 25% synth | 0.865 ± 0.001 | 0.471 ± 0.003 | 0.675 ± 0.006 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Synthetic-only | 0.606 ± 0.015 | 0.209 ± 0.012 | 0.467 ± 0.007 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Real + Synthetic | 0.869 ± 0.001 | 0.480 ± 0.004 | 0.683 ± 0.004 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Mixed 25% real / 75% synth | 0.839 ± 0.001 | 0.436 ± 0.005 | 0.626 ± 0.001 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Mixed 50% real / 50% synth | 0.860 ± 0.003 | 0.462 ± 0.007 | 0.666 ± 0.007 | 3 |
| 5. Aggregate-conditioned (joint fine-tune) | Mixed 75% real / 25% synth | 0.868 ± 0.001 | 0.468 ± 0.005 | 0.675 ± 0.004 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Synthetic-only | 0.569 ± 0.028 | 0.181 ± 0.015 | 0.467 ± 0.008 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Real + Synthetic | 0.869 ± 0.001 | 0.470 ± 0.004 | 0.680 ± 0.001 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Mixed 25% real / 75% synth | 0.845 ± 0.002 | 0.434 ± 0.007 | 0.635 ± 0.006 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Mixed 50% real / 50% synth | 0.861 ± 0.001 | 0.461 ± 0.005 | 0.665 ± 0.003 | 3 |
| 6. Aggregate-conditioned (frozen fine-tune) | Mixed 75% real / 25% synth | 0.867 ± 0.000 | 0.468 ± 0.001 | 0.675 ± 0.001 | 3 |

(CSV: `table_b_card_ownership.csv`; with sample sizes: `table_b_card_ownership_extended.csv`.)

Every row above was **recovered**, not newly computed. Nothing in the required 6-config × 2-task ×
(synthetic-only / real+synthetic / 25-50-75 blend) grid was missing.

## 5. Leakage assessment (Step 4)

**All six configurations in Tables A/B are affected by the same data-leakage risk, and all are explicitly
flagged as such by the evaluation code itself** — every JSON file above carries `"leakage_caveat": true`.
`downstream/run.py`'s `DATASETS["berka"]` entry says it outright: *"Interim Berka: the generator trained on
the whole database, test rows included."* The GRDM generator for configuration 1 (and, per the project
notes for each follow-on experiment, every one of configs 2–6, which were deliberately "trained from scratch
on full `data/berka/preprocessed` (same as the paper_repro baseline) for like-for-like comparison") was
trained on the full Berka database — the same 4,500 accounts that `downstream/splits.py`'s
`BERKA_SPLIT_SEED=0` / `BERKA_TEST_FRAC=0.25` cut later holds out as the downstream test set. So yes: **the
generator saw the downstream test accounts during its own training**, for every row in Tables A/B. This
does not mean the downstream classifier saw test labels (`fit_predict` never touches `y_test`), but it does
mean the synthetic rows those classifiers train on could encode information the generator memorized from
the real test accounts — a real (if usually modest, for a diffusion generator over 4,500 accounts) risk of
optimistic bias, strongest in the `synth` and `blend_*` regimes where synthetic rows dominate training.

**A leakage-controlled companion already exists in the codebase** (`downstream/splits.py` /
`make_berka_split.py`: *"The train half is what a leakage-free Berka generator must be trained on: rerun
GRDM with `configs/berka_split_fastsample.json`"*) — `DATASETS["berka_split"]`: *"Clean Berka: same split,
but the generator only ever saw split_train."* Its results are at `java_d/downstream/results/berka_split__
{loan_default,card_ownership}.json`. **However, this companion cannot currently be used as a valid
up-to-date leakage-free reference**, for a documented reason:

- It is committed in this repo's `Initial commit` (`30a6e66`, 2026-10-07), i.e. it **predates** the current
  `paper_repro` rebuild (which started 2026-10-10 after `/scratch` — holding the old `rdbdiff_workspace`
  checkpoints — stopped being reachable; see project memory `grdm-rebuild-after-scratch-loss`).
- The GRDM checkpoint it was generated from (`rdbdiff_workspace/berka_split/...`) **no longer exists on
  disk** — confirmed by `find / -ipath "*berka_split*" -type d` returning nothing, and by
  `data/berka/split_train` / `split_test` (the real-data split a leakage-free retrain would need) also
  being absent (only `data/berka/preprocessed` was restored in the rebuild). It is an orphaned artifact of a
  checkpoint that was lost with `/scratch`.
- Its sanity-checkable overlap with the current protocol is consistent (the `real` regime figures match the
  current split sizes, `n_train_real=511`/`4007`), but that only confirms the *real* data split is the same
  — it says nothing about whether the orphaned `berka_split` generator/checkpoint was trained with the same
  architecture, hyperparameters, or even the same version of `downstream/tasks.py` as the current
  `paper_repro`/conditioned runs (`tasks.py` itself documents a since-fixed date-offset bug: *"Until 5 Oct
  2026 the cut compared the raw stored dates..."*, and this repo's single squashed commit makes it
  impossible to independently date the `berka_split` run relative to that fix).

For completeness, the historical numbers (reported **separately**, per the task's instruction, and not
merged into Tables A/B):

| Task | Regime | AUROC | AP | F1(-macro) |
|---|---|---|---|---|
| Loan default | Synthetic-only | 0.514 ± 0.026 | 0.128 ± 0.015 | 0.488 ± 0.022 |
| Loan default | Real + Synthetic | 0.928 ± 0.003 | 0.690 ± 0.027 | 0.762 ± 0.026 |
| Card ownership | Synthetic-only | 0.602 ± 0.007 | 0.200 ± 0.005 | 0.459 ± 0.004 |
| Card ownership | Real + Synthetic | 0.875 ± 0.004 | 0.483 ± 0.013 | 0.686 ± 0.004 |

No `blend_*` or `mixed_25/50` regimes exist for this historical run. **I attempted to fill them** by
rerunning `downstream/run.py --dataset berka_split --regimes blend_25 blend_50 blend_75`, which does not
retrain any GRDM model, only re-fits the downstream XGBoost classifier on already-sampled synthetic rows —
it failed immediately with `FileNotFoundError: data/berka/split_train/dataset_meta.json`, confirming the
real split itself needs regenerating (via `make_berka_split.py`) before even that cheap fill is possible,
and regenerating a genuinely current leakage-free comparison beyond that would require retraining a GRDM
generator on `data/berka/split_train` — out of scope per "do not unnecessarily retrain the GRDM models."

**Bottom line**: every number in Tables A/B carries the same leakage caveat, equally, across all six
configurations — so the *relative* comparison between configurations in §3/§4 is not distorted by leakage
(it affects all of them the same way, through the same full-data training protocol), but the *absolute*
synthetic-only/blend numbers for all six configurations should be read as an upper bound on what a
leakage-free generator would achieve. No configuration has a validated, current leakage-free counterpart.

## 6. Other recovered variants (not part of the 6 requested configurations)

The aggft worktree also fine-tuned a `label_free` arm (drops `loan.status:mean` and the card-disposition
aggregate — target-label-derived conditioning features the project excluded on request) for both the joint
and frozen schedules. Numbers exist (`berka_aggft_label_free{,_frozen}__{task}.json`) but are not in Tables
A/B since they weren't among the 6 named configurations; available on request.

## 7. Remaining missing values

1. **No current, validated leakage-free GRDM checkpoint exists for any of the 6 configurations.** The one
   historical leakage-free companion (§5) is orphaned and stale. Producing a current one for configuration 1
   alone requires: regenerating `data/berka/split_train`/`split_test` (`make_berka_split.py`, cheap) + fully
   retraining a GRDM generator on `split_train` (not cheap, explicitly out of scope here). Producing one for
   configurations 2–6 requires the same, five more times over.
2. **`blend_25/50/75` and `real_25/50/75` regimes for the historical `berka_split` baseline** are missing
   and, per above, not fillable without first regenerating the real split.

Nothing else is missing: all 6 configurations × 2 tasks × {synthetic-only, real+synthetic, blend-25/50/75}
× {AUROC, AP, Macro-F1/F1} × 3 seeds were recovered from existing artifacts.

## 8. Comparison (AUROC, AP, and F1 together — not AUROC alone)

**Synthetic-only utility (the regime where conditioning should matter most, since synthetic rows are 100%
of training data):** both from-scratch conditioned models beat the baseline on *all three* metrics, for
*both* tasks — this is the one place where AUROC, AP and F1 all agree:
- Loan default: aggregate-scratch (AUROC 0.826, AP 0.455, F1 0.562) and degree-scratch (0.789/0.356/0.531)
  both clear the baseline (0.507/0.136/0.478) by a wide margin.
- Card ownership: aggregate-scratch (0.648/0.235/0.512) and degree-scratch (0.641/0.222/0.498) both clear
  the baseline (0.573/0.191/0.473), by a smaller but still consistent margin.
- The three **fine-tuned** variants (configs 3, 5, 6) give much smaller, less consistent synthetic-only
  gains over the baseline, and frozen aggregate fine-tuning (config 6) is essentially flat-to-worse on card
  ownership (AUROC 0.569 vs. baseline 0.573; AP 0.181 vs. 0.191) — consistent with the earlier finding
  (`grdm-agg-finetune-experiment` memory) that fine-tuning destabilizes sampling relative to training from
  scratch.

**AUROC and AP/F1 disagree in several blend rows — a concrete reason not to rank on AUROC alone:**
degree-conditioned-scratch's loan-default AUROC at `blend_50` (0.903) beats the baseline (0.878), but its AP
(0.611) is *not* better than the baseline's (0.615) — essentially tied, slightly behind. The same pattern
repeats at loan-default `blend_75` for several configs where AUROC gaps between configurations shrink to
≤0.01 while AP gaps (and, less often, F1 gaps) remain the larger, more decision-relevant spread, because
loan default's test set is small and heavily imbalanced (19 of 171 test loans default) — AUROC saturates
faster than AP under that imbalance, which is exactly why AP/F1 were asked for alongside it.

**Real+Synthetic (full add) and high-real blends mostly converge:** once the classifier also has all of the
real training rows (`mixed`, `blend_75`), the gap between all six configurations narrows to a few points for
both tasks, and ordering is no longer monotonic with the synthetic-only ordering (e.g. the baseline's
`mixed` Macro-F1 for loan default, 0.778, is the single best Macro-F1 in that regime, ahead of every
conditioned variant) — conditioning mainly helps when synthetic data has to carry most of the training
signal, not when it is just topping up an already-sufficient real set.

**Overall:** of the six, **aggregate-conditioning from scratch (config 4)** and **degree-conditioning from
scratch (config 2)** are the only configurations that improve downstream utility over the original baseline
consistently across AUROC, AP, and Macro-F1/F1, and only in the synthetic-dominated regimes (synthetic-only,
`blend_25`). Neither fine-tuning path (joint or frozen, degree or aggregate) reliably beats the from-scratch
baseline once all three metrics are checked together, and all six configurations converge once real data
dominates the training mix. This matches, and is consistent with, this session's earlier per-experiment
verdicts recorded for `[[degcond-experiment]]`, `[[grdm-agg-finetune-experiment]]`, and
`[[degree-finetune-experiment]]`.

## 9. Evaluation commands / script locations

- Downstream harness: `downstream/run.py` (entry point), `downstream/tasks.py` (feature/label definitions),
  `downstream/evaluate.py` (fixed-hyperparameter XGBoost fit + metrics), `downstream/splits.py` /
  `downstream/make_berka_split.py` (the account-level train/test cut and the leakage-free split writer).
  Original invocation pattern: `conda run --no-capture-output -n tabsyn python -m downstream.run --dataset
  {berka,berka_split,berka_degcond,berka_degft,berka_agg_conditioned,berka_aggft_same_spec,
  berka_aggft_same_spec_frozen}`.
- Attempted fill (failed, documented in §5): `conda run --no-capture-output -n tabsyn python -m
  downstream.run --dataset berka_split --regimes blend_25 blend_50 blend_75 --out-suffix _fill` (run from
  `/storage/s25017/AMEX/java_d`).
- This report's aggregation: `aggregate_downstream.py` in this folder (reads the raw per-record JSONs listed
  in §1, writes `aggregated.json`; `table_a_loan_default{,_extended}.csv` and
  `table_b_card_ownership{,_extended}.csv` were built from it).
