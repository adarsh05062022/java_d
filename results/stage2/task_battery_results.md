# Task-agnostic utility battery — Berka (leakage-free)

> **Correction (2026-10-16, later the same day):** the "HRD d2" model scored here never learned its
> entity state (state loss 1.0 for all 200k steps) and its account head was dead. See
> `berka_hrd_state_failure.md`. The HRD d2 numbers below are real observations of THAT model —
> sibling attention + size conditioning + a random shared per-account code — not of a working D1+D2.
> Interpretation points 2 and 4 are revised accordingly; the battery itself is unaffected.

Status: 2026-10-16. GRDM 3 seeds; HRD d2 **1 seed** (s1); HRD ordering-only (h0h) 1 seed (s0).
The remaining HRD seeds (d2 s0/s2, h0h s1/s2) are scored automatically as they finish
(`stage2/launch/task_battery_watch.sh`, log `metrics/task_battery/berka/logs/watch.log`).

## Why

The Berka loan/card tasks are two hand-picked targets. The goal is a generator whose data is useful for
tasks nobody has chosen yet, on databases it was not designed on. This battery replaces "two tasks" with
"every predictable quantity in the schema", built only from `dataset_meta.json` + `*_domain.json`.

## Method (code: `java_d-stage2/stage2/eval/task_battery.py`, `task_battery_summary.py`)

- **Targets** (every non-dimension table T): each column of T (35 total on Berka incl. structure);
  `struct_count` = log1p(#children of T in relation C); `struct_has` = 1[#children > 0] when the real rate
  is in [5%, 95%]. Discrete → top-10 classes + other (macro AUROC); continuous → real-train CDF (R²).
- **Feature views:** `own` (rest of the row), `up` (parent + grandparent rows and their other child
  tables' aggregates — never an aggregate over the path back to T), `sib` (leave-one-out aggregates over
  rows sharing the parent), `down` (child aggregates + grandchild counts), `rel` = up+sib+down, `all`.
  Struct targets drop every feature derived from the target relation. Leakage rules are unit-tested
  (`stage2/tests/test_task_battery.py`, 4 tests; suite 28/28 pass).
- **Protocol:** XGBoost, fixed hyper-parameters (depth 6, η 0.1, 300 rounds), ≤ 20k training rows from
  the evaluated database, scored on ≤ 20k rows of the real held-out test split (account-level split).
  Retention = (TSTR − chance) / (TRTR − chance), only on tasks where real data has signal
  (AUROC ≥ 0.6, R² ≥ 0.05). A synthetic training set with nothing learnable counts as retention 0.
- **Not pre-registered.** Thresholds and views were fixed before any synthetic database was scored and
  not changed afterwards, but this was not written down in advance; treat as exploratory.

## Results (observed)

Median retention per feature view (GRDM mean ± SD over 3 seeds; HRD single seeds):

| view | n tasks | GRDM | h0h (s0) | **HRD d2 (s1)** |
|---|---|---|---|---|
| own — same row | 15 | 0.991 ± 0.003 | 0.996 | 0.974 |
| up — parents | 20 | −0.050 ± 0.014 | −0.035 | **0.501** |
| sib — siblings | 11 | −0.664 ± 0.022 | −0.560 | **0.582** |
| down — children | 6 | −0.056 ± 0.026 | 0.008 | **0.720** |
| rel — all cross-row | 28 | −0.320 ± 0.109 | −0.232 | **0.513** |
| all features | 29 | 0.467 ± 0.128 | 0.574 | **0.931** |

Share of tasks (rel view) keeping ≥ 50% of the real signal: GRDM 7%, h0h 11%, HRD 50%; keeping
≤ 10%: GRDM 74%, h0h 79%, HRD 14%. Structure targets (rel view): GRDM median −1.12 (count) / 0.07 (has);
HRD 0.54 / 0.93.

Full per-task table: `metrics/task_battery/berka/summary.md`; rows: `summary_per_task.csv`.

## Interpretation

1. **GRDM's cross-table relations are not just weak, they are wrong.** Negative retention means a model
   trained on GRDM data, using relational features, predicts real data *worse than the constant*. Within-row
   relations are near perfect (0.99). This matches the Stage 1 finding (sibling/parent independence) on
   a much wider set of targets than loan/card.
2. **The as-run HRD's gain is general, not loan-specific.** It turns every cross-row family positive (parents,
   siblings, children, structure). Because its entity state was not learned, the gain must come from sibling
   attention + size conditioning (+ a shared random per-account code), not from D1.
3. **Ordering alone (h0h) does nothing**: the gain needs the entity state + sibling attention.
4. **HRD's cost is visible and specific.** Same-row retention drops slightly (mean-clipped 0.83 vs 0.97),
   concentrated in order (`amount` own 0.56, `k_symbol` own 0.51) and trans `balance` (0.77); and the
   **account table** regresses: `account_date` 0.08 (GRDM 0.75–0.99), `frequency` 0.29 (0.74–0.83),
   trans count from account 0.06 → −0.06 (GRDM 0.80–1.0). This is the same place the fidelity loss is
   (account column fidelity 0.62). Verified cause: the account head was dead / blind to its own input
   (sum aggregation over ~230 transactions), not the 34 state columns (`berka_hrd_state_failure.md`).
5. **Card:** the battery's "disp has a card" (which, unlike the curated task, keeps `operation`) is
   0.64 retained by HRD vs 0.04–0.07 for GRDM — HRD transmits the card ↔ card-withdrawal link through the
   disp → card edge and attention, but (curated task) not the card ↔ balance link (see the card
   diagnosis: the entity state cannot see grandchildren).

## Caveats

- HRD is one seed; the 3-seed numbers will replace these when the runs finish.
- 29 eligible tasks, not independent (several targets share features).
- Negative retentions are unbounded below for small real signal (e.g. trans.account rel, real R² 0.18);
  the clipped mean and keep50/lost shares are the robust summaries.
- Berka is the dataset HRD's multi-table version was built on. Generality to unseen databases
  (ccs, instacart_05, f1, movie_lens are available locally) is untested and needs GPU training of both
  GRDM and HRD there — not launched, needs approval.
