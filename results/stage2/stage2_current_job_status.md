# Stage 2 — status of the California Table 5 job

> **UPDATE 2026-10-15: COMPLETE.** All three arms exited 0 (05:03, 05:04, 05:09 IST on 2026-10-14;
> `logs/missing_regimes_master_v2.log`). The six `*_missing.json` files each hold 63 records, with no
> duplicates or non-finite values, and every record matches its log line (verified 2026-10-15). Table 5
> has **88/88 cells**: `results/stage2/table5/TABLE5_eleven_regimes.{csv,md}`. The earlier PENDING
> table is preserved in `table5_preserved/`. Summary and paired comparisons: `stage2_diagnostic_results.md` §B1.
> The snapshot below is kept unchanged as the record of the in-progress state.

Snapshot taken 2026-10-14 01:04–01:10 IST by read-only inspection (`ps`, `/proc/<pid>`, log files).
Nothing was signalled, restarted, relaunched or edited.

## 1. What is running

| item | value |
|---|---|
| job | the 42 missing cells of California "Table 5" (11 regimes × 4 configurations × 2 tasks = 88 cells; 46 were already populated) |
| launcher | `run_missing_regimes_v2.sh` (attempt 2), in another Claude session's scratchpad: `/tmp/claude-1070/-storage-s25017-AMEX-java-d/46e23495-657a-42f6-a440-67b32dd06429/scratchpad/`. A copy is preserved at `results/stage2/table5_preserved/run_missing_regimes_v2.sh` |
| launched | 2026-10-14 00:23:39 (`java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/logs/missing_regimes_master_v2.log`) |
| session / pgid | 425955 (own session via `setsid`, parent = init); this replaces attempt 1 (2026-10-13 15:42), which died at ~16:19 with no output file |
| working directory | `/storage/s25017/AMEX/java_d-calagg-ft` (branch `grdm-california-agg-finetune`, uncommitted) |
| Python PIDs | 426127 `california_agg_ft`, 426121 `california_agg_ft_label_free`, 426123 `california_ft_control` |
| command (each arm) | `conda run -n tabsyn python -u -m downstream.run --dataset <arm> --tasks california_inctot california_empstat --regimes real_25 real_50 real_75 blend_25 blend_75 mixed_25 mixed_50 --out-dir experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/downstream --out-suffix _missing` |
| environment | `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2`, `RDBDIFF_XGB_JOBS=2` (verified in `/proc/<pid>/environ`) |
| resources | **CPU only** (XGBoost `tree_method="hist"`, `downstream/evaluate.py:26-39`); ~187% CPU per process; **no GPU use** |
| seeds | generator seeds 0, 1, 2 (existing synthetic data under `rdbdiff_workspace/{california_agg_ft,california_agg_ft_label_free,california_ft_control}/paper_repro/seed_N`); XGBoost seeds 0, 1, 2; subsampling RNG `default_rng(1000·(grdm_seed+1)+xgb_seed)` (`downstream/run.py`) |
| configuration | no retraining, no resampling; reuses the existing synthetic databases |

## 2. Progress (01:03 IST)

Each arm needs 2 tasks × 3 generator seeds × 7 regimes × 3 XGBoost seeds = **126 fits**.

| arm | fits done | position |
|---|---|---|
| `california_agg_ft` | 39 / 126 | `california_inctot`, seed 1, last regime (`blend_75`) |
| `california_agg_ft_label_free` | 39 / 126 | same |
| `california_ft_control` | 39 / 126 | same |

**Output files already produced: none yet.** `downstream.run` writes its JSON only when a task
finishes (`run.py`, `json.dump` after `run_task`). The first files,
`downstream/<arm>__california_inctot_missing.json`, should appear when seed 2 of income finishes.
Until then the completed fits exist only as log lines in `logs/downstream_<arm>_missing.log`.

**Can partial results be reused?** Only after a task's JSON is written. The log lines carry the primary
metric to four decimals but not the secondary metrics, so a crash mid-task would lose that task's
work; the log is not a substitute for the JSON.

## 3. Estimated remaining time (from actual progress)

* Measured this run: income seed 0 took 1,308 s of fit time for the 7 regimes × 3 XGBoost seeds.
* The baseline's own log of the same regimes took 2,661 s for income and 9,334 s for employment (seed 0),
  so employment costs ≈ 3.5× income here as well.
* Remaining per arm: ≈ 0.4 income seeds (~10 min) + 3 employment seeds (≈ 3 × 76 min).

**ETA: income JSONs ~01:30 IST; employment JSONs and job completion ≈ 05:00–06:30 IST on 2026-10-14**,
depending on CPU contention from other users. This is an estimate, not a measurement.

## 4. Checks done

* **No errors** in any of the three arm logs; no tracebacks, no warnings about missing columns, and
  0 out-of-domain categorical cells in every synthetic seed loaded so far.
* **Not stalled**: all three logs were written at 01:03; processes are at ~187% CPU.
* **Consistency check (passed):** the `real_*` regimes subsample real rows only. Their values in this run
  equal the baseline's to the printed precision (e.g. `real_50` seed 0 XGBoost 0: R² 0.5477 here vs
  0.54766 in `results/paper_repro/downstream/california__california_inctot.json`; `real_25` 0.5392 vs
  0.53922; `real_75` 0.5509 vs 0.55090). So the job reproduces the harness's deterministic behaviour.
* **Risk:** the launcher lives in another session's `/tmp` scratchpad. The processes are detached
  (`setsid`, parent = init), so they should survive that session's end, unlike attempt 1. If the
  job dies again, rerun the preserved copy of the launcher; it writes only `*_missing.json` files and
  never overwrites the original 3-regime results.

## 5. After completion (Phase B1, not yet done)

1. Confirm 6 `*_missing.json` files (3 arms × 2 tasks), each with 63 records (3 seeds × 7 regimes × 3 XGBoost seeds).
2. Rebuild `eleven_regime_addendum/TABLE_B_eleven_regimes.{md,csv}` with the preserved
   `results/stage2/table5_preserved/build_table_b.py` (it already reads the `_missing` files) and
   check that the PENDING count goes from 42 to 0.
3. Check for duplicate (regime, grdm_seed, xgb_seed) keys and spot-check metrics against the log lines.

**Table 5 is not complete.** 46/88 cells populated; 42 in progress.
