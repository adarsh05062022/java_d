# Consolidated Berka relational-fidelity table (5 GRDM configurations)

Answers the request to complete the relational-fidelity table across:

1. **Original GRDM baseline** (`paper_repro`)
2. **Degree-conditioned GRDM, trained from scratch**
3. **Degree-conditioned GRDM, fine-tuned from the original checkpoint**
4. **Aggregate-conditioned GRDM, trained from scratch**
5. **Aggregate-conditioned GRDM, fine-tuned from the original checkpoint**

No model was retrained. Everything below is either read directly from existing experiment
outputs, or computed fresh in this session (2026-10-13) by re-running existing, unmodified
evaluation code against existing synthetic-database CSVs. No checkpoint or experiment output
file was overwritten anywhere.

## Where each configuration actually lives

| # | Configuration | Worktree | Experiment dir |
|---|---|---|---|
| 1 | Original GRDM | `/storage/s25017/AMEX/java_d` | `rdbdiff_workspace/berka/paper_repro/seed_{0,1,2}` |
| 2 | Degree from scratch | `/storage/s25017/AMEX/java_d_degcond` | `experiments/GRDM_DEGREE_CONDITIONED_SCRATCH` |
| 3 | Degree fine-tuned | `/storage/s25017/AMEX/java_d_degft` | `experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE` |
| 4 | Aggregate from scratch | `/storage/s25017/AMEX/java_d-agg` | `experiments/GRDM_AGGREGATE_CONDITIONED_SCRATCH` |
| 5 | Aggregate fine-tuned | `/storage/s25017/AMEX/java_d-aggft` | `experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE` |

All five configurations train/evaluate against the **same real data**
(`data/berka/preprocessed`, which is the identical symlink target
`/storage/s25017/AMEX/complex_data/berka/preprocessed` in every worktree — checked with
`readlink -f`) and the **same evaluation code** (`downstream/data.py`, `downstream/tasks.py`,
and `rdbdiff/third_party/clava_ddpm/gen_multi_report.py` are byte-identical across all five
worktrees — checked with `diff`). Seeds 0/1/2 are used throughout; every cell below has n=3.

Config 5's own directory already contains a unified evaluation script
(`experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/evaluate_run.py`) that scores
**configs 1 and 4 as well** (as `baseline` and `agent2_scratch`) with one protocol, so those
three configurations' full 11-metric rows were already sitting in
`java_d-aggft/.../metrics/comparison.json` / `comparison_refs.json` / `comparison_final.json`.
Configs 2 and 3 (degree-conditioned) had metrics 1–7 (the paper's own fidelity suite) but were
**missing all four dependency metrics** (aggregate KS, parent↔aggregate correlation error,
aggregate↔aggregate correlation error, strong-dependency sign agreement) — nobody had pointed
that evaluation code at the degree-conditioned synthetic databases yet. That gap is what this
session filled; see "What was actually computed in this session" below.

## Main table

| Metric | Original GRDM | Degree from scratch | Degree fine-tuned | Aggregate from scratch | Aggregate fine-tuned¹ |
|---|---|---|---|---|---|
| Cardinality | 99.65 ± 0.05 | 99.65 ± 0.05 | 99.65 ± 0.05 | 99.65 ± 0.05 | 99.65 ± 0.05 |
| Column-shape fidelity | 96.997 ± 0.107 | 96.023 ± 0.094 | 90.907 ± 1.096 | 96.443 ± 0.160 | 84.457 ± 0.776 |
| Intra-table trends | 98.310 ± 0.043 | 97.506 ± 0.176 | 94.831 ± 0.600 | 97.520 ± 0.094 | 88.253 ± 0.442 |
| 1-hop relational fidelity | 91.487 ± 0.200 | 92.148 ± 0.035 | 88.877 ± 0.555 | 92.700 ± 0.049 | 80.913 ± 2.176 |
| 2-hop relational fidelity | 95.610 ± 0.071 | 95.050 ± 0.171 | 91.101 ± 0.964 | 95.017 ± 0.092 | 84.413 ± 1.554 |
| 3-hop relational fidelity | 92.710 ± 0.449 | 92.653 ± 0.586 | 92.988 ± 0.906 | 93.047 ± 0.772 | 92.853 ± 0.550 |
| Average 2-way fidelity | 95.263 ± 0.052 | 94.935 ± 0.114 | 91.681 ± 0.697 | 95.063 ± 0.045 | 85.067 ± 0.576 |
| Aggregate KS distance (↓ better) | 0.2728 ± 0.0012 | 0.2037 ± 0.0013 | 0.2199 ± 0.0030 | 0.1425 ± 0.0054 | 0.3390 ± 0.0239 |
| Parent–aggregate correlation error (↓ better) | 0.1413 ± 0.0031 | 0.1213 ± 0.0023 | 0.1859 ± 0.0116 | 0.0863 ± 0.0034 | 0.2103 ± 0.0190 |
| Aggregate–aggregate correlation error (↓ better) | 0.2447 ± 0.0016 | 0.1800 ± 0.0026 | 0.2463 ± 0.0172 | 0.0760 ± 0.0049 | 0.2836 ± 0.0572 |
| Strong-dependency sign agreement, \|r\|≥0.1 (↑ better) | 0.6250 ± 0.0794 | 0.7639 ± 0.0196 | 0.6944 ± 0.0494 | 0.8843 ± 0.0458 | 0.7546 ± 0.0797 |

All cells: mean ± population standard deviation (`numpy.std`, ddof=0) over the 3 trained seeds
(0, 1, 2). Fidelity metrics (rows 1–7) are on the paper's 0–100 scale; dependency metrics
(rows 8–11) are on their native 0–1 scale. The sign-agreement denominator (parent-attribute ×
child-aggregate pairs with \|real Spearman r\| ≥ 0.1) is **72 in every single seed and every
configuration** — it is fixed by the real data, not by the model, which is what makes this
metric directly comparable across configurations.

¹ **Aggregate fine-tuned = `ft_same_spec`, the joint (unfrozen-backbone) fine-tune using Agent
2's 31-aggregate spec — the literal "fine-tuned from the original checkpoint" configuration,
and the one whose aggregate spec matches config 4 exactly.** Per the experiment's own
`REPORT.html`, **this specific run generates corrupted data at sampling time**: up to 78% of
account rows and 77% of transaction rows fall outside the real data's range in model space,
with the instability already present 20 epochs into the joint fine-tuning phase. The numbers
above were computed with the identical protocol as every other column, so they are
*mechanically* comparable — but they measure a sampling failure mode for this arm, not a clean
read of "what fine-tuning does to relational fidelity." A frozen-backbone counterpart of the
same fine-tune does **not** collapse and stays close to baseline; see the appendix below. If
your downstream use of this table is "did fine-tuning help or hurt," read the appendix before
drawing that conclusion from this column alone.

## Per-metric documentation

| # | Metric | n (seeds) | Evaluation script / function | Source file |
|---|---|---|---|---|
| 1–7 (all 5 configs) | Cardinality, Column Shapes, Intra-Table Trends, 1/2/3-HOP, AVG 2-WAY | 3 | `rdbdiff/third_party/clava_ddpm/gen_multi_report.py:gen_multi_report()` — the paper's own relational-fidelity evaluator, invoked automatically by `main.py` at the end of every sampling run, byte-identical in all five worktrees | baseline/agg-scratch/agg-finetune: `java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/comparison.json` (built by `evaluate_run.py` + `compare.py`). degree-scratch: `java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/metrics/fidelity_seed{0,1,2}.json`. degree-finetune: `java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/fidelity_seed{0,1,2}.json`. |
| 8–11 (all 5 configs) | Aggregate KS, parent↔aggregate corr. error, aggregate↔aggregate corr. error, sign agreement | 3 | `evaluate_run.py:account_view()` + `aggregate_metrics()` in `java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/` — builds a one-row-per-account view (own columns + loan columns + `trans`/`order` aggregates + disp/card counts) from each config's real/synthetic `synthetic_database` CSVs, then takes KS distance per aggregate column and Spearman-correlation deltas between real and synthetic account views | baseline/agg-scratch/agg-finetune: same `comparison.json` / `comparison_refs.json` / `comparison_final.json` as above (already computed before this session). degree-scratch/degree-finetune: **computed fresh in this session** — `compute_degree_dep_metrics.py` (this folder) imports `account_view`/`aggregate_metrics`/`load_syn`/`check_schema` unmodified from that same `evaluate_run.py` and runs them against the already-generated synthetic databases at `java_d_degcond/.../workspace/seed_{0,1,2}/synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database` and the equivalent `java_d_degft` path. Raw per-seed and summary output: `degree_dep_metrics_per_seed.json`, `degree_dep_metrics_summary.json` in this folder. |

All cells are directly comparable: same real reference data, same code, same seeds, same fixed
schema (72 strong dependencies, same aggregate-column set) in every configuration — with the
one documented caveat on the "Aggregate fine-tuned" column above.

One precision note on degree-scratch/degree-finetune rows 1–7: those two experiments' own
`compare.py` scripts report a slightly different standard deviation in their own
`metrics/comparison.md` (e.g. Column Shapes "96.02 ± 0.12" for degree-scratch) because they use
pandas' sample standard deviation (ddof=1). The `evaluate_run.py` pipeline that produced the
baseline/aggregate numbers uses `numpy.std` (ddof=0, population std). To keep every column in
this table on the same convention, the degree-scratch/degree-finetune means and SDs above were
**recomputed in this session directly from the per-seed `fidelity_seed{0,1,2}.json` files** with
`numpy.std` (ddof=0) rather than copied from those experiments' own comparison.md.

## What was actually computed in this session (not previously available anywhere)

Only the four dependency metrics (rows 8–11) for **degree from scratch** and **degree
fine-tuned** were missing before this session — confirmed by `grep`-ing every `.md`/`.json`
file in both experiment directories for "unavailable"/aggregate/cross_table/sign-agreement
language (no hits) and by inspecting their `metrics/` directories directly (only
`fidelity_seed{N}.json`, with rows 1–7, existed).

These were recovered **without retraining** by reusing `evaluate_run.py`'s
`account_view()`/`aggregate_metrics()` code unmodified and pointing it at the synthetic
databases that config 2/3's own training+sampling runs had already written to disk:

- `java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/workspace/seed_{0,1,2}/synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database/`
- `java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/workspace/seed_{0,1,2}/synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database/`

Schema/FK integrity (`evaluate_run.py:check_schema()`) was verified to pass (`schema_ok=True`,
`fk_integrity=True`) for all 6 synthetic databases (3 seeds × 2 configs) before the aggregate
metrics were trusted. Command used:

```bash
cd /storage/s25017/AMEX/java_d-aggft
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n tabsyn python \
    /storage/s25017/AMEX/java_d/results/relational_fidelity_consolidated/compute_degree_dep_metrics.py
```

## Appendix: other aggregate fine-tune arms (not in the main table)

The aggregate fine-tune experiment ran four arms, all n=3, all scored with the identical
protocol. The main table uses `ft_same_spec`; the other three are given here for completeness
and because they materially change the "did fine-tuning help" conclusion:

| Metric | ft_same_spec (main table) | ft_label_free | ft_same_spec_frozen | ft_label_free_frozen |
|---|---|---|---|---|
| Column-shape fidelity | 84.457 ± 0.776 | 82.467 ± 0.476 | 96.880 ± 0.078 | 96.853 ± 0.087 |
| Intra-table trends | 88.253 ± 0.442 | 87.073 ± 0.821 | 98.137 ± 0.056 | 98.080 ± 0.085 |
| 1-hop relational fidelity | 80.913 ± 2.176 | 79.227 ± 1.575 | 91.183 ± 0.246 | 91.227 ± 0.068 |
| 2-hop relational fidelity | 84.413 ± 1.554 | 83.223 ± 0.370 | 95.660 ± 0.188 | 95.613 ± 0.184 |
| 3-hop relational fidelity | 92.853 ± 0.550 | 92.170 ± 0.476 | 92.630 ± 1.221 | 92.650 ± 1.258 |
| Average 2-way fidelity | 85.067 ± 0.576 | 83.797 ± 0.246 | 95.173 ± 0.167 | 95.147 ± 0.130 |
| Aggregate KS distance | 0.3390 ± 0.0239 | 0.3718 ± 0.0424 | 0.2574 ± 0.0060 | 0.2588 ± 0.0052 |
| Parent–aggregate correlation error | 0.2103 ± 0.0190 | 0.2302 ± 0.0102 | 0.1387 ± 0.0006 | 0.1377 ± 0.0020 |
| Aggregate–aggregate correlation error | 0.2836 ± 0.0572 | 0.3018 ± 0.0059 | 0.2205 ± 0.0086 | 0.2260 ± 0.0155 |
| Strong-dependency sign agreement | 0.7546 ± 0.0797 | 0.6852 ± 0.0991 | 0.7130 ± 0.0429 | 0.7222 ± 0.0227 |

- `ft_same_spec` / `ft_label_free`: joint fine-tune (unfrozen backbone), 31 / 27 aggregates. Both
  collapse at sampling (see caveat above).
- `ft_same_spec_frozen` / `ft_label_free_frozen`: the end-of-phase-1 snapshot (backbone frozen,
  only the new `agg_projs` head trained). Does not collapse; stays within noise of the original
  GRDM baseline, with a small, real improvement in aggregate KS (0.257–0.259 vs. 0.273) and a
  small drop in parent–aggregate correlation error (0.137–0.139 vs. 0.141) — i.e. a small, safe
  gain, not the large gain seen from training the aggregate conditioning from scratch (config 4).
  Source: same `comparison.json`, keys `ft_same_spec_frozen` / `ft_label_free_frozen`.

## Metrics that remain unavailable

**None of the 11 requested metrics are unavailable for any of the 5 configurations** — every
cell in the main table has n=3 and a named source. The only thing flagged as not simply usable
at face value is the "Aggregate fine-tuned" column's sampling-collapse caveat documented above;
that is a data-quality finding about that specific arm, not a missing-metric situation, and the
frozen-backbone counterpart in the appendix gives the uncorrupted comparison if you need one.

## Files in this folder

- `README.md` — this write-up
- `relational_fidelity_table.csv` — the main table, 5 columns, mean ± sd per cell
- `relational_fidelity_detailed.csv` — long format: one row per (metric, configuration) with mean, sd, n, script/function, source file, comparability note
- `degree_dep_metrics_summary.json` / `degree_dep_metrics_per_seed.json` — raw output of the dependency-metric computation run in this session
- `compute_degree_dep_metrics.py` — the (read-only) script that produced them
