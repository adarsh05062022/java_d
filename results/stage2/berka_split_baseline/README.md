# Leakage-free Berka GRDM baseline (berka_split), verified and rescored 2026-10-16

**Source runs (historical, reused; not retrained):**
`/bhallukah/s25017/ClavaDDPM/rdb-diffusion/rdbdiff_workspace/berka_split/paper_repro/seed_{0,1,2}`.
They were trained on 2026-08-09/10 with the original GRDM code and `configs/berka_split_fastsample.json`:
data_path `data/berka/split_train`, n_hops 1, num_neighbors −1, sum aggregation, 200k steps,
T = 2000. Logs are in `logs/train/paper_repro_berka_split_seed{0,1,2}.log` of that repo.

## Leakage checks (all pass)

* **Split.** `data/berka/split_{train,test}` is byte-identical to `java_d-agg/data/berka/split_*`:
  an account-level split, test fraction 0.25, seed 0, disjoint accounts and clients.
* **Normalizers.** For every table and all 3 seeds, the stored QuantileTransformer equals a refit on
  `split_train` and differs from a refit on the full database. `district`, the shared 77-row
  dimension table, is identical either way by design. The `account` normalizer has 112 quantiles
  (3,375 // 30), against the 150 a full-database fit (4,500 // 30) would have.
* **Synthetic sizes.** They match `split_train` (3,375 accounts, 4,007 clients, ~0.79 M transactions),
  not the full database.
* **Harness.** `downstream/{tasks,evaluate,data,splits}.py` are identical to `java_d`. `run.py`
  differs only in the California CLAVA_ROOT path.

## Rescoring

The repo's own `downstream/results/berka_split__*.json` came from different dates (card 2026-08-10,
loan 2026-10-05) and predate or straddle the task-definition changes, so the baseline was rescored
with the current harness. Command:
`stage2/eval/downstream_eval.py --base berka_split --grdm-seeds 0 1 2` (XGBoost seeds 0–4, all 11 regimes).
Outputs are in `downstream/`, labelled `leakage_caveat: false`.

| task (test positives) | real | synth (TSTR) | mixed | blend_50 |
|---|---|---|---|---|
| loan default (19 / 171) | AUROC 0.939, AP 0.762 | **0.514 ± 0.026**, AP 0.128 | 0.928 ± 0.003 | 0.888 ± 0.022 |
| card ownership (220 / 1,362) | AUROC 0.874, AP 0.480 | **0.602 ± 0.007**, AP 0.200 | 0.875 ± 0.004 | 0.865 ± 0.003 |

± is the population SD over the 3 generator seeds, each the mean of 5 XGBoost seeds.

**Known issue in that repo.** `data/berka/preprocessed` is a dangling symlink to `/scratch`, which
is lost. It is not needed for scoring; retraining there would need it repointed to
`/storage/s25017/AMEX/complex_data/berka/preprocessed`.
