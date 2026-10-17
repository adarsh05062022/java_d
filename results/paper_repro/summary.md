# GRDM paper reproduction: Table 1, GRDM column

Written by `scripts/repro_summary.py`; do not edit by hand. Runs use `configs/<dataset>_fastsample.json`, `--exp_name paper_repro`, conda env `tabsyn`, launched by `scripts/resume_runs.sh` under `scripts/watchdog.sh`. Scores are the "Final results" block of each seed's log (copied to `logs/`). Δ is ours minus the paper; the last column is the first reproduction of August 2026 (`docs/REPRODUCTION.md` §2).

## berka (3/3 seeds finished)

| metric | seed 0 | seed 1 | seed 2 | ours | paper | Δ | Aug 2026 |
|---|---|---|---|---|---|---|---|
| Cardinality | 99.72 | 99.59 | 99.64 | 99.65 ± 0.07 | 99.65 ± 0.05 | +0.00 | 99.65 |
| Column Shapes | 96.85 | 97.04 | 97.10 | 97.00 ± 0.13 | 96.84 ± 0.14 | +0.16 | 96.89 |
| Intra-Table Trends | 98.25 | 98.35 | 98.33 | 98.31 ± 0.05 | 98.23 ± 0.03 | +0.08 | 98.26 |
| 1-HOP | 91.22 | 91.70 | 91.54 | 91.49 ± 0.24 | 91.41 ± 0.11 | +0.08 | 91.43 |
| 2-HOP | 95.67 | 95.51 | 95.65 | 95.61 ± 0.09 | 95.57 ± 0.04 | +0.04 | 95.71 |
| 3-HOP | 92.16 | 93.26 | 92.71 | 92.71 ± 0.55 | 92.43 ± 0.69 | +0.28 | 92.06 |
| AVG 2-WAY | 95.19 | 95.30 | 95.30 | 95.26 ± 0.06 | not reported | – | 95.25 |

## california (3/3 seeds finished)

| metric | seed 0 | seed 1 | seed 2 | ours | paper | Δ | Aug 2026 |
|---|---|---|---|---|---|---|---|
| Cardinality | 99.96 | 99.97 | 99.96 | 99.96 ± 0.01 | 99.96 ± 0.01 | +0.00 | 99.96 |
| Column Shapes | 98.61 | 98.61 | 98.50 | 98.57 ± 0.06 | 98.75 ± 0.00 | -0.18 | 98.60 |
| Intra-Table Trends | 97.54 | 97.56 | 97.38 | 97.49 ± 0.10 | 97.35 ± 0.02 | +0.14 | 97.53 |
| 1-HOP | 96.39 | 96.56 | 96.39 | 96.45 ± 0.10 | 96.16 ± 0.05 | +0.29 | 96.48 |
| AVG 2-WAY | 96.96 | 97.06 | 96.88 | 96.97 ± 0.09 | not reported | – | 97.00 |
