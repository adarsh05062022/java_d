# GRDM paper reproduction: Table 1, GRDM column

Written by `scripts/repro_summary.py`; do not edit by hand. Runs use `configs/<dataset>_fastsample.json`, `--exp_name paper_repro`, conda env `tabsyn`, launched by `scripts/resume_runs.sh` under `scripts/watchdog.sh`. Scores are the "Final results" block of each seed's log (copied to `logs/`). Δ is ours minus the paper; the last column is the first reproduction of August 2026 (`docs/REPRODUCTION.md` §2).

## berka (0/3 seeds finished)

| metric | seed 0 | seed 1 | seed 2 | ours | paper | Δ | Aug 2026 |
|---|---|---|---|---|---|---|---|
| Cardinality | pending | pending | pending | – | 99.65 ± 0.05 | – | 99.65 |
| Column Shapes | pending | pending | pending | – | 96.84 ± 0.14 | – | 96.89 |
| Intra-Table Trends | pending | pending | pending | – | 98.23 ± 0.03 | – | 98.26 |
| 1-HOP | pending | pending | pending | – | 91.41 ± 0.11 | – | 91.43 |
| 2-HOP | pending | pending | pending | – | 95.57 ± 0.04 | – | 95.71 |
| 3-HOP | pending | pending | pending | – | 92.43 ± 0.69 | – | 92.06 |

## california (0/3 seeds finished)

| metric | seed 0 | seed 1 | seed 2 | ours | paper | Δ | Aug 2026 |
|---|---|---|---|---|---|---|---|
| Cardinality | pending | pending | pending | – | 99.96 ± 0.01 | – | 99.96 |
| Column Shapes | pending | pending | pending | – | 98.75 ± 0.00 | – | 98.60 |
| Intra-Table Trends | pending | pending | pending | – | 97.35 ± 0.02 | – | 97.53 |
| 1-HOP | pending | pending | pending | – | 96.16 ± 0.05 | – | 96.48 |
