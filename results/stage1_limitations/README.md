# Stage 1 diagnostics: GRDM limitations

Supporting computation for `reports/stage1_grdm_limitations.md` (2026-10-14).

Everything here is read-only with respect to earlier work. The scripts read synthetic CSVs and
model-space pickles that earlier experiments already wrote, plus the real data. They train nothing,
sample nothing, touch no checkpoint, and write only to `outputs/`.

| script | what it measures | output |
|---|---|---|
| `diagnostics/berka_diagnostics.py` | sampled-structure correlations; one-OWNER independence test; sibling coherence (eta², chance-corrected eps², residual after parent columns); feature → label AUROC inside each database; P(card \| balance octile); saturation and model-space out-of-range rows | `outputs/berka_diagnostics.json` |
| `diagnostics/california_diagnostics.py` | within-row, parent → child, group-size and sibling conditionals vs a real-test noise floor; one-head independence test; head–spouse correlations | `outputs/california_diagnostics.json` |
| `diagnostics/california_relationship_importance.py` | XGBoost feature-set ablation (own / + household / + co-member rollups), real vs synthetic training, real test | `outputs/california_relationship_importance.json` |
| `diagnostics/california_household_gap.py` | PUMA dependency (no model) and the PUMA vs other-household-columns R² split | `outputs/california_household_gap.json` |
| `diagnostics/schedule_amplification.py` | reverse-step ε-error amplification; per-table seed and high-t coverage under uniform seeding | `outputs/schedule_amplification.json` |
| `diagnostics/summarize.py` | renders every table used in the report | `outputs/summary_tables.md` |

Configuration → synthetic-data paths are in `diagnostics/common.py`. Berka tables use population SD
(ddof=0) and California tables sample SD (ddof=1), following each dataset's existing tables. Commands
are in Appendix B of the report.
