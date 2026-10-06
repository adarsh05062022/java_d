# rdb-diffusion: project guide

This folder is a working copy of the GRDM code (upstream commit `bc0fb67`) plus the studies built
on it: the fidelity reproduction, the downstream-utility audit of Berka and California, the fix
program, and the dependency-fidelity checks. `README.md` is the upstream GRDM README; this file
maps everything added on top.

**Start here:** the full write-up of the study is `reports/grdm-final-report.html`.

## Layout

```
rdb-diffusion/
├── PROJECT_GUIDE.md        this file
├── README.md               upstream GRDM README
├── main.py                 train, sample and evaluate one GRDM run (upstream, locally patched)
├── pyproject.toml  .gitignore  .pre-commit-config.yaml
│
├── rdbdiff/                GRDM model package (upstream + local patches; gnn.py is new)
├── configs/                one JSON per dataset or variant: <variant>_fastsample.json
├── data/                   databases; data/berka/split_train + split_test is the leakage-free split,
│                           split_train_agg and split_train_reparam are the agg and reparam inputs
├── rdbdiff_workspace/      checkpoints and synthetic databases, one folder per variant (32 GB, git-ignored)
│
├── downstream/             downstream-utility harness (Python package)
│   ├── results/            one JSON per variant and task, plus the analysis outputs;
│   │                       _superseded_2026-10-05/ keeps the files the 5 Oct fixes replaced
│   ├── figures/            PNGs from the mechanism study
│   ├── relmlp/             DeepSets probe (has its own README)
│   └── results_relmlp/
├── dep_fidelity/           dependency-fidelity checks and constraint repair (Python package)
│   └── results/
├── sweep/                  unattended train → score → report runner, its manifest and state
├── scripts/                resume_runs.sh (launch or resume seeds), watchdog.sh (keep runs alive),
│                           rescore_downstream.sh (re-score every downstream result, no GPU)
│
├── reports/
│   ├── grdm-final-report.html   the consolidated report (current)
│   └── archive/            superseded pages, kept unchanged
├── docs/
│   ├── REPRODUCTION.md     fidelity reproduction and run instructions
│   ├── CHANGES.md          every local code change; CHANGES.patch is the diff as of 25 Aug
│   ├── downstream/         DIAGNOSIS.md, FIXES.md, RESULTS_downstream.md
│   ├── dep_fidelity/       REPORT.md (the checks), REPAIR.md (constraint repair)
│   └── references/         GRDM paper (NeurIPS 2025), AmEx report
└── logs/
    ├── train/              paper_repro_<variant>_seed<N>.log: training, sampling, fidelity evaluation
    ├── scoring/            downstream_*, decode_*, depfid_* scoring runs and report regeneration
    ├── analysis/           one-off experiments: mechanism study, flatten, DeepSets, repair ablations
    └── infra/              watchdog and sweep-launcher logs, lock and pid files
```

## Where to find things

| looking for | go to |
|---|---|
| The study's conclusions | `reports/grdm-final-report.html` |
| Fidelity scores of a run | `logs/train/paper_repro_<variant>_seed<N>.log` ("Final results"), or `eval_result.pkl` in the run folder |
| Downstream-utility scores | `downstream/results/<variant>__<task>.json`; tables in `docs/downstream/RESULTS_downstream.md` |
| Dependency-check scores | `dep_fidelity/results/*.json` |
| A synthetic database | `rdbdiff_workspace/<variant>/paper_repro/seed_<N>/synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database/` |
| What changed in the GRDM code | `docs/CHANGES.md` |
| Why each fix variant exists | `docs/downstream/FIXES.md` |

## Variants in rdbdiff_workspace

| status | variants |
|---|---|
| complete, 3 seeds | `berka` (full data, plus a `smoke_test` run), `berka_split` (the baseline), `berka_split_s2`, `_s3`, `_s5`, `_s7`, `_reparam`, `_reparam_s10`, `_discrete`, `california`, `ccs` |
| complete, 3 seeds, unusable | `california_discrete` (loss was NaN from the first step) |
| complete, 1 seed | `berka_split_s4`, `berka_split_s6`, `california_s2`, `_s3`, `_s4`, `_s5`, `_s7` |
| sampled, fidelity not evaluated | `berka_split_agg`, 3 seeds (the evaluation crashes on the `agg_*` columns, report §10.4) |
| partial | `berka_split_s12`: seed 0 complete, seeds 1–2 stopped at 83% |
| retired, checkpoints only | `berka_split_s8`, `_s9`, `_s10`, `_s11`, `_s13`, `_s12_wide`, `_reparam_s12`, `california_s12` |

## Running things

All commands run from this folder, through the shared conda environment `tabsyn`.

```bash
cd /scratch/s25017/ClavaDDPM/rdb-diffusion

# train or resume a variant's seeds (detached; resumes from checkpoints, skips finished seeds)
RDBDIFF_DATASET=berka_split scripts/resume_runs.sh 0 1 2
RDBDIFF_DATASET=berka_split_s3 RDBDIFF_EXTRA_ARGS="--seed_balance_alpha 0.5" scripts/resume_runs.sh 0 1 2

# keep the runs listed in watchdog.sh alive across the node's ~24 h process kill
setsid nohup scripts/watchdog.sh < /dev/null >> logs/infra/watchdog.log 2>&1 & disown

# unattended sweep over sweep/manifest.tsv (details in sweep/README.md)
setsid nohup ./sweep/run_sweep.sh < /dev/null >> sweep/run_sweep.log 2>&1 & disown

# score a variant's downstream utility, then rebuild the tables
conda run -n tabsyn python -m downstream.run --dataset berka_split_s3
conda run -n tabsyn python -m downstream.report

# after changing downstream/tasks.py: re-score every Berka variant and analysis (about 10 min)
setsid nohup scripts/rescore_downstream.sh < /dev/null >> logs/scoring/rescore.log 2>&1 & disown

# dependency checks on a finished run
conda run -n tabsyn python -m dep_fidelity.run --synthetic <run>/synthetic_database --label <name>
```

The full command list is in Appendix A of the report and in `docs/REPRODUCTION.md`.

## Generated files

Re-running these scripts overwrites the file they write:

| file | written by |
|---|---|
| `docs/downstream/RESULTS_downstream.md` | `downstream/report.py` (the sweep also runs it) |
| `docs/downstream/DIAGNOSIS.md` | `downstream/diagnose.py` |
| `reports/archive/downstream_report.html` | `downstream/make_artifact.py` |

## What moved on 5 October 2026

| before | now |
|---|---|
| `grdm-final-report.html` | `reports/` |
| `berka-utility-investigation.html`, `signal-death-study.html`, `six-column-test.html`, `downstream/downstream_report.html` | `reports/archive/` |
| `REPRODUCTION.md`, `CHANGES.md`, `CHANGES.patch` | `docs/` |
| `downstream/DIAGNOSIS.md`, `FIXES.md`, `RESULTS_downstream.md` | `docs/downstream/` |
| `dep_fidelity/REPORT.md`, `REPAIR.md` | `docs/dep_fidelity/` |
| the GRDM paper and AmEx report PDFs | `docs/references/` |
| `resume_runs.sh`, `watchdog.sh` | `scripts/` |
| `logs/paper_repro_*` | `logs/train/` |
| `logs/downstream_*`, `decode_*`, `depfid_*`, `score_agg_3seed`, `report_regen` | `logs/scoring/` |
| `logs/a1_wrapper`, `aggval*`, `blend*`, `crossdomain*`, `diagnose_aggregates`, `feature_compare*`, `flatten*`, `relmlp*`, `repair_*` | `logs/analysis/` |
| `logs/watchdog*`, `logs/.watchdog*`, `logs/sweep_launch.log` | `logs/infra/` |

Scripts, generators, docs and the report were updated to the new paths. Nothing was deleted and
no git-tracked file moved. The archived pages and `docs/CHANGES.patch` were left exactly as they
were, so the paths written inside them are the old ones.
