# GRDM (rdb-diffusion) — Reproduction Notes

> **Note, 5 Oct 2026.** Loan-default numbers in this document were computed with a date window that ended 185 days before each loan (raw stored dates compared across tables, which count from different days). The corrected numbers are in `reports/grdm-final-report.html` and `docs/downstream/RESULTS_downstream.md`; card-ownership numbers are unaffected.

Reproducing **Table 1** of *Joint Relational Database Generation via Graph-Conditional Diffusion
Models* (GRDM, NeurIPS 2025, arXiv:2505.16527). Paper PDF is in `docs/references/`.

Repo: `/scratch/s25017/ClavaDDPM/rdb-diffusion`
Companion repo (baseline, separately replicated): `/scratch/s25017/ClavaDDPM/ClavaDDPM`

Last updated: 2026-08-13.

---

## 1. Status at a glance

| Dataset | Seeds | Status | Max deviation from paper |
|---|---|---|---|
| **Berka** | 0,1,2 | ✅ **done — replicated** | 0.37 pts |
| **California** | 0,1,2 | ✅ **done — replicated** | 0.32 pts |
| **CCS** | 0,1,2 | ✅ **done — replicated** | 2.77 pts (1-HOP) |
| **berka_split** (leakage-free rerun, §8) | 0,1,2 | ✅ **done** | n/a — utility only |
| **S1-S3 fix experiments** (§9) | 0,1,2 | ✅ **done** | n/a — utility only |
| **S4-S5 fix experiments** (§9) | 0 | ✅ done; seeds 1,2 not run | n/a — utility only |
| **california_s5 guard** (§9) | 0 | 🔄 sampling/eval, ~done | n/a — utility only |
| Instacart 05 / MovieLens / RelBench-F1 | — | not attempted | — |

§8 adds a downstream XGBoost utility evaluation (real vs synthetic-only vs mixed) on top of the
paper's fidelity metrics. §9 diagnoses *why* Berka underperforms and measures five candidate fixes.

Only the **GRDM column** of Table 1 is being reproduced (not the SDV / SingleTable / Denorm
baselines, not the Table 2 ablations, not the appendix experiments). The ClavaDDPM baseline was
reproduced separately in the sibling repo.

---

## 2. Results

### Berka (n=3) — REPLICATED

| Metric | seed 0 | seed 1 | seed 2 | **Ours** | **Paper** | Δ |
|---|---|---|---|---|---|---|
| Cardinality | 99.72 | 99.59 | 99.64 | 99.65 ± 0.07 | 99.65 ± 0.05 | −0.00 |
| Column Shapes | 96.84 | 96.83 | 97.01 | 96.89 ± 0.10 | 96.84 ± 0.14 | +0.05 |
| Intra-Table Trends | 98.31 | 98.16 | 98.30 | 98.26 ± 0.08 | 98.23 ± 0.03 | +0.03 |
| 1-HOP | 91.42 | 91.14 | 91.72 | 91.43 ± 0.29 | 91.41 ± 0.11 | +0.02 |
| 2-HOP | 95.71 | 95.65 | 95.76 | 95.71 ± 0.06 | 95.57 ± 0.04 | +0.14 |
| 3-HOP | 91.18 | 91.98 | 93.01 | 92.06 ± 0.92 | 92.43 ± 0.69 | −0.37 |
| AVG 2-WAY | 95.22 | 95.13 | 95.40 | 95.25 ± 0.14 | (not reported) | — |

### California (n=3) — REPLICATED

| Metric | seed 0 | seed 1 | seed 2 | **Ours** | **Paper** | Δ |
|---|---|---|---|---|---|---|
| Cardinality | 99.96 | 99.97 | 99.96 | 99.96 ± 0.01 | 99.96 ± 0.01 | +0.00 |
| Column Shapes | 98.67 | 98.55 | 98.58 | 98.60 ± 0.06 | 98.75 ± 0.00 | −0.15 |
| Intra-Table Trends | 97.63 | 97.45 | 97.51 | 97.53 ± 0.09 | 97.35 ± 0.02 | +0.18 |
| 1-HOP | 96.45 | 96.47 | 96.52 | 96.48 ± 0.04 | 96.16 ± 0.05 | +0.32 |
| AVG 2-WAY | 97.04 | 96.96 | 97.01 | 97.00 ± 0.04 | (not reported) | — |

### CCS (n=3) — REPLICATED, with one caveat

| Metric | seed 0 | seed 1 | seed 2 | **Ours** | **Paper** | Δ |
|---|---|---|---|---|---|---|
| Cardinality | 99.80 | 99.75 | 99.83 | 99.79 ± 0.03 | 99.79 ± 0.03 | +0.00 |
| Column Shapes | 96.73 | 96.96 | 97.95 | 97.21 ± 0.53 | 97.30 ± 0.36 | −0.09 |
| Intra-Table Trends | 94.50 | 93.47 | 96.42 | 94.80 ± 1.22 | 94.82 ± 1.20 | −0.02 |
| 1-HOP | 86.60 | 79.61 | 81.63 | 82.61 ± 2.94 | 85.38 ± 3.03 | **−2.77** |
| AVG 2-WAY | 88.75 | 83.39 | 85.66 | 85.93 ± 2.20 | (not reported) | — |

Three of four metrics land within 0.1pt of the paper, closer than either Berka or California. 1-HOP
is the outlier at −2.77pt — still under one paper-sigma (3.03), so not a red flag by the σ-band
rule below, but it is the largest single-metric deviation across all three datasets and the only
one worth a second look if this number matters downstream. Not yet root-caused; no seed shows the
paper's 85.38 and seed 1 (79.61) pulls the mean down substantially, so this may be genuine seed
variance rather than a systematic gap — three seeds isn't enough to tell apart from a real,
smaller effect.

### How to judge these numbers

Compare on **absolute deviation, not sigma**. The paper's reported stds are extremely tight
(some ±0.00–0.04), so a 0.1pt difference can look like 3σ while being scientifically irrelevant.
All three replicated datasets land within ~0.4pt on Cardinality/Column Shapes/Intra-Table Trends,
with per-seed spread comparable to the paper's own; CCS's 1-HOP is the one metric across all three
datasets that misses by more than a point.

### Caveat worth carrying forward

Berka's near-perfect Cardinality is **partly structural, not earned**: `district` is declared a
dimension table, so it is held fixed at t=0 and conditioned on rather than generated. Its two
`district → *` relationships score exactly 100.00 and pull the average up; genuinely-generated
relationships score 98.98–99.93. California has **no dimension tables**, so its result is a
cleaner test of the method — and it still reproduces. The paper's headline claim (large 2-HOP /
3-HOP margins over autoregressive baselines) does **not** carry this asterisk.

Also note: Berka's bimodal seed-fragility seen in the ClavaDDPM replication does **not** appear
here, plausibly because the dimension-table treatment removes the root-sampling lottery that
caused it.

---

## 3. Environment

Use the shared conda env **`tabsyn`** (not a fresh `rdbdiff` env as the upstream README
suggests). Always invoke as `conda run -n tabsyn ...`, never bare `python`.

Installed with `--no-deps` to protect existing pins (notably `numpy==1.26.4`, needed by
`faiss-gpu-cu12` for the sibling ClavaDDPM repo):

```bash
pip install -e . --no-deps --ignore-requires-python   # repo pins >=3.11; env is 3.10, code is compatible
pip install "relbench[full]" wandb                     # relbench pulls torch_geometric
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

Kept the env's existing **torch 2.6.0+cu124** rather than downgrading to the repo's pinned
`torch==2.5.1`. Side effects of the install: scikit-learn 1.7.2→1.6.1, sympy/fsspec bumps. Verified
afterwards that numpy is still 1.26.4 and faiss still imports.

---

## 4. Data

All datasets reuse ClavaDDPM's already-downloaded preprocessed data — **no re-download needed**.
The formats are identical (`<table>.csv`, `<table>_domain.json`, `dataset_meta.json`).

```bash
data/berka/preprocessed -> /scratch/s25017/ClavaDDPM/ClavaDDPM/complex_data/berka/preprocessed
data/ccs/preprocessed   -> /scratch/s25017/ClavaDDPM/ClavaDDPM/complex_data/ccs/preprocessed
data/california/train                       # ships with the repo, no symlink needed
```

Still available in `ClavaDDPM/complex_data/` if you want to extend: `instacart_05`, `movie_lens`.
(RelBench-F1 ships with this repo at `data/f1`.)

---

## 5. Local changes to the upstream code

None of these alter the paper's methodology. The first group is required to run at all; the
second is the flag-gated fix machinery from §9, which is inert unless a flag is passed.

**`rdbdiff/random_graph_sampler.py`** — `np.concat` → `np.concatenate` (3 sites).
`np.concat` is a NumPy≥2.0-only alias; the env pins numpy 1.26.4. Crashes during structure
sampling in the multi-parent matching path, so it always triggers on Berka (`disp` has 2 parents).

**`rdbdiff/trainer.py`** + **`main.py`** — **mid-training checkpointing and resume.**
This is the important one. Upstream only saved at the very end of all 200k steps, and this node
kills processes at ~24h (see §7), so a run could lose 82% of its training. Added:
- `save_checkpoint(step)` — writes model + EMA + optimizer + step + loss history to
  `checkpoint.pt`, called at **every epoch boundary**. Written via a `.tmp` file then `os.replace`
  so a kill mid-write cannot leave a truncated checkpoint.
- `has_checkpoint()` / `load_checkpoint()` → returns the step to resume from.
- `run_training_loop(start_step=...)` — resumes mid-run. Checkpoints are only taken at epoch
  boundaries, which keeps the existing per-epoch node-count assert valid after a resume.
- `main.py` calls the resume path automatically when `checkpoint.pt` exists.

Checkpoints are ~184MB each and overwrite in place (one per seed, not one per epoch).

**`rdbdiff/utils/data.py`** — **`round_synthetic_data` no longer assumes a dense label encoding.**
Upstream asserted `real_unique == arange(len(real_unique))` for every categorical column, i.e. that
codes 0..K-1 all occur in the training rows. That holds for the shipped datasets but not for a
subsample: the `berka_split` holdout (§8) removes every client with birth-year code 73, leaving a
gap, and all three seeds crashed *after* training and sampling. Now clips to
`real_unique.max()` — which is what the function's own TODO asks for, and is identical to the old
behaviour whenever the codes are contiguous (then `max == len - 1`), so the Berka and California
numbers in §2 are unaffected.

**Flag-gated fix machinery (§9).** All default to upstream behaviour, so §2/§8 are unaffected:
`rdbdiff/utils/data.py` gains `decode_categorical_matched` (S1) and `decode_onehot_synthetic_data`
(S4) plus a `categorical_encoding` switch on `preprocess_database`/`denormalize_data`;
`rdbdiff/utils/graph.py` gains one-hot expansion in `make_pkey_fkey_graph` (S4);
`rdbdiff/dataloader.py` gains a `WeightedRandomSampler` for seed balancing (S3); `trainer.py`
relaxes the per-epoch node-count assert, which only holds when seeds are drawn without
replacement. `main.py` exposes `--categorical_decoding`, `--categorical_encoding` and
`--seed_balance_alpha`. `resume_runs.sh` gains `RDBDIFF_EXTRA_ARGS` and `RDBDIFF_THREADS`
(default 4, to keep each process under ~200% CPU when several run in parallel).

**`rdbdiff/utils/data.py` — `sort_columns` is now deterministic.** It built the id-column list
with `list(set(df) - ...)`, and set iteration order over strings varies *between processes*, so
two runs of the same seed wrote CSVs whose headers differed (`loan_id,account_id` vs
`account_id,loan_id`). Values were unaffected and consumers here match on column name, but the
output was not reproducible. Found by the S1 guard that reconstructs a stored sample and compares
it to what is on disk.

**New files (not upstream):** `resume_runs.sh`, `configs/{berka,california,ccs}_fastsample.json`.

The `*_fastsample.json` configs are byte-identical to the upstream configs **except**
`sample_batch_size: 20000 → 100000`. This is fidelity-safe: sampling uses `num_neighbors=-1`
(full neighborhoods) and `disjoint=False`, so batch grouping is purely a scheduling choice.
**Training hyperparameters are untouched** (K=1, T=2000, lr 6e-4, wd 1e-5, batch 4096,
200k steps / 100k for California) — changing those would stop this being a faithful reproduction.

---

## 6. Running / resuming

```bash
cd /scratch/s25017/ClavaDDPM/rdb-diffusion

scripts/resume_runs.sh                                # berka seeds 0 1 2
RDBDIFF_DATASET=ccs scripts/resume_runs.sh            # ccs seeds 0 1 2
RDBDIFF_DATASET=california scripts/resume_runs.sh 0 2 # specific seeds
RDBDIFF_GPUS="4 6" scripts/resume_runs.sh 1 2         # pin GPUs explicitly
```

The script is **idempotent and safe to re-run at any time** — it skips seeds that are already
running or already finished (`eval_result.pkl` exists), and resumes the rest from their last
checkpoint. This is the single command to run after a 24h kill.

It launches each job with `setsid nohup ... disown` (survives session/SSH death) and
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8` (see §7).

Logs: `logs/train/paper_repro_<dataset>_seed<N>.log`
Results: `rdbdiff_workspace/<dataset>/paper_repro/seed_<N>/synthetic_data_sampled_structure_size_multiplier_1.0/`

Check status:
```bash
for f in logs/train/paper_repro_*_seed*.log; do printf "%-40s %s\n" "$f" "$(tail -1 $f)"; done
grep -A 8 "Final results" logs/train/paper_repro_ccs_seed0.log
```

---

## 7. Cluster gotchas (all learned the hard way)

**~24h process lifetime.** Processes are SIGTERM'd at ~24h17m — three runs died within 15s of
each other at exactly that age. No visible scheduler/systemd limit and `dmesg` is locked down, so
the mechanism is unconfirmed, but the behaviour is reproducible. **This is why checkpointing
exists.** Just re-run `resume_runs.sh`.

**Detach properly or lose the job.** `setsid nohup ... < /dev/null > log 2>&1 &` + `disown`.
A job launched only via the agent's background-shell was killed when the session restarted, after
burning hours of GPU time.

**Pick GPUs by utilization, not free memory.** Each run needs only **~1.3GB VRAM**, so free
memory is a useless signal — a GPU at 0% util with 47GB allocated by an idle neighbour is an
excellent host. Moving seeds off contended GPUs gave a **~2.2x speedup** (12,700 vs 5,900
steps/h). `resume_runs.sh` averages 5 utilization samples to rank GPUs. Check with:
```bash
for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' '; echo; sleep 2; done
```

**Cap CPU threads.** PyTorch sizes its thread pool to all 224 cores by default, and the dataloader
runs in-process (`num_workers=0`). Three uncapped runs hit ~2000–2500% CPU each and pushed node
load to 280 on a 224-core shared box. `OMP_NUM_THREADS=8` (+ MKL/OPENBLAS/NUMEXPR) drops it to
~80% per process with no throughput loss. Already in `resume_runs.sh`.

**Timing reference** (H200, contended): Berka ~13–14h training + ~1–1.5h sampling/eval per seed.
California ~2h total (100k steps, 2 tables). CCS in between.

---

## 8. Downstream utility (XGBoost): real vs synthetic vs mixed

Everything in §2 is a **fidelity** metric. This section asks a different question: is the
synthetic data *useful*? Four relational tasks, three training regimes, all evaluated on the same
real held-out test set. Full tables in `docs/downstream/RESULTS_downstream.md`; harness in
`downstream/` (`run.py`, `tasks.py`, `evaluate.py`, `report.py`).

| dataset | task | metric | real | synthetic only | real + 100% syn | signal retained |
|---|---|---|---|---|---|---|
| california | individual INCTOT | R² | 0.5527 | 0.5272 | 0.5500 | **95.4%** |
| california | individual EMPSTAT | macro-F1 | 0.6850 | 0.6752 | 0.6813 | **98.6%** |
| **berka_split** | loan default at origination | AUROC | 0.8886 | 0.5917 | 0.8547 | **23.6%** |
| **berka_split** | card ownership (dependency probe) | AUROC | 0.8739 | 0.6021 | 0.8746 | **27.3%** |
| berka ⚠ | loan default at origination | AUROC | 0.8886 | 0.5798 | 0.8673 | 20.5% |
| berka ⚠ | card ownership (dependency probe) | AUROC | 0.8739 | 0.6094 | 0.8741 | 29.3% |

`berka_split` is the **leakage-free** protocol and the one to quote; ⚠ `berka` is the interim run
whose generator saw the test rows. The two agree to within the seed spread, so training on the
test rows was not what made this synthetic data look bad — the generator simply does not carry
these relationships. Real-only baselines are identical across the two by construction: both
evaluate the same fixed split, and only what the generator saw differs.

**California synthetic data is a near drop-in replacement for real; Berka synthetic data is not
usable on its own.** The fidelity scores do not predict that gap: Column Shapes 98.60 vs 96.89 and
1-HOP 96.48 vs 91.43 separate the two datasets by a few points, while downstream utility separates
them by an order of magnitude. California is 2 tables / 1 relationship; Berka is 8 tables / 9
relationships with a multi-parent `disp`.

**Mixing is safe but not an accuracy win.** Adding synthetic rows on top of the full real training
set moves every primary metric by at most 0.034 at every ratio (25 / 50 / 100%). The exception is
minority-class recall: on leakage-free Berka loan default it lifts macro-F1 up to +0.038 and
balanced accuracy +0.031 at a cost of −0.034 AUROC, behaving like oversampling.

**Two distinct Berka failure modes, only one visible in the marginals.** Loan default breaks at the
marginal — in the leakage-free run the default rate falls 0.112 → ~0.073 and class B collapses from
24 real rows to 1–10 synthetic across seeds. Card ownership keeps its marginal almost exactly
(0.168 vs 0.160–0.167) and still collapses 0.874 → 0.602, because the lost thing is the
*cross-table dependency*, not the rate. Matching per-column distributions is not sufficient for
utility.

**Report balanced accuracy, not accuracy.** The synthetic-only Berka model roughly matches the real
model on plain accuracy (0.8628 vs 0.8655 leakage-free; it *beats* it in the interim run) while
sitting at chance on balanced accuracy (0.500–0.515), by predicting the majority class nearly
everywhere.

### Two leaks found while building this (both fixed)

The first feature sets gave real-data AUROC of **1.0000** on loan default and **0.9805** on card
ownership. Those were ceilings measuring the schema, not the model. `*_label_encoders.pkl` in the
preprocessed directory names the culprits:

- `trans.k_symbol`=4 is `SANKC. UROK`, sanction interest charged only once an account is already in
  debt, and `trans_balance_min` alone separated the classes perfectly. **Fixed properly:**
  transaction aggregates are now restricted to `trans_date < loan_date`, making this a genuine
  prediction-at-origination task. That leaves a median of 30 prior transactions per loan and drops
  `SANKC. UROK` from 77 accounts to 8. Real AUROC → 0.8722.
- `trans.operation`=5 is `VYBER KARTOU`, a credit-card withdrawal, which cannot exist unless the
  disposition holds a card. Cards are issued a median of 348 days after account opening with only
  51% inside the first year, so **no time window separates cause from effect**. The deterministic
  giveaway is dropped and the task is scored as a *cross-table dependency probe* rather than a
  forecast. Real AUROC → 0.8737.

### Berka leakage-free rerun

Berka ships no train/test split, so `downstream/make_berka_split.py` cuts one at the `account` hub
(25% test, stratified on loan outcome) into `data/berka/split_{train,test}`. The cut is clean
because `client`↔`disp` is strictly 1:1, `loan` is 1-per-account and `card` is 1-per-disp, so no
row straddles the boundary; `district` is a dimension table and stays whole on both sides.

`configs/berka_split_fastsample.json` retrains GRDM on `split_train` only, with training
hyperparameters untouched. **Done — 3 seeds, results in the table above.** All three seeds crashed
after training and sampling on the dense-label-encoding assert described in §5; that fix is a
no-op on the shipped datasets. Launch or resume with:

```bash
RDBDIFF_DATASET=berka_split scripts/resume_runs.sh 0 1 2
conda run --no-capture-output -n tabsyn python -m downstream.run --dataset berka_split
```

The interim and clean protocols use the *same* fixed split, so the only difference between them is
what the generator was allowed to see.

### Running the harness

```bash
conda run --no-capture-output -n tabsyn python -m downstream.run --dataset california
conda run --no-capture-output -n tabsyn python -m downstream.run --dataset berka
conda run --no-capture-output -n tabsyn python -m downstream.report
```

Berka takes ~2 min, California ~70 min (CPU only, no GPU). Hyperparameters are fixed and identical
across regimes — no per-regime tuning, and no early stopping, since an early-stopping set would
have to be real and would leak real rows into the synthetic-only regime. Runs are deterministic
(verified: identical metrics on re-run).

---

## 9. Why Berka fails, and five fixes

§8 established *that* Berka synthetic data is unusable (20-27% of the real signal) while
California's is nearly a drop-in replacement. This section establishes *why*. Full evidence in
`docs/downstream/DIAGNOSIS.md` (`downstream/diagnose.py`), fixes and their measured effects in
`docs/downstream/FIXES.md`.

### Four root causes, each with a test that could have falsified it

**H1 - Conditioning depth.** `gnn_params["num_layers"] = n_hops = 1`, so generating a `loan`
conditions on its parent `account`'s two own columns and nothing else; the account's ~235
transactions and the district are two hops away. *Test (A2):* restrict a **real-data** model to
that same window and see whether it reproduces the synthetic ceiling.

| feature tier | n | AUROC |
|---|---|---|
| T0 loan's own row | 6 | 0.6015 |
| T1 + 1-hop parent — **GRDM's actual window** | 9 | 0.5529 |
| T2 + 2-hop (district, child aggregates) | 97 | **0.8886** |

Synthetic-only, with the *full* 97 features, scores **0.5917** — squarely in the T0/T1 band and
nowhere near T2. The synthetic data carries about as much loan-default signal as a real model
allowed to see only the loan row and its parent's two columns. The 2-hop context is worth
~0.29 AUROC and GRDM cannot see any of it. **This is the main cause.**

**H2 - Categoricals are ordinal continuous variables.** Class codes enter `x` as raw integers,
are quantile-transformed *alongside* numerical columns (so a 4.7%-mass class gets a
proportionally narrow target band), diffused, and rounded back. *Test (A1):* averaged over every
discrete column the effect is negligible (median relative change 0.000), so this is not damaging
on its own. It is confined to small tables — every table with 4,000+ rows changes by under 1%,
while `loan.status` B loses 76% and `account.frequency` code 1 loses 74%.

**H3 - Gradient starvation.** Seeds are drawn uniformly over rows and the loss is computed on
seed nodes only, so per 4096-seed batch `trans` gets 4007.8 and `loan` gets **2.6**. H2 and H3
are not independent: the ordinal metric supplies the failure mode, gradient starvation supplies
the conditions.

**H4 - Structure is generated independently of attributes.** `RandomGraphSampler` draws each
child's in-degree from that relationship's marginal with `rng.choice(...)`, and the entire
structure is sampled *before* any attribute exists. So a **structural** label — "does this
disposition hold a card?" — is decided by a degree-matching random draw that cannot depend on
any attribute. *Test (A5):* train and evaluate entirely *within* each world, to separate "the
relationship is absent" from "it is merely different".

| task | real → real | synthetic → synthetic |
|---|---|---|
| loan default | 0.8886 | 0.6301 |
| card ownership | 0.8739 | **0.5702** |

A model trained on synthetic data cannot recover the relationship even when tested on synthetic
data, so it is genuinely absent. This one is **architectural**: information flows
structure → attributes only, so no amount of conditioning depth or categorical fixing can repair
a structural label.

### The four fixes

All are flag-gated and off by default; the reproduction in §2 and §8 is unaffected.

| fix | targets | change | retrain |
|---|---|---|---|
| **S1** matched decoding | H2 | `--categorical_decoding match` — place decode thresholds so the synthetic marginal matches the real one | no |
| **S2** deeper conditioning | H1 | `--n_hops 2 --num_neighbors -1 16` | yes |
| **S3** balanced seeding | H3 | `--seed_balance_alpha 0.5` — weight seeds by `count(type)^-alpha` | yes |
| **S4** one-hot categoricals | H2 | `--categorical_encoding onehot` — indicator columns, no quantile transform on them, argmax decode | yes |
| **S5** S2 + S3 combined | H1 + H3 | both flags together | yes |

**Prediction recorded before the retrains finished**, from the diagnosis above: S2 should improve
loan default (it opens the 0.29-AUROC gap between T1 and T2) and should **not** improve card
ownership (H4 is architectural). S1/S4 should repair rare-class fidelity without moving
downstream utility. S3 should help mainly by amplifying whatever S2/S4 achieve on small tables.

### S1 result: marginal restored, utility unchanged

S1 reproduces the real class marginal exactly — `loan.status` B returns to 24/24/23 rows across
seeds (real 24) and the default rate to 0.110-0.113 (real 0.112). Downstream utility does not
move: loan default 0.5917 → **0.5963**, card ownership 0.6021 → **0.6009**, both inside seed
noise. This is the cleanest available evidence that the marginal collapse was a *symptom*, and
that H1/H4 are what actually bind.

### Final results, all seeds — and a falsified prediction

| fix | seeds | loan default | card ownership |
|---|---|---|---|
| baseline | 3 | 0.5917 ± 0.044 (23.6%) | 0.6021 ± 0.017 (27.3%) |
| S1 matched decoding | 3 | 0.5963 ± 0.024 (24.8%) | 0.6009 ± 0.018 (27.0%) |
| **S2 `n_hops=2`** | 3 | 0.5264 ± 0.122 (6.8%) | **0.6687 ± 0.017 (45.1%)** |
| **S3 balanced seeding** | 3 | **0.6613 ± 0.035 (41.5%)** | 0.6164 ± 0.050 (31.1%) |
| S4 one-hot | 1 | 0.5526 (13.5%) | 0.5358 (9.6%) |
| **S5 = S2 + S3** | 1 | 0.6322 (34.0%) | **0.6813 (48.5%)** |

**The prediction recorded above was wrong, and informatively so.** It said S2 should improve loan
default and should *not* improve card ownership, because card ownership is a structural label and
H4 makes structure independent of attributes. The measurement is the reverse: S2 nearly doubles
retained signal on **card ownership** (27.3% → 45.1%) and, with all three seeds in, *actively hurts*
loan default (23.6% → 6.8%, not merely flat as the single-seed pilot suggested).

The card-ownership result exposes an asymmetry in H4. Structure is sampled first and independently
of attributes, but attributes are then sampled *conditioned on structure* — so information flows
structure → attributes, and how far it flows is exactly `n_hops`. At `n_hops=1` an `account`'s
attributes cannot see whether its dispositions hold cards, because `card` is two hops from
`account` (`account → disp → card`). At `n_hops=2` they can, so account balance patterns become
correlated with card ownership and a model trained on synthetic data recovers part of the real
relationship. H4 makes a structural label unfixable only when the structure in question lies
outside the conditioning window; widening the window is precisely what repairs it.

Loan default getting *worse* under S2, despite A2 showing the most headroom there, is not fully
explained, but S3 supplies the working theory: `n_hops=2` widens every table's effective batch
(the fan-out that made `--num_neighbors -1 -1` cost ~941,000 nodes/batch), which further dilutes
`loan`'s already-thin 2.6-seeds-per-batch share in relative terms even though the raw count is
unchanged. **S3 alone (balanced seeding, no depth change) is the single best fix for loan default**
(23.6% → 41.5%) and the only fix that *improves* fidelity over baseline rather than trading it away
(Column Shapes 96.86 → 96.82–96.99, 1-HOP 90.38 → 91.78–91.91, all three seeds). **S5 combines
both**: it keeps S2's card-ownership gain (48.5%, the best of any fix) while recovering most of
S2's loan-default loss (34.0% vs S2's 6.8%), which is exactly what the dilution theory predicts —
rebalancing the seeds offsets the extra cost that depth imposes on the small tables it doesn't
target.

**S2 raises card-ownership utility while *lowering* fidelity** (Column Shapes 96.86 → ~95.0, 1-HOP
90.38 → ~89.3). On this intervention the two axes move in opposite directions — a sharper version
of §8's finding that fidelity does not predict utility. S3 is the counter-example: it is the one
fix where utility and fidelity move together, both up.

**S4 is harmful everywhere and was dropped after one seed.** Berka: 1-HOP fell to 80.82, both
tasks got worse. The decisive result was the California regression guard: synthetic-only R² went
from 0.5244 to **−4.9684** (worse than predicting the mean) and macro-F1 from 0.6749 to 0.3533 — a
fix that breaks the database that already worked was not worth chasing further on the one that
didn't, so S4's remaining Berka seeds were never run. Gaussian diffusion over a one-hot simplex
with argmax decoding loses far more than the ordinal metric it removes. Full discrete/multinomial
diffusion — the parent class's unused `num_classes` path — remains the untried principled option,
and would not share this failure mode since it never treats the simplex as Gaussian.

One further correction: a generalised form of H2, that rare *interior* codes are systematically
absorbed by dense neighbours, does **not** survive testing. Stratified by rarity, interior classes
lose no more mass than edge classes, and band width in quantile-normal space is uncorrelated with
retention (Spearman ρ = −0.03, n=237). The effect is real per-column (`loan.status` B retains
0.24×, `trans.k_symbol` 4 retains 0.14×) but is not a general law.

### Bottom line

**Ship S3 if only single-parent-aggregate tasks matter; ship S5 as the best all-around fix.** S5
is within 7 points of S3's best loan-default number while beating every fix on card ownership,
because it is the only variant that gives the multi-hop dependency a wide-enough window *and*
enough gradient to learn it. For anyone generating a deep relational schema with GRDM: set
`n_hops` to at least the distance between a row and the tables that carry its predictive signal,
cap `num_neighbors` beyond the first hop so the batch stays tractable, and pair any increase in
depth with balanced seed sampling — depth alone can cost more on the tables it doesn't target than
it gains on the one it does. The published `n_hops=1`, uniform-sampling default is well matched to
a two-table schema like California and badly matched to Berka on both axes at once.

Full per-fix design notes and discussion: `docs/downstream/FIXES.md`. Full tables with all seeds and
both metrics: `docs/downstream/RESULTS_downstream.md`.

---

## 10. Suggested next steps

1. **S5 seeds 1-2 and its California guard** — S5 (§9) is the strongest all-around fix on one
   seed; the natural next step is error bars and confirming it doesn't regress California, the
   same protocol already applied to S2/S3/S4. The California guard is mid-run (§1).
2. **Root-cause CCS's 1-HOP deviation** (§2) — 3 seeds isn't enough to tell genuine seed variance
   from a systematic gap; a few more seeds would settle it.
3. **Instacart 05** — highest scientific value remaining. The paper claims its single largest
   margin there (2-HOP: 92.95 vs best baseline 16.22, a 5.7x gain). Data is already in
   `ClavaDDPM/complex_data/instacart_05`; needs a `configs/instacart_05_fastsample.json` and a
   symlink. Note the paper used `--n_hops 2 --num_neighbors -1 -1 --train_batch_size 1024` for it.
4. **RelBench-F1** — data ships with the repo at `data/f1`; 9 tables, the most complex schema.
5. Optionally the Table 2 ablation (`--fixed_structure`) — cheap, since it reuses trained
   checkpoints and only re-runs sampling.
6. **Consolidated 3-dataset table** combining Berka/California/CCS from §2 into one summary,
   now that all three have finished.
