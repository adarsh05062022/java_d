# GRDM reproduction — full report source: results, code changes, experiments

> **Note, 5 Oct 2026.** Loan-default numbers in this document were computed with a date window that ended 185 days before each loan (raw stored dates compared across tables, which count from different days). The corrected numbers are in `reports/grdm-final-report.html` and `docs/downstream/RESULTS_downstream.md`; card-ownership numbers are unaffected.

Self-contained source material for a write-up. Every number is re-derived from the result files in
`downstream/results/` and the run logs, not recalled from notes.

Upstream baseline: commit `bc0fb67` of the GRDM repo. Paper: *Joint Relational Database Generation
via Graph-Conditional Diffusion Models*, NeurIPS 2025, arXiv:2505.16527.

| companion file | contents |
|---|---|
| `CHANGES.patch` | `git diff HEAD` — the exact source diff, 673 lines, 6 files |
| `REPRODUCTION.md` | operational notes: environment, data, how to run, cluster gotchas |
| `docs/downstream/DIAGNOSIS.md` | root-cause analysis in full |
| `docs/downstream/FIXES.md` | per-fix design notes |
| `docs/downstream/RESULTS_downstream.md` | auto-generated tables, every seed and metric |

**Contents.** §1 what was done · §2 fidelity reproduction · §3 downstream utility · §4 leakage-free
rerun · §5 root-cause diagnosis · §6 five fixes · §7 code changes · §8 harness · §9 how to
reproduce · §10 caveats and limits

---

## 1. What was done

**Phase 1 — reproduce Table 1 (fidelity).** Berka, California, CCS; 3 seeds each. All replicate.

**Phase 2 — ask whether the synthetic data is *useful*, not merely faithful.** Four relational
downstream tasks scored with XGBoost under three training regimes (real / synthetic-only / mixed),
all on the same real held-out test set.

**Phase 3 — diagnose why Berka fails and try to fix it.** Four root-cause hypotheses, each with a
test that could have falsified it; then five interventions measured separately.

Headline: **California synthetic data retains 95–99% of the real signal; Berka retains 24–27%. The
fidelity metrics do not predict that gap.** The cause is the generator's conditioning window, and
the two fixes that address it recover roughly half of what was lost.

---

## 2. Phase 1 — fidelity reproduction

Only the **GRDM column** of Table 1 is reproduced (not SDV / SingleTable / Denorm baselines, not the
Table 2 ablations). Judge these on **absolute deviation, not sigma**: the paper's stds are extremely
tight (some ±0.00–0.04), so a 0.1pt difference can look like 3σ while being scientifically
irrelevant.

### Berka (n=3) — replicated

| Metric | seed 0 | seed 1 | seed 2 | **Ours** | **Paper** | Δ |
|---|---|---|---|---|---|---|
| Cardinality | 99.72 | 99.59 | 99.64 | 99.65 ± 0.07 | 99.65 ± 0.05 | −0.00 |
| Column Shapes | 96.84 | 96.83 | 97.01 | 96.89 ± 0.10 | 96.84 ± 0.14 | +0.05 |
| Intra-Table Trends | 98.31 | 98.16 | 98.30 | 98.26 ± 0.08 | 98.23 ± 0.03 | +0.03 |
| 1-HOP | 91.42 | 91.14 | 91.72 | 91.43 ± 0.29 | 91.41 ± 0.11 | +0.02 |
| 2-HOP | 95.71 | 95.65 | 95.76 | 95.71 ± 0.06 | 95.57 ± 0.04 | +0.14 |
| 3-HOP | 91.18 | 91.98 | 93.01 | 92.06 ± 0.92 | 92.43 ± 0.69 | −0.37 |
| AVG 2-WAY | 95.22 | 95.13 | 95.40 | 95.25 ± 0.14 | (not reported) | — |

### California (n=3) — replicated

| Metric | seed 0 | seed 1 | seed 2 | **Ours** | **Paper** | Δ |
|---|---|---|---|---|---|---|
| Cardinality | 99.96 | 99.97 | 99.96 | 99.96 ± 0.01 | 99.96 ± 0.01 | +0.00 |
| Column Shapes | 98.67 | 98.55 | 98.58 | 98.60 ± 0.06 | 98.75 ± 0.00 | −0.15 |
| Intra-Table Trends | 97.63 | 97.45 | 97.51 | 97.53 ± 0.09 | 97.35 ± 0.02 | +0.18 |
| 1-HOP | 96.45 | 96.47 | 96.52 | 96.48 ± 0.04 | 96.16 ± 0.05 | +0.32 |
| AVG 2-WAY | 97.04 | 96.96 | 97.01 | 97.00 ± 0.04 | (not reported) | — |

### CCS (n=3) — replicated, with one caveat

| Metric | seed 0 | seed 1 | seed 2 | **Ours** | **Paper** | Δ |
|---|---|---|---|---|---|---|
| Cardinality | 99.80 | 99.75 | 99.83 | 99.79 ± 0.03 | 99.79 ± 0.03 | +0.00 |
| Column Shapes | 96.73 | 96.96 | 97.95 | 97.21 ± 0.53 | 97.30 ± 0.36 | −0.09 |
| Intra-Table Trends | 94.50 | 93.47 | 96.42 | 94.80 ± 1.22 | 94.82 ± 1.20 | −0.02 |
| 1-HOP | 86.60 | 79.61 | 81.63 | 82.61 ± 2.94 | 85.38 ± 3.03 | **−2.77** |
| AVG 2-WAY | 88.75 | 83.39 | 85.66 | 85.93 ± 2.20 | (not reported) | — |

CCS matches the paper more closely than Berka or California on three of four metrics. **1-HOP is the
outlier at −2.77pt** — still inside one paper-sigma (3.03), but the largest single-metric deviation
across all three datasets. Not root-caused: no seed reaches the paper's 85.38, and seed 1 (79.61)
drags the mean down, so this may be seed variance rather than a systematic gap. Three seeds cannot
distinguish those.

### A caveat on Berka's Cardinality

Berka's near-perfect Cardinality is **partly structural, not earned**: `district` is declared a
dimension table, held fixed at t=0 and conditioned on rather than generated. Its two `district → *`
relationships score exactly 100.00 and pull the average up; genuinely-generated relationships score
98.98–99.93. California has **no dimension tables**, so its result is a cleaner test — and it still
reproduces. The paper's headline claim (large 2-HOP / 3-HOP margins over autoregressive baselines)
does not carry this asterisk.

Also worth noting: Berka's bimodal seed-fragility seen in the ClavaDDPM replication does **not**
appear here, plausibly because the dimension-table treatment removes the root-sampling lottery.

---

## 3. Phase 2 — downstream utility

Everything in §2 is a *fidelity* metric. This asks a different question: is the synthetic data
*useful*?

**Protocol.** Three regimes — real only, synthetic only, mixed (real + 25/50/100% synthetic).
Synthetic-only is **subsampled to the real training row count**, so real vs synthetic compares data
*quality*, not quantity; the mixed rows then vary quantity on top of the full real set. All regimes
use **identical, untuned** XGBoost hyperparameters — per-regime tuning would make the regimes
incomparable, and an early-stopping set would have to be drawn from real data, leaking real rows
into the synthetic-only regime. Every regime is scored on the **same real held-out test set**. Runs
are deterministic (verified: identical metrics on re-run).

### The four tasks

| task | grain | label | relational span |
|---|---|---|---|
| `california_inctot` | individual | INCTOT (income quantile bin, 0–99) | 1-hop join + sibling aggregates |
| `california_empstat` | individual | EMPSTAT (4 classes) | same feature space |
| `berka_loan_default` | loan | `status ∈ {B,D}`, at origination | 2 hops (account + child aggregates) |
| `berka_card_ownership` | disposition | holds a card | 3 hops — cross-table dependency probe |

Feature sets deliberately span joins and child aggregates; single-table features would exercise
nothing the relational generator does beyond a per-table tabular model.

### Summary — signal retained

"Retained" = share of the real baseline's signal above the metric's floor (chance for AUROC, zero
for R² and macro-F1).

| dataset | task | metric | real | synthetic only | real + 100% syn | retained |
|---|---|---|---|---|---|---|
| california | individual INCTOT | R² | 0.5527 | 0.5272 | 0.5500 | **95.4%** |
| california | individual EMPSTAT | macro-F1 | 0.6850 | 0.6752 | 0.6813 | **98.6%** |
| berka_split | loan default | AUROC | 0.8886 | 0.5917 | 0.8547 | **23.6%** |
| berka_split | card ownership | AUROC | 0.8739 | 0.6021 | 0.8746 | **27.3%** |

### Full regime tables

**California — INCTOT (regression)**, mean ± sd over 3 generator × 3 XGBoost seeds:

| regime | n train | R² | RMSE | MAE |
|---|---|---|---|---|
| real only | 1,521,638 | 0.5527 ± 0.0003 | 11.0238 ± 0.0039 | 5.5943 ± 0.0040 |
| synthetic only | 1,521,638 | 0.5272 ± 0.0020 | 11.3331 ± 0.0240 | 5.7722 ± 0.0137 |
| real + 25% syn | 1,902,048 | 0.5522 ± 0.0002 | 11.0301 ± 0.0027 | 5.5967 ± 0.0030 |
| real + 50% syn | 2,282,457 | 0.5514 ± 0.0004 | 11.0398 ± 0.0048 | 5.6007 ± 0.0038 |
| real + 100% syn | 3,043,276 | 0.5500 ± 0.0004 | 11.0573 ± 0.0050 | 5.6084 ± 0.0039 |

**California — EMPSTAT (4-class)**:

| regime | n train | accuracy | macro-F1 | weighted-F1 | AUROC (ovr) |
|---|---|---|---|---|---|
| real only | 1,521,638 | 0.8761 ± 0.0001 | 0.6850 ± 0.0001 | 0.8617 ± 0.0001 | 0.9370 ± 0.0001 |
| synthetic only | 1,521,638 | 0.8732 ± 0.0002 | 0.6752 ± 0.0009 | 0.8577 ± 0.0002 | 0.9307 ± 0.0003 |
| real + 25% syn | 1,902,048 | 0.8760 ± 0.0002 | 0.6832 ± 0.0004 | 0.8613 ± 0.0002 | 0.9371 ± 0.0001 |
| real + 50% syn | 2,282,457 | 0.8759 ± 0.0001 | 0.6820 ± 0.0006 | 0.8611 ± 0.0002 | 0.9371 ± 0.0001 |
| real + 100% syn | 3,043,276 | 0.8759 ± 0.0001 | 0.6813 ± 0.0005 | 0.8609 ± 0.0001 | 0.9368 ± 0.0001 |

**Berka (leakage-free) — loan default**, mean ± sd over 3 generator × 5 XGBoost seeds:

| regime | n train | AUROC | avg precision | macro-F1 | balanced acc | accuracy |
|---|---|---|---|---|---|---|
| real only | 511 | 0.8886 ± 0.0116 | 0.3850 ± 0.0195 | 0.5428 ± 0.0254 | 0.5375 ± 0.0184 | 0.8655 ± 0.0064 |
| synthetic only | 505 | 0.5917 ± 0.0445 | 0.1565 ± 0.0169 | 0.4980 ± 0.0467 | 0.5145 ± 0.0344 | 0.8628 ± 0.0232 |
| real + 25% syn | 639 | 0.8771 ± 0.0152 | 0.3726 ± 0.0272 | 0.5590 ± 0.0398 | 0.5502 ± 0.0307 | 0.8663 ± 0.0104 |
| real + 50% syn | 767 | 0.8733 ± 0.0138 | 0.3852 ± 0.0393 | 0.5805 ± 0.0602 | 0.5682 ± 0.0495 | 0.8737 ± 0.0148 |
| real + 100% syn | 1,016 | 0.8547 ± 0.0211 | 0.3651 ± 0.0367 | 0.5722 ± 0.0446 | 0.5618 ± 0.0360 | 0.8678 ± 0.0100 |

**Berka (leakage-free) — card ownership**:

| regime | n train | AUROC | avg precision | macro-F1 | balanced acc | accuracy |
|---|---|---|---|---|---|---|
| real only | 4,007 | 0.8739 ± 0.0013 | 0.4801 ± 0.0059 | 0.6924 ± 0.0093 | 0.6889 ± 0.0089 | 0.8364 ± 0.0056 |
| synthetic only | 4,007 | 0.6021 ± 0.0173 | 0.2003 ± 0.0108 | 0.4593 ± 0.0058 | 0.5003 ± 0.0021 | 0.8362 ± 0.0026 |
| real + 25% syn | 5,009 | 0.8763 ± 0.0047 | 0.4879 ± 0.0161 | 0.6878 ± 0.0109 | 0.6830 ± 0.0108 | 0.8353 ± 0.0059 |
| real + 50% syn | 6,011 | 0.8742 ± 0.0031 | 0.4830 ± 0.0146 | 0.6867 ± 0.0120 | 0.6816 ± 0.0135 | 0.8351 ± 0.0049 |
| real + 100% syn | 8,014 | 0.8746 ± 0.0046 | 0.4831 ± 0.0171 | 0.6863 ± 0.0090 | 0.6818 ± 0.0102 | 0.8343 ± 0.0054 |

### Four findings from Phase 2

**Fidelity does not predict utility.** Column Shapes 98.60 vs 96.89 and 1-HOP 96.48 vs 91.43
separate the two databases by a few points; downstream utility separates them by an order of
magnitude. California is 2 tables / 1 relationship; Berka is 8 tables / 9 relationships with a
multi-parent `disp`.

**Mixing is safe but not an accuracy win.** Adding synthetic rows on top of the full real training
set moves every primary metric by at most 0.034 at any ratio. The exception is minority-class
recall: on Berka loan default it lifts macro-F1 up to +0.038 and balanced accuracy +0.031 at a cost
of −0.034 AUROC, behaving like oversampling.

**Two distinct Berka failure modes, only one visible in the marginals.** Loan default breaks at the
marginal — default rate falls 0.112 → ~0.073, class B collapses from 24 real rows to 1–10 synthetic.
Card ownership keeps its marginal almost exactly (0.168 real vs 0.160–0.167 synthetic) and *still*
collapses 0.874 → 0.602, because what is lost is the **cross-table dependency**, not the rate.
Matching per-column distributions is not sufficient for utility.

**Report balanced accuracy, not accuracy.** The synthetic-only Berka model roughly matches the real
model on plain accuracy (0.8628 vs 0.8655) while sitting at **chance on balanced accuracy**
(0.500–0.515) — it predicts the majority class nearly everywhere. Any evaluation reporting accuracy
alone would have scored this synthetic data as excellent.

### Label prevalence — the mechanism behind the marginal collapse

Leakage-free generator, trained on `split_train` only:

| source | loan rows | A (repaid) | B (unpaid) | C (running ok) | D (in debt) | default rate | card rate |
|---|---|---|---|---|---|---|---|
| real | 511 | 150 | **24** | 304 | 33 | 0.112 | 0.168 |
| synthetic seed 0 | 505 | 164 | **1** | 304 | 36 | 0.073 | 0.167 |
| synthetic seed 1 | 508 | 144 | **6** | 327 | 31 | 0.073 | 0.160 |
| synthetic seed 2 | 499 | 158 | **10** | 301 | 30 | 0.080 | 0.164 |

Note the asymmetry: `loan.status` class B is nearly erased, while the card rate is reproduced almost
exactly — yet the card *task* still collapses. That contrast is the core evidence that marginals and
dependencies fail independently.

### Two label leaks found and fixed

The first feature sets produced real-data AUROC of **1.0000** (loan) and **0.9805** (card) — ceilings
measuring the schema, not the model. The Berka label encoders (`*_label_encoders.pkl`) name the
causes exactly:

- **`trans.k_symbol` = 4 = `SANKC. UROK`** — sanction interest, charged *only once an account is
  already in debt*, which is what a defaulted loan is; `trans_balance_min` alone separated the
  classes perfectly. **Fixed properly**: transaction aggregates restricted to
  `trans_date < loan_date`, making it a genuine prediction-at-origination task. Leaves a median of
  30 prior transactions per loan and drops `SANKC. UROK` from 77 accounts to 8. Real AUROC → 0.872.
- **`trans.operation` = 5 = `VYBER KARTOU`** — a credit-card withdrawal, which cannot exist unless
  the disposition holds a card. Cards are issued a median of **348 days** after account opening,
  only 51% inside the first year, so **no time window separates cause from effect**. The
  deterministic giveaway is dropped and the task is scored honestly as a *cross-table dependency
  probe* rather than a forecast. Real AUROC → 0.874.

A leakage guard runs in `run.py`: aggregates may not derive from the target column, and the `card`
table may contribute the label only.

---

## 4. The leakage-free Berka rerun

Berka ships no train/test split, and the Phase-1 generator trained on the whole database — including
the rows later used as a test set. `downstream/make_berka_split.py` cuts one at the `account` hub
(25% test, stratified on loan outcome) into `data/berka/split_{train,test}`. The cut is clean
because `client`↔`disp` is 1:1, `loan` is 1-per-account and `card` is 1-per-disp, so no row straddles
the boundary; `district` is a dimension table and stays whole on both sides.

| task | protocol | real | synthetic | mixed | retained |
|---|---|---|---|---|---|
| loan default | interim (generator saw test rows) | 0.8886 | 0.5798 | 0.8673 | 20.5% |
| loan default | **leakage-free** | 0.8886 | 0.5917 | 0.8547 | 23.6% |
| card ownership | interim (generator saw test rows) | 0.8739 | 0.6094 | 0.8741 | 29.3% |
| card ownership | **leakage-free** | 0.8739 | 0.6021 | 0.8746 | 27.3% |

**The rerun confirms the interim result rather than overturning it** — both deltas sit inside the
seed spread. Training on the test rows was not what made this synthetic data look bad. Real-only
baselines are identical across protocols by construction: both evaluate the same fixed split, and
only what the generator saw differs.

---

## 5. Phase 3 — root-cause diagnosis

Four hypotheses, each with a test that could have falsified it.

### H1 — Conditioning depth (the main cause)

The GNN has `n_hops = 1` layers, so a row is generated conditioned on its own noisy features plus
its direct neighbours' *raw* columns and nothing further. Generating a `loan` sees its account's two
columns; the account's ~235 transactions are two hops away. A `card` sees only `disp_type`.

**Test (A2):** restrict a **real-data** XGBoost to exactly that window and see whether it reproduces
the synthetic ceiling.

| task | conditioning window | features | real AUROC |
|---|---|---|---|
| loan default | own row only | 6 | 0.6015 ± 0.0131 |
| loan default | + 1-hop parents — **GRDM's actual window** | 9 | 0.5561 ± 0.0087 |
| loan default | + 2-hop (full) | 97 | **0.8886 ± 0.0116** |
| card ownership | own row only | 1 | 0.6038 |
| card ownership | + 1-hop parents — **GRDM's actual window** | 8 | 0.6708 ± 0.0016 |
| card ownership | + 2-hop (full) | 90 | **0.8739 ± 0.0013** |

Synthetic-only, with the *full* feature set, scores **0.5917** (loan) and **0.6021** (card) — squarely
in the restricted bands, nowhere near the full-window numbers. **The synthetic ceiling *is* the
conditioning window.** The generator is not failing to model what it sees; it never sees the rest.
California's `individual ← household` is exactly one hop, which is why the same configuration works
there.

A subtlety worth reporting: the effective window sits **one hop behind the nominal depth**. Baseline
(`n_hops=1`) scores 0.60 on card against an *own-row* ceiling of 0.604, not the 1-hop ceiling; S2
(`n_hops=2`) reaches 0.67–0.69 against the *1-hop* ceiling of 0.671. A neighbour's features are
themselves still noisy while the row is generated.

### H2 — Categoricals are ordinal continuous variables

Class codes enter `x` as raw integers, are quantile-transformed *alongside* numerical columns (so a
4.7%-mass class gets a proportionally narrow target band), Gaussian-diffused, and rounded back.

**Real but not load-bearing** — see S1 in §6. And the *generalised* form of this hypothesis is
**false**: stratified by rarity, interior classes lose no more mass than edge classes, and band width
in quantile-normal space is uncorrelated with retention (Spearman ρ = −0.03, n = 237). The effect is
real per-column (`loan.status` B retains 0.24×, `trans.k_symbol` 4 retains 0.14×,
`account.frequency` 1 retains 0.26×) but is not a general law. An early draft asserted the general
version; the data refuted it.

### H3 — Gradient starvation

Seeds are drawn uniformly over rows and the diffusion loss is computed on **seed nodes only**, so
per-batch gradient is proportional to table size:

| table | rows | share of seed pool | seeds per 4096-row batch | steps per full pass |
|---|---|---|---|---|
| trans | 792,600 | 97.85% | 4007.8 | 1.0 |
| order | 4,874 | 0.60% | 24.6 | 166.2 |
| client | 4,007 | 0.49% | 20.3 | 202.2 |
| disp | 4,007 | 0.49% | 20.3 | 202.2 |
| account | 3,375 | 0.42% | 17.1 | 240.0 |
| card | 672 | 0.083% | 3.4 | 1205.4 |
| **loan** | **511** | **0.063%** | **2.6** | **1585.2** |

`trans` takes 4007.8 of every 4096 seeds; `loan` takes 2.6.

### H4 — Structure is generated independently of attributes

`RandomGraphSampler` draws each child's in-degree from that relationship's marginal, and the entire
structure is sampled *before* any attribute exists. So a **structural** label — "does this
disposition hold a card?" — is decided by a degree-matching random draw that cannot depend on any
attribute.

**Test (A5):** train and evaluate entirely *within* each world, separating "the relationship is
absent" from "it is merely different".

| task | real → real | synthetic → synthetic |
|---|---|---|
| loan default | 0.8886 | 0.6301 |
| card ownership | 0.8739 | **0.5702** |

A synthetic-trained model cannot recover the relationship even when tested on synthetic data, so it
is genuinely absent. §6 shows this is only partly architectural.

---

## 6. Phase 3 — the five fixes

All flag-gated and off by default. Each changes exactly one thing and is scored on the same fixed
split and test set, so a delta is attributable to that change alone. Each writes to its own
workspace and results file.

| fix | targets | change | retrain? |
|---|---|---|---|
| **S1** matched decoding | H2 | `--categorical_decoding match` — place decode thresholds so the synthetic marginal matches the real one | no |
| **S2** deeper conditioning | H1 | `--n_hops 2 --num_neighbors -1 16` | yes |
| **S3** balanced seeding | H3 | `--seed_balance_alpha 0.5` — weight seeds by `count(type)^-α` | yes |
| **S4** one-hot categoricals | H2 | `--categorical_encoding onehot` | yes |
| **S5** combined | H1+H3 | S2 and S3 flags together | yes |

### Berka results

Synthetic-only AUROC, leakage-free split. Real baselines: 0.8886 loan, 0.8739 card.

| fix | seeds | loan default | retained | card ownership | retained |
|---|---|---|---|---|---|
| baseline | 3 | 0.5917 ± 0.044 | 23.6% | 0.6021 ± 0.017 | 27.3% |
| S1 matched decoding | 3 | 0.5963 ± 0.024 | 24.8% | 0.6009 ± 0.018 | 27.0% |
| S2 `n_hops=2` | 3 | 0.5264 ± 0.122 | 6.8% | **0.6687 ± 0.017** | **45.1%** |
| **S3 balanced seeding** | 3 | **0.6613 ± 0.035** | **41.5%** | 0.6164 ± 0.050 | 31.1% |
| S4 one-hot | 1 | 0.5526 ± 0.026 | 13.5% | 0.5358 ± 0.025 | 9.6% |
| **S5 = S2 + S3** | 1 | 0.6322 ± 0.006 | 34.0% | **0.6813 ± 0.008** | **48.5%** |

### California regression guards

Does a fix break the database that already worked? Baseline shown at both 3 seeds and seed 0; the
delta column compares **seed 0 to seed 0**, since the fixes have only one generator seed.

| variant | seeds | INCTOT R² | Δ vs seed-0 base | EMPSTAT macro-F1 | Δ vs seed-0 base |
|---|---|---|---|---|---|
| real | — | 0.5527 | — | 0.6850 | — |
| baseline (3 seeds) | 3 | 0.5272 | — | 0.6752 | — |
| baseline (seed 0) | 1 | 0.5244 | — | 0.6749 | — |
| S2 `n_hops=2` | 1 | **0.5332** | **+0.0088** | 0.6738 | −0.0010 |
| S3 balanced seeding | 1 | **0.5285** | **+0.0041** | 0.6748 | −0.0000 |
| S4 one-hot | 1 | **−4.9684** | **−5.4929** | **0.3533** | **−0.3215** |

S2 and S3 pass and even nudge California *up*. **S4 fails catastrophically** — R² below zero is worse
than predicting the mean.

### Fidelity side-effects of each fix (Berka, seed 0 unless noted)

| variant | Column Shapes | 1-HOP | 2-HOP | 3-HOP |
|---|---|---|---|---|
| baseline | 96.86 | 90.38 | 95.05 | 91.63 |
| S2 `n_hops=2` | 94.73 | 88.88 | 92.86 | 89.26 |
| **S3** (3 seeds) | **96.82–96.99** | **91.78–91.91** | 95.18 | 90.47 |
| S4 one-hot | 94.06 | 80.82 | 91.94 | 83.39 |
| S5 = S2+S3 | 94.48 | 88.82 | 91.70 | 90.19 |

### The four findings worth writing up

**S1 is the most informative negative result.** It restores every categorical marginal *exactly* —
`loan.status` class B from 1–10 rows back to 24/24/23, default rate 0.073 → 0.110–0.113 against a
real 0.112 — and moves downstream utility by less than 0.005 either way. The minority-class collapse
is the most *visible* symptom, exactly what a fidelity table flags, and repairing it perfectly
changes nothing downstream. **H2 is real, reproducible, and not load-bearing.**

**A recorded prediction was falsified.** Before the retrains finished, the diagnosis predicted S2
would improve *loan default* (it opens the 0.29-AUROC gap between the restricted and full windows)
and would **not** improve *card ownership* (H4 makes structure independent of attributes). The
measurement is the reverse. The card result exposes an asymmetry in H4: structure is sampled first,
but attributes are then sampled *conditioned on structure*, and how far that information flows is
exactly `n_hops`. At `n_hops=1` an account's attributes cannot see whether its dispositions hold
cards (`card` is two hops from `account`); at `n_hops=2` they can, so account balance patterns
become correlated with card ownership. **H4 makes a structural label unfixable only when the
structure lies outside the conditioning window** — widening the window is what repairs it.

**Depth and gradient balance are coupled.** S2 alone *hurts* loan default (23.6% → 6.8%, confirmed
across three seeds; the single-seed pilot had suggested it was merely flat). Working theory:
`n_hops=2` widens every table's effective batch — `--num_neighbors -1 -1` would cost ~941,000
nodes/batch versus ~12,700 at one hop — which further dilutes `loan`'s already-thin 2.6-seed share.
S3 alone is the best single fix for loan default and the **only** fix that improves fidelity rather
than trading it away. S5 combines both and recovers most of what depth alone costs, exactly as the
dilution theory predicts.

**Fidelity and utility can move in opposite directions.** S2 raises card-ownership utility
(27.3% → 45.1%) while *lowering* fidelity (Column Shapes 96.86 → 94.73, 1-HOP 90.38 → 88.88). S3 is
the counter-example where both rise. This is a sharper version of the Phase-2 finding.

### Recommendation

**Ship S3 for single-parent-aggregate tasks; ship S5 as the best all-around fix.** S5 is within
7 points of S3 on loan default while beating every fix on card ownership, because it is the only
variant that gives the multi-hop dependency a wide-enough window *and* enough gradient to learn it.

For anyone generating a deep relational schema with GRDM: set `n_hops` to at least the distance
between a row and the tables carrying its predictive signal; cap `num_neighbors` beyond the first
hop so the batch stays tractable; and pair any increase in depth with balanced seed sampling,
because depth alone can cost more on the tables it doesn't target than it gains on the one it does.
The published `n_hops=1`, uniform-sampling default is well matched to a two-table schema like
California and badly matched to Berka on both axes at once.

---

## 7. Code changes

Eight upstream files modified: **731 insertions, 68 deletions**. Exact diff in `CHANGES.patch`
(regenerated 2026-08-25; it now also covers the S6/S7 wall-clock machinery and the S8-S11
architecture flags).
Two groups — changes *required to run at all*, and *flag-gated fix machinery* inert unless a flag is
passed. Nothing alters the paper's methodology, and §2's numbers are unaffected by the second group.

### 7.1 Required to run

| file | change | why |
|---|---|---|
| `rdbdiff/random_graph_sampler.py` | `np.concat` → `np.concatenate`, 3 sites | `np.concat` is a NumPy ≥2.0 alias; env pins 1.26.4. Crashes in the multi-parent matching path, so it always triggers on Berka (`disp` has two parents). |
| `rdbdiff/trainer.py`, `main.py` | mid-training checkpoint + resume | Upstream saves only after all 200k steps; the node SIGTERMs at ~24h, so a run could lose 82% of its training. |
| `rdbdiff/utils/data.py` | `round_synthetic_data` no longer assumes a dense label encoding | Upstream asserted every categorical column's codes are exactly `0..K-1` — true of shipped data, false of any subsample. The leakage-free split drops every client with birth-year code 73; all three seeds crashed *after* training and sampling. |
| `rdbdiff/utils/data.py` | `sort_columns` made deterministic | It built the id-column list with `list(set(df) - ...)`; set iteration order over strings varies *between processes*, so two runs of the same seed wrote CSVs with different headers. Values unaffected, but output was not reproducible. |

**Checkpointing.** `save_checkpoint(step)` writes model + EMA + optimizer + step + loss history at
every epoch boundary, via a `.tmp` file then `os.replace`, so a kill mid-write cannot leave a
truncated file. `main.py` resumes automatically when `checkpoint.pt` exists. ~184 MB, overwritten in
place.

**Dense-encoding fix.** Now clips to `real_unique.max()` — what the function's own `TODO` asked for
— and is *identical* to the old behaviour whenever codes are contiguous, since then
`max == len - 1`. This is why §2's numbers are provably unaffected.

**How the `sort_columns` bug surfaced.** Not by inspection — by the S1 verification guard (§8.2),
which reconstructs a stored sample from its cached tensor and asserts it matches the CSVs on disk.
It flagged a mismatch on every table; values were identical and only column *order* differed. A
useful example of a verification step catching a latent defect it wasn't designed to look for.

### 7.1b Wall-clock budget and resumable sampling (added for S6/S7, 2026-08-23)

The 3-hop runs made the ~24h17m SIGTERM binding in a way it had not been before, so two more
resume points were added. Both are inert unless `--max_hours` is passed.

| file | change | why |
|---|---|---|
| `main.py` | `--max_hours` flag; one deadline shared by training and sampling; `sys.exit(42)` when it is spent | A process must stop and checkpoint *itself* before the node's SIGTERM lands, rather than being killed at an arbitrary point. 42 marks a planned stop, distinguishing it from a crash. |
| `rdbdiff/trainer.py` | `deadline` arg; raises `TrainingBudgetExhausted` at the next epoch boundary once spent | Epoch boundary is the only safe stop: the checkpoint is written there and the per-epoch node-count assert stays valid on resume. |
| `rdbdiff/sampler.py` | `save/load_sampling_checkpoint`, `checkpoint_every` (default 25 timesteps), `deadline`, per-timestep progress print | **Sampling had no resume at all.** The 2000-step reverse-diffusion loop ran monolithically; at 3 hops it costs ~7.7 h and a kill anywhere in it lost the whole stage. The loop's entire state is `(t, x per node type)`, so it checkpoints cleanly. |
| `watchdog.sh` | self-restart before its own 24h SIGTERM; `flock` single-instance guard; pidfile | The watchdog was itself a long-lived process, so it was killed at ~24h too — after which nothing relaunched anything. It now hands off to a fresh detached copy every 12h and on SIGTERM. |

**Sampling checkpoint.** `{t, node_type_to_x}` written every 25 timesteps via `.tmp` + `os.replace`
(~25 MB on Berka), removed on successful completion. Bounded loss on a kill is 25 timesteps ≈ 6 min.
Verified by SIGKILLing mid-sample and confirming the relaunch resumed at the checkpointed `t`.

**Lock-fd inheritance bug (found in testing, worth recording).** The watchdog's backgrounded
`sleep` inherited the `flock` fd, so it kept the lock held after the parent exited — the successor
was refused the lock and died silently, leaving nothing watching. Fixed with `sleep "$INTERVAL" 9>&-`.
The failure only appears on the hand-off path, which is exactly the path that has to work unattended.

### 7.2 Flag-gated fix machinery

| flag | default | fix | touches |
|---|---|---|---|
| `--categorical_decoding {round,match}` | `round` | S1 | `utils/data.py` (`decode_categorical_matched`) |
| `--n_hops`, `--num_neighbors` | `1`, `[-1]` | S2 | already upstream; config only |
| `--seed_balance_alpha FLOAT` | `0.0` | S3 | `dataloader.py` (`WeightedRandomSampler`), `trainer.py` |
| `--categorical_encoding {ordinal,onehot}` | `ordinal` | S4 | `utils/graph.py`, `utils/data.py` |
| `--gnn_aggr NAME [NAME ...]` | config (`sum`) | S8 | `gnn.py` |
| `--gnn_residual` | off | S9 | `gnn.py` |
| `--degree_features` | off | S10 | `utils/graph.py` (`add_relation_degrees`), `model.py` |
| `--self_conditioning`, `--self_cond_aux_weight` | off, `1.0` | S11 | `model.py` |

`trainer.py` also relaxes its per-epoch node-count assert, valid only when seeds are drawn without
replacement — S3 draws *with* replacement, so only the total is checked and the per-type loss
average guards against a node type going unvisited. Epoch-boundary checkpointing is preserved.

At α=0.5, `loan` rises from 2.6 to 78.2 seeds per batch while `trans` keeps 3052. Full balancing
(α=1.0) would give every table ~585 and starve `trans`, which is 98% of the data and whose own
quality still counts.

### 7.2b The architecture fixes (S8-S11)

S1-S7 all left the network alone. S8-S11 change the denoiser; each is layered on S5, the best
configuration the earlier fixes produced. Full write-up in `docs/downstream/FIXES.md`.

Measured training-step cost on Berka at `n_hops=2`, as a multiple of S5: residual **1.01x**,
degree features **1.11x**, multi-statistic aggregation **1.44x**, self-conditioning **1.83x**;
S12 (the first three) **1.53x**, S13 (all four) **2.43x**. Sampling pays the same multipliers.

`rdbdiff/gnn.py` keeps relbench's `HeteroGraphSAGE` on the default path and only substitutes
`RelationalGraphSAGE` when residuals or a multi-aggregator are actually requested, so the
published baseline is bit-identical rather than merely equivalent -- checked by building both
modules under one seed and comparing every parameter tensor and the forward output.

Two details worth recording because they are easy to get wrong:

* The degree block is stored at a **uniform width across node types** (one column per edge type,
  zeros where the relation does not apply). That is what lets it survive the
  `to_homogeneous`/`to_heterogeneous` round trip `dataloader.py` performs; a per-type-width
  attribute would not. Columns are indexed by **sorted** edge type, so the real graph and the
  independently built sampled-structure graph agree on which column means which relation.
* Self-conditioning feeds the first pass's x0 estimate forward **detached**, so the auxiliary
  reconstruction loss in `GRDM.relational_gaussian_loss` is the only thing training that head.
  It is added to the total loss but deliberately kept out of `node_type_to_loss`, so the
  trainer's per-table loss averages still mean what they meant before.

### 7.3 New files

| file | purpose |
|---|---|
| `rdbdiff/gnn.py` | `RelationalGraphSAGE` (multi-statistic aggregation, residuals) and `build_gnn`, which falls back to relbench's module on the default path. |
| `resume_runs.sh` | idempotent launcher; skips running/finished seeds, resumes the rest. `RDBDIFF_{DATASET,GPUS,EXTRA_ARGS,THREADS}`. |
| `watchdog.sh` | re-runs the launcher on a timer so experiments survive the ~24h kill unattended; pins one GPU per experiment. |
| `downstream/` | the entire utility-evaluation harness (§8). |
| `configs/*_fastsample.json` | per-experiment configs; identical to upstream except `sample_batch_size 20000 → 100000`, plus one per fix with only `dataset_name` changed. |

`sample_batch_size` is fidelity-safe: sampling uses `num_neighbors=-1` and `disjoint=False`, so batch
grouping is purely a scheduling choice. **Training hyperparameters are untouched** (K=1, T=2000,
lr 6e-4, wd 1e-5, batch 4096, 200k steps / 100k for California).

---

## 8. The evaluation harness (`downstream/`)

### 8.1 Modules

| module | role |
|---|---|
| `tasks.py` | the four task definitions: label extraction + relational feature construction |
| `run.py` | experiment driver: regimes, seeds, dataset/fix registry |
| `evaluate.py` | fixed-hyperparameter XGBoost fit, metrics, bootstrap CI |
| `data.py` | loading, type alignment, FK integrity assertions |
| `splits.py`, `make_berka_split.py` | the leakage-free Berka split |
| `redecode.py` | re-decode a cached sample without resampling (S1); also the baseline-identical guard |
| `diagnose.py` | the root-cause tests A1/A2/A3/A5 |
| `report.py`, `make_artifact.py` | regenerate the markdown tables and the HTML report |

### 8.2 Verification built into the harness

- **Baseline-identical guard.** `redecode.py --decoding round --verify` reconstructs the stored
  baseline from its cached sample and asserts a match — the proof that S1's machinery left the
  reproduction untouched. All 3 seeds: identical.
- **Determinism.** Re-running the harness produces byte-identical metrics.
- **Leakage guards.** Aggregates may not derive from the target column; `card` contributes the label
  only; train/test id-disjointness asserted per table.
- **FK integrity** asserted on both real and synthetic tables before any feature is built.

---

## 9. Reproducing all of it

```bash
cd /scratch/s25017/ClavaDDPM/rdb-diffusion

# Source changes onto a clean upstream checkout
git checkout bc0fb67 && git apply CHANGES.patch

# Phase 1 — fidelity
scripts/resume_runs.sh                                   # berka seeds 0 1 2
RDBDIFF_DATASET=california scripts/resume_runs.sh
RDBDIFF_DATASET=ccs scripts/resume_runs.sh

# Phase 2 — leakage-free split + downstream utility
conda run -n tabsyn python -m downstream.make_berka_split
RDBDIFF_DATASET=berka_split scripts/resume_runs.sh 0 1 2
conda run -n tabsyn python -m downstream.run --dataset california
conda run -n tabsyn python -m downstream.run --dataset berka_split

# Phase 3 — diagnosis
conda run -n tabsyn python -m downstream.diagnose

# Phase 3 — fixes
conda run -n tabsyn python -m downstream.redecode --dataset berka_split \
    --seeds 0 1 2 --decoding round --verify        # guard: baseline unchanged
conda run -n tabsyn python -m downstream.redecode --dataset berka_split \
    --seeds 0 1 2 --decoding match                 # S1, no retrain

RDBDIFF_DATASET=berka_split_s2 RDBDIFF_GPUS=0 \
  RDBDIFF_EXTRA_ARGS="--n_hops 2 --num_neighbors -1 16" scripts/resume_runs.sh 0 1 2
RDBDIFF_DATASET=berka_split_s3 RDBDIFF_GPUS=1 \
  RDBDIFF_EXTRA_ARGS="--seed_balance_alpha 0.5" scripts/resume_runs.sh 0 1 2
RDBDIFF_DATASET=berka_split_s4 RDBDIFF_GPUS=2 \
  RDBDIFF_EXTRA_ARGS="--categorical_encoding onehot" scripts/resume_runs.sh 0
RDBDIFF_DATASET=berka_split_s5 RDBDIFF_GPUS=3 \
  RDBDIFF_EXTRA_ARGS="--n_hops 2 --num_neighbors -1 16 --seed_balance_alpha 0.5" scripts/resume_runs.sh 0

# California regression guards (same EXTRA_ARGS, dataset california_s{2,3,4,5})
for f in s1 s2 s3 s4 s5; do
  conda run -n tabsyn python -m downstream.run --dataset berka_split_$f
done

# Regenerate every table and the HTML report
conda run -n tabsyn python -m downstream.report
conda run -n tabsyn python -m downstream.make_artifact
```

Long jobs: `setsid nohup scripts/watchdog.sh < /dev/null >> logs/infra/watchdog.log 2>&1 & disown`.

**Environment.** Shared conda env `tabsyn`, always via `conda run -n tabsyn`. Installed with
`--no-deps` to protect existing pins (notably `numpy==1.26.4`). Kept torch 2.6.0+cu124 rather than
the repo's pinned 2.5.1. Full detail in `REPRODUCTION.md` §3.

---

## 10. Caveats and limits — state these in the report

1. **Berka's loan test set has 19 positives.** Its AUROC column is genuinely noisy (real-only
   bootstrap 95% CI [0.8106, 0.9297]). Treat loan-task magnitudes as indicative and lean on the
   ordering plus the card task, which has 220 positives.
2. **S4 and S5 have one generator seed.** S4 was stopped deliberately once its California guard came
   back catastrophic. S5's error bars are the single most valuable outstanding run.
3. **The `california_s5` guard has not finished** (sampling stage at time of writing), so S5 is not
   yet confirmed non-regressive on California.
4. **CCS 1-HOP is 2.77pt below the paper** and not root-caused; three seeds cannot separate seed
   variance from a systematic gap.
5. **The card-ownership task is a dependency probe, not a forecast.** Its features are partly
   downstream of the label (no timestamp separates cause from effect); this is stated in the code
   and should be stated in the report.
6. **S2's loan-default regression is not fully explained.** The batch-dilution theory is consistent
   with S3 and S5 recovering the loss, but it has not been tested directly.
7. **Only the GRDM column of Table 1 was reproduced** — no baselines, no Table 2 ablations, no
   appendix experiments.
