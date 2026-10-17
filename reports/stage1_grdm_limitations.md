# AMEX Stage 1 — Fundamental limitations of GRDM

**Diagnostic investigation report** · 2026-10-14 · repository `/storage/s25017/AMEX/java_d` (branch `paper-repro`)

**Scope.** This stage only inspected code and analysed outputs that already existed. No model was trained,
no new sampling was run, no checkpoint was read or written, and no existing result file was modified.
New computation (CPU only) was limited to diagnostics on existing synthetic CSVs, written to the new
folder `results/stage1_limitations/`.

**Evidence labels used throughout**

| label | meaning |
|---|---|
| **VERIFIED-code** | read directly in the source; file and line given |
| **VERIFIED-data** | computed in this stage from existing outputs (`results/stage1_limitations/outputs/*.json`), 3 generator seeds unless stated |
| **VERIFIED-prior** | an existing result file, numerically re-checked by the 2026-10-13 audit (`results/consolidated_audit/AUDIT_REPORT.md`) or by this stage |
| **REPORTED-prior** | stated in an earlier experiment's report from its own outputs; not re-derived here |
| **HISTORICAL** | from the Aug–Oct study (`reports/grdm-final-report.html`); its checkpoints and synthetic data were on `/scratch` and no longer exist, so it cannot be re-verified |
| **PRELIMINARY** | a real measurement on one seed or one fit |
| **HYPOTHESIS** | an explanation not yet tested by an intervention |
| **PROPOSED** | an experiment not run |
| **MISSING** | evidence that does not exist yet |

---

## 0. Status of the protected experiments (Table 5)

**The California 11-regime downstream job (Table 5) is not running.** Nothing in this stage stopped it,
and it was not restarted.

| item | finding |
|---|---|
| launcher | `/tmp/claude-1070/-storage-s25017-AMEX-java-d/4941965e-.../scratchpad/run_missing_regimes.sh`; three `downstream.run` arms in parallel (`california_agg_ft`, `california_agg_ft_label_free`, `california_ft_control`), regimes `real_25 real_50 real_75 blend_25 blend_75 mixed_25 mixed_50`, `--out-suffix _missing` |
| launched | 2026-10-13 15:42 (`java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/logs/missing_regimes_master.log`) |
| last activity | all three arm logs stop at 16:18–16:19 on 2026-10-13, partway through `california_inctot` seed 2 (`logs/downstream_*_missing.log`); the master log has no "done" lines |
| processes | none: no Python process owned by `s25017` existed at 2026-10-14 00:05; the GPU processes on the node belong to other users |
| outputs | no `*_missing.json` was written. `downstream.run` writes its JSON only at the end of a task, so the ~37 minutes of completed fits are not on disk except as log lines |
| probable cause | the job was a background task of an earlier Claude session and most likely ended with it. **Not verified** |
| consequence | 42 of 88 Table 5 cells remain empty. This is missing evidence, **not** a negative result for any method. `results/consolidated_audit/AUDIT_REPORT.md` still describes the job as running; that statement is now stale (the file was left untouched) |
| to resume (not done) | rerun the launcher above. It reuses existing synthetic data, needs no GPU, and takes ~10 h of CPU at 2 threads per arm |

No other experiment of this project is running. All other worktrees (`java_d-agg`, `java_d-aggft`,
`java_d_degcond`, `java_d_degft`, `java_d-calagg-ft`) have finished runs, with outputs on disk.

---

## 1. Executive summary

**Bottom line.** GRDM generates each row with a local denoiser that sees only the row's own noisy
values and its direct neighbours' noisy values. It keeps no state for an entity or a group, and the
graph structure is drawn before, and independently of, every attribute. Two consequences can be
measured directly in the synthetic data:

* **Rows that share a parent behave as independent draws given the parent's columns.** This holds to
  within 0.1 percentage point on two databases.
* **Cross-relation structure is uncorrelated.**

These are exactly the dependencies the downstream tasks aggregate. The evidence that this is
architectural is stronger than "a metric is poor": the code fixes what each denoiser can see, a
model-free independence test matches the code's prediction, and a change to the information flow
alone, at matched data, objective and budget, breaks the independence.

**Main findings**

1. **Sibling independence (VERIFIED-data, both databases).**
   * California: baseline GRDM puts exactly one household head in **37.7%** of households, against
     **93.3%** in real data. A model in which each member's head status is drawn independently given
     household size predicts **37.6%**.
   * Berka: **72.7%** of accounts have exactly one OWNER disposition, against **100%** real; the
     independence prediction is **72.7%**.
   * The between-account share of transaction-balance variance is **0.020 vs 0.553 real**. The
     between-household share of AGE variance is **0.047 vs 0.291**. The head–spouse AGE correlation is
     **0.17 vs 0.93**.
   * Within-row conditionals are close to the real-vs-real noise floor.
2. **Coordination restores what the downstream task uses, when the model can see siblings.**
   Aggregate conditioning trained from scratch (Berka, same data, steps and schedule) gives:
   * exactly one OWNER in **99.1%** of accounts (independence prediction 89.7%);
   * loan child-aggregate → label signal retained **0.74** (baseline 0.04), and loan train-on-synthetic
     AUROC 0.507 → 0.826;
   * fixed-batch ε-loss −15%, concentrated on `trans`/`loan`/`order` (−20 to −27%).

   **But it over-couples:** transaction-balance share 0.695 vs 0.553 real, order-amount 0.218 vs 0.003,
   California INCTOT 0.113 vs 0.076.
3. **Card ownership is lost by every model tested (VERIFIED-data).** The child-aggregate → card signal
   retained is −0.04 to +0.01 in all six Berka configurations, and `trans_balance_mean` AUROC is
   0.838 → 0.49–0.50 everywhere. Card existence is decided by the attribute-agnostic structure sampler,
   three hops from the transactions. No variant gives the transactions a path to it.
4. **The structure sampler drops cross-relation correlations (VERIFIED-code and data).**
   * ρ(#trans, #order): 0.26 → 0.00.
   * ρ(#order, #loan): 0.29 → 0.01.
   * Share of loan accounts with ≥1 standing order: 1.00 → 0.85. The structure is identical across all
     configurations (verified by digest), so no attribute-side change can repair this.
5. **Sampling is fragile, and fine-tuning exposes it (VERIFIED-code and data; cause partly HYPOTHESIS).**
   * β₁₉₉₉ = 0.999 makes the first reverse step multiply ε-error by **31.6**, and x̂₀ is never clipped.
   * In 200k steps the `loan` head sees only **259** training samples at t = 1999, against 400,766 for
     `trans`.
   * Rows outside the data range in model space, loan table: 1.5% baseline, 4.2% aggregate-scratch,
     16% degree-scratch, **51% degree fine-tune, 54% aggregate joint fine-tune**.
   * Collapse therefore occurs **without** an aggregate pathway (degree fine-tune). Aggregates amplify
     it; they are not its root.
   * Even the original model's raw samples contain values up to |x| ≈ 40,000 against a real bound of
     5.2, hidden by the decoder's clipping.
6. **Why California fidelity rose but utility did not (VERIFIED-data / PRELIMINARY).**
   * The restored co-member structure is worth ~+0.002 R² for income even with real data.
   * Aggregate fine-tuning degraded the within-row conditionals the task uses: INCTOT | age × EDUC gap
     0.21 → 0.32; EMPSTAT total-variation distance 0.0066 → 0.016.
   * The task-relevant household → member gap (+0.051 R² real vs +0.030 synthetic) is untouched.
     About 40% of it traces to **PUMA**, a 233-code nominal column diffused as a Gaussian ordinal:
     η²(INCTOT; PUMA) is 0.062 real vs 0.004 synthetic.
7. **The paper's k-HOP metrics cannot see the lost dependencies (VERIFIED-code).** They score only
   (top-ancestor column, bottom-descendant column) pairs on joined rows, never sibling or cousin pairs.
   Both Berka 3-HOP paths start at the fixed `district` table, which is why 3-HOP is flat across every
   configuration.
8. **Scratch vs fine-tuning is confounded (VERIFIED-code).** The two differ in steps (200k vs
   44,880 / 21,120), LR schedule, frozen phases, initialisation and training seed, and the fine-tuned
   arms are contaminated by sampling collapse. Existing experiments cannot isolate "training from
   scratch".

**Recommendation.** Develop a substantially new coordination mechanism for relational generation,
rather than adding further conditioning vectors to GRDM. Four core components are implicated: the
denoiser's information flow, the structure generator, the categorical representation and the sampler
parameterisation. The single capability the evidence justifies most is **explicit, generated
group-level (entity) state with within-group interaction, calibrated so that it neither under- nor
over-couples siblings**. Build it on GRDM's existing diffusion and data code so comparisons stay
controlled. Test it first on California (§14, experiment E1), with zero-training falsification checks
(E0) beforehand.

---

## 2. Actual GRDM architecture (reconstructed from code)

All paths are relative to `/storage/s25017/AMEX/java_d` unless stated. The baseline code path is
upstream GRDM (`bc0fb67`) plus inert, flag-gated local patches (`docs/CHANGES.md`). The `paper_repro`
runs use `configs/{berka,california}_fastsample.json` with `n_hops=1`, `num_neighbors=[-1]`,
`gnn_aggr="sum"` and no fix flags (confirmed in `logs/train/paper_repro_berka_seed0.log`, lines 1–40).

### 2.1 Data representation (VERIFIED-code)

| question | implementation | where |
|---|---|---|
| table | one node type per table | `rdbdiff/utils/graph.py:17` `make_pkey_fkey_graph` |
| record | one node; non-key columns form one float vector `x ∈ R^{d_τ}`, numerical first, then categorical | `graph.py:44-85`, `rdbdiff/utils/data.py:30` `sort_columns` |
| foreign keys | removed from the features. Each FK becomes edge types `(child, f2p_fk, parent)` and `(parent, rev_f2p_fk, child)`. No edge features | `graph.py:101-122` |
| parent–child association | explicit edges only; nothing in a row's features says how many children it has (no degree features in the baseline) | `graph.py`; `add_relation_degrees` is behind `--degree_features`, off in `paper_repro` |
| numerical columns | per-table `QuantileTransformer(output_distribution="normal", n_quantiles=min(N/30,1000))` | `data.py:97-163` |
| categorical columns | **integer codes quantile-transformed and diffused as continuous values**, then rounded back. One-hot and multinomial paths exist behind flags and are off | `data.py:118-135`, `main.py:108-135` |
| variable child counts | neighbourhood **sum** inside `SAGEConv`, then per-node `LayerNorm`, which divides out the sum's magnitude | relbench `HeteroGraphSAGE` (`relbench/modeling/nn.py:131-179`) |
| table-level / group-level information | **none.** No table token, no group latent, no per-entity state beyond the row's own columns | `rdbdiff/model.py:362-547` |
| dimension tables (Berka `district`) | never diffused: held at t = 0, never a seed, no head | `model.py:160-165`, `dataloader.py:257-268` |
| structure | **sampled first and independently of attributes** by `RandomGraphSampler`. Each relation's in-degrees are drawn i.i.d. from that relation's real degree histogram; multi-parent tables (`disp`) are matched by shuffling stubs at random. The code's own comments mark joint or conditional degree sampling as "not implemented" | `rdbdiff/random_graph_sampler.py:88-130, 199-204, 244-255, 305-306` |

### 2.2 Denoising network (VERIFIED-code)

`GNNDenoiser.forward` (`rdbdiff/model.py:478-547`), for every node *u* in a sampled subgraph:

```
h_u^(0) = W_type(u) · x_t^(u) + φ(sinusoid(t_u))            φ = Linear(128)–SiLU–Linear(128)
h^(1)   = ReLU(LayerNorm_node( Σ_r SAGE_r(h^(0)) ))         SAGE_r(h)_v = W_r^l Σ_{u∈N_r(v)} h_u + W_r^r h_v   (sum aggr)
ε̂_v    = MLP_type(v)(h_v^(1))                               seed nodes only
```

* **Inputs:** noisy features `x_t` of every node in the seed's K-hop subgraph, and one integer t per
  node (shared within a subgraph). **Output:** ε̂ for the seed rows only.
* **Timestep:** sinusoidal embedding, an MLP, and an addition to the input projection.
* **Conditioning:** the only conditioning is the neighbours' noisy rows, passed through **one**
  message-passing layer (`num_layers = n_hops = 1`). There is no attention, no hop-specific processing
  and no residual. Separate SAGE weights per edge type are summed over edge types.
* **Sharing:** one GNN for all tables; a separate input projection and MLP head per table. All columns
  of a row share one head and one noise level.
* **Parameter split (Berka):** 11.54 M in total. **95% (10.98 M) are in the per-table MLP heads**,
  4.6% (0.53 M) in message passing (`logs/train/paper_repro_berka_seed0.log:53-58`). California:
  8.65 M, of which 66 k are in the GNN.
* **What a row can see.**
  * A `trans` seed's subgraph holds the transaction and its account. It does **not** hold the account's
    other ~235 transactions, its loan, or its dispositions' cards (all two or more hops away).
  * An `individual` sees its household's noisy columns. It does not see co-members or the household's
    size.

### 2.3 Aggregate conditioning (VERIFIED-code; `java_d-agg/rdbdiff/aggregates.py`, diff in `experiments/GRDM_AGGREGATE_CONDITIONED_SCRATCH/code_snapshot.patch`)

| question | implementation |
|---|---|
| what is computed | for each parent and relation: log1p(count), plus mean/std/min/max of chosen child columns, **over the parent's full child set**, in model (quantile-normal) space. Berka: 31 statistics on `account` (trans, order, loan, disp) and `disp` (card). California: 20 on `household` (16 label-free) |
| raw or learned | raw statistics, standardised with μ, σ fitted once on the clean training graph and clipped to ±10 (`aggregates.py:256-261`). Mapped by a per-table `Linear–SiLU–Linear` (last layer zero-initialised) |
| where it enters | **added to the parent's input embedding `h^(0)` before message passing.** Children see it only through the parent's message |
| which attributes it affects | directly, the parent's head; indirectly, every child's denoiser through the single GNN layer. It does **not** reach grandchildren: e.g. no "cards among my dispositions" on `account` |
| objective | unchanged (ε-MSE) |
| available when | **training:** `attach_batch` rebuilds the statistic at the subgraph's t; in-subgraph children reuse the exact noisy value the denoiser sees, all others are re-noised with `q_sample`. **Sampling:** `attach_full` recomputes it before every reverse step from the synthetic graph's current `x_t` |
| real vs synthetic | same function. At training its inputs are forward-noised **real** children; at sampling they are **model-generated** children, which can include blown-up rows. This is the exposure-bias path in §7 |
| variable group sizes | handled (CSR index, count feature, empty-set mask). One parent–child level only |

Degree conditioning (`java_d_degcond/rdbdiff/degree_conditioning.py`, `model_degree.py`) adds
standardised log1p per-relation child counts of the row and of its parents (30 columns on Berka) plus
zero-count flags. They are injected before the GNN and again at the head, both zero-initialised.
**VERIFIED-code** via `experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/architecture_notes.md` and the module.

### 2.4 Training objective and optimisation (VERIFIED-code)

* **Forward process** (`tab_ddpm/gaussian_multinomial_diffsuion.py:206`):
  `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`.
  * Cosine schedule `ᾱ(s) = cos²(((s+0.008)/1.008)·π/2)`, `β_t = min(1 − ᾱ(t+1)/ᾱ(t), 0.999)`, T = 2000
    (`:20-62`).
  * One `t ~ U{0..1999}` per seed subgraph; every node in the subgraph is noised at that t
    (`rdbdiff/model.py:137-158`).
* **Loss** (`model.py:238-288`):
  `L = Σ_{τ present in batch} (1/|S_τ|) Σ_{v∈S_τ} (1/d_τ) ‖ε_v − ε̂_v‖²`.
  * A sum of per-table means over the batch's seed rows. Each present table has equal weight per batch.
  * There is no relational, aggregate, consistency or auxiliary loss on the baseline path.
* **Seeds:** drawn uniformly over all non-dimension rows (`dataloader.py`, `shuffle=True`,
  `disjoint=True`).
  * Every row is visited ~759 times in 200k steps. But `loan` has 2.6 seed rows per batch, appears in
    92.5% of batches, and yields 259 training samples at t = 1999 over the whole run, against 400,766
    for `trans` (VERIFIED-data, `outputs/schedule_amplification.json`).
* **Optimiser:** AdamW, lr 6·10⁻⁴, weight decay 10⁻⁵, linear decay to 0, no warmup, no gradient
  clipping (`rdbdiff/trainer.py:51-84`). EMA rate 0.999 every step (`tab_ddpm/utils_train.py:17`).
  Batch 4,096 seeds. 200k steps (Berka) or 100k (California).
* **Checkpoint selection:** the **final** EMA weights (`model_ema.pt`) are used. `best_epoch` is
  logged but never used (`trainer.py:154-165`, `main.py:535-536`).
* **Table types:** no separate objective per table type beyond the per-table mean.
* **Fine-tuning recipes** (other worktrees; VERIFIED-code from their configs and reports):

  | recipe | phase 1 | phase 2 | notes |
  |---|---|---|---|
  | Berka aggregate FT | 5,280 steps, backbone frozen, new layers only, lr 6e-4 | 39,600 joint steps, peak lr 1e-4, 264-step warmup, linear decay | fresh AdamW, training seed 1000+N |
  | California aggregate FT | 5,070 steps, frozen, lr 6e-4 | 20,280 joint steps, lr 1e-4 | |
  | California control | — | 20,280 joint steps, lr 1e-4, no aggregates | |
  | Berka degree FT | 5,280 steps, encoder only, lr 3e-5 | 15,840 joint steps, lr 3e-5 → 0 | |

### 2.5 Sampling and reconstruction (VERIFIED-code)

1. **Structure.** `RandomGraphSampler(seed)` builds every table's rows and FKs (`main.py:557-600`).
   The same seed gives the same structure; all six Berka configurations share it per seed (digest-verified, §6).
2. **Initial noise.** `x_T ~ N(0, I)` for every non-dimension row (`rdbdiff/sampler.py:77-86`).
3. **Reverse steps.** For t = 1999 … 0 (`sampler.py:92-149`), every row is denoised once at the same
   global t:
   * full one-hop neighbourhoods, `disjoint=False`, a fresh `NeighborLoader` per step;
   * `x̂₀ = √(1/ᾱ_t) x_t − √(1/ᾱ_t − 1) ε̂`;
   * `μ = c₁ x̂₀ + c₂ x_t`;
   * `x_{t−1} = μ + √β_t z`.

   **x̂₀ is never clipped.** `clip_denoised` is passed but `gaussian_p_mean_variance` ignores it
   (`gaussian_multinomial_diffsuion.py:233-281`). Conditioned models recompute aggregates before each
   step; nothing else differs between the original and the conditioned samplers.
4. **Amplification.** The derivative of `x_{t−1}` with respect to ε̂ is
   `β_t / (√(1−ᾱ_t) √α_t)`: **31.6 at t = 1999**, 1.5 at 1998, 0.83 at 1997, and below 0.5 for every
   later step (`outputs/schedule_amplification.json`).
5. **Decoding** (`main.py:640-680`, `data.py:280-372`):
   * inverse quantile transform, which parks out-of-range values at the training min/max;
   * categorical codes are rounded and clipped to `[0, max code]`;
   * integer columns with ≤ 32 values are snapped to the nearest real value;
   * other numerical columns are rounded to 6 decimals.
6. **Constraints enforced:** PK uniqueness and FK integrity (by construction of the structure) and
   categorical code range. **Not enforced:** date order, ledger identities, roles (one owner, one
   head), cross-table logic.

### 2.6 What is verified and what is not

**Verified in code:** everything in §2.1–2.5.

**Inferred, not directly measured:**

* That `LayerNorm(mode="node")` removes count information. This is mathematically true for the
  magnitude of the sum, but the root term `W^r h_v` is normalised together with it, so some degree
  signal could survive in direction. The California group-size result in §6.3 is consistent with
  near-total loss.
* That multi-step reverse dynamics could in principle carry sibling information through the parent's
  state. The data in §6 show they effectively do not.
* Reproduction of the paper's equations from the paper itself: the PDF is not on disk, so the
  equations above are the code's.

---

## 3. Architecture and training data flow

```
                   ┌───────────────────────── REAL DATABASE (CSV + domain + dataset_meta) ─────────────────────────┐
                   │ tables → nodes; FK → edges (child→parent, parent→child); district fixed (dimension table)      │
                   └──────────────┬──────────────────────────────────────────────────────────────┬───────────────┘
                                  │ per-table QuantileTransformer (numerical AND categorical codes) │ degree histograms
                                  ▼                                                              ▼ (per relation, independent)
 TRAINING                    graph_real (HeteroData, x ∈ N(0,1)-ish)                      RandomGraphSampler (sampling only)
   seed rows ~ Uniform(all non-dimension rows) ──► NeighborLoader(disjoint, K=1, all neighbours)
   t ~ U{0..1999} per subgraph; x_t = √ᾱ x0 + √(1-ᾱ) ε for EVERY node of the subgraph
        │                                   [agg variant: a_t(parent) = stats(children x_t over FULL child set) → AggMLP]
        ▼
   h0 = W_type x_t + φ(t) (+AggMLP(a_t)) (+deg) ──► 1 × SAGE(sum) → LayerNorm → ReLU ──► MLP_type(seed) ──► ε̂(seed)
        │                                                                                               │
        └────────────────────────── loss = Σ_tables mean_seed ‖ε − ε̂‖²  (no relational term) ◄──────────┘
                                    AdamW 6e-4 → 0, EMA 0.999, final EMA kept

 GENERATION
   structure: for each relation independently, draw child counts from its real degree histogram; random stub matching
        │        (no dependence on attributes, on other relations' counts, or on parent's own degree)
        ▼
   x_T ~ N(0,I) for every row ──► for t = 1999..0: [agg: recompute a_t from current synthetic x_t]
                                    every row: ε̂ from its 1-hop noisy neighbourhood → x̂0 (unclipped) → x_{t-1}
        ▼
   inverse quantile (clips to train min/max) → round/clip categorical codes → snap small-integer columns → CSV
```

**Where relational information enters:** only through the single message-passing layer, and only
from the neighbours' noisy values at the same t. For t > 1600 (20% of steps) √ᾱ_t < 0.31, so the
neighbours are mostly noise. Rows two or more hops apart have no within-step path. Across steps,
their only channel is the observed columns of the shared neighbour (Berka `account` has two:
`frequency` and `account_date`).

---

## 4. Verified experimental findings (existing results)

### 4.1 Berka relational fidelity, five configurations (VERIFIED-prior; mean ± population SD, ddof=0, n=3)

Source: `results/relational_fidelity_consolidated/relational_fidelity_table.csv`.

| metric | original | degree scratch | degree FT | aggregate scratch | aggregate joint FT¹ | aggregate frozen FT |
|---|---|---|---|---|---|---|
| Column Shapes | 97.00 ± 0.11 | 96.02 ± 0.09 | 90.91 ± 1.10 | 96.44 ± 0.16 | 84.46 ± 0.78 | 96.88 ± 0.08 |
| Intra-table trends | 98.31 ± 0.04 | 97.51 ± 0.18 | 94.83 ± 0.60 | 97.52 ± 0.09 | 88.25 ± 0.44 | 98.14 ± 0.06 |
| 1-HOP | 91.49 ± 0.20 | 92.15 ± 0.04 | 88.88 ± 0.56 | 92.70 ± 0.05 | 80.91 ± 2.18 | 91.18 ± 0.25 |
| 2-HOP | 95.61 ± 0.07 | 95.05 ± 0.17 | 91.10 ± 0.96 | 95.02 ± 0.09 | 84.41 ± 1.55 | 95.66 ± 0.19 |
| 3-HOP | 92.71 ± 0.45 | 92.65 ± 0.59 | 92.99 ± 0.91 | 93.05 ± 0.77 | 92.85 ± 0.55 | 92.63 ± 1.22 |
| aggregate KS ↓ | 0.273 ± 0.001 | 0.204 ± 0.001 | 0.220 ± 0.003 | 0.143 ± 0.005 | 0.339 ± 0.024 | 0.257 ± 0.006 |
| aggregate × aggregate \|Δρ\| ↓ | 0.245 ± 0.002 | 0.180 ± 0.003 | 0.246 ± 0.017 | 0.076 ± 0.005 | 0.284 ± 0.057 | 0.221 ± 0.009 |

¹ This arm's sampling collapsed (§7). Cardinality is 99.65 ± 0.05 everywhere: the structure is shared,
so Cardinality carries no information about the attribute model.

### 4.2 Berka downstream, synthetic-only (TSTR) and real + synthetic (VERIFIED-prior; ddof=0; **leakage caveat**: every generator saw the test accounts)

Source: `results/paper_repro/downstream/berka_prediction_tables/table_{a,b}_*.csv`.

| configuration | loan TSTR AUROC/AP | loan real+syn AUROC | card TSTR AUROC/AP | card real+syn AUROC |
|---|---|---|---|---|
| real only (reference) | 0.939 / 0.762 | — | 0.874 / 0.480 | — |
| original GRDM | 0.507 / 0.136 | 0.928 | 0.573 / 0.191 | 0.873 |
| degree scratch | 0.789 / 0.356 | 0.944 | 0.641 / 0.222 | 0.872 |
| degree FT | 0.578 / 0.174 | 0.916 | 0.617 / 0.220 | 0.876 |
| aggregate scratch | **0.826 / 0.455** | 0.936 | **0.648 / 0.235** | 0.866 |
| aggregate joint FT | 0.595 / 0.167 | 0.921 | 0.606 / 0.209 | 0.869 |
| aggregate frozen FT | 0.534 / 0.164 | 0.921 | 0.569 / 0.181 | 0.869 |

The loan test set has 19 positives. The original's TSTR AUROC alone varies by an SD of 0.067 across
seeds.

### 4.3 California (VERIFIED-prior; sample SD, ddof=1, n=3; leakage-free by construction)

Source: `results/consolidated_audit/table4_*.csv`, `table5_*.csv`; `java_d-calagg-ft/.../metrics/comparison.json`.

| quantity | original | control (+20k steps) | aggregate FT | label-free aggregate FT | floor |
|---|---|---|---|---|---|
| household-summary KS ↓ | 0.149 | 0.149 | **0.054** | 0.055 | 0.003 |
| within/total variance ratio MAE ↓ | 0.156 | 0.157 | **0.070** | 0.070 | 0.001 |
| held-out ε-loss ↓ | 0.3003 | 0.2996 | **0.2874** | 0.2876 | — |
| TSTR income R² (real 0.5527) | 0.5274 ± 0.0025 | 0.5273 | 0.5253 | 0.5236 | — |
| TSTR employment macro-F1 (real 0.6850) | 0.6757 ± 0.0008 | 0.6752 | 0.6742 | 0.6739 | — |

All 18 paired TSTR / mixed / blend differences of the aggregate arms against the original are negative
(−0.0004 to −0.006). Table 5 is 46/88 populated (§0).

### 4.4 Existing ablations of the aggregate pathway

* **California (VERIFIED-prior).** Zeroing the aggregate block at evaluation raises held-out loss to
  0.3405 ± 0.0030; shuffling it across households raises it to 0.3682 ± 0.0061. The conditioned model
  scores 0.2874 and the original 0.3003. The model therefore uses the block, and depends on it beyond
  the original's capability.
* **Berka (REPORTED-prior, `java_d-aggft/.../metrics/sampling_path_probe_seed0*.json`).**
  Teacher-forced ε-MSE at t = 100 rises from 2.69 to 24.8 when aggregates are zeroed in the
  **from-scratch** model, and only from 2.91 to 3.55 in the **joint fine-tuned** model. Scratch
  training learned to rely on the pathway; fine-tuning barely did.

### 4.5 Historical study (HISTORICAL; leakage-free `berka_split`; checkpoints lost)

Source: `reports/grdm-final-report.html`, `docs/downstream/{DIAGNOSIS,FIXES}.md`.

* **Real-data window ceilings (generator-independent).** Loan: own row 0.60, plus one-hop parent 0.56,
  everything 0.94. Card: 0.60, 0.67, 0.87. The published generator (0.51 / 0.60) sits at the own-row
  ceiling.
* **Child aggregates "absent rather than shifted".** Synthetic → synthetic card AUROC is 0.594, and
  the aggregate confidence intervals cover 0.5 in every seed.
* **Depth.** n_hops = 2 with 16 neighbours and balanced seeding retained 50.4% of the card signal
  (loan 24.9%), at about −1.8 Column Shapes and 2× compute. Three hops did not help.
* **Six-column data-level aggregates (`agg`)** on account and disp retained card 48.7% and loan 35.8%
  (3 seeds). This is the only variant reported to move card substantially without extra depth.
* **ClavaDDPM, same harness** (its generator saw the test rows): card 60.5%, loan 42.6% retained,
  against GRDM's 29.3% and 7.2%.
* **Failed fixes on California.** One-hot categoricals: R² −4.97. Multinomial diffusion: loss NaN.

---

## 5. Failure-mode matrix

Priority: **H** = blocks or defines Stage 2; **M** = must be controlled; **L** = note.

| # | Observation | Supporting evidence | Verified? | Possible explanations | Alternative explanations | Additional test required | Architectural implication | Priority |
|---|---|---|---|---|---|---|---|---|
| F1 | Siblings are generated as conditionally independent draws given the parent's columns | Exactly-one-head 0.377 vs independence prediction 0.376 (real 0.933); exactly-one-OWNER 0.727 vs 0.727 (real 1.000); trans.balance ε² 0.020 vs 0.553; AGE ε² 0.047 vs 0.291; head–spouse AGE corr 0.17 vs 0.93 (§6) | VERIFIED-data (2 datasets × 3 seeds) | No within-step sibling path at K = 1, no group state, and a parent with too few columns to relay group factors | Under-training (no: the control with +20% steps gives 0.376 vs 0.376); ordinal RELATE decoding (distorts the marginal, cannot create independence) | K = 2 or sibling-set arm; longer-training baseline (§12) | **Yes**: needs a group-level coordination channel | H |
| F2 | Child-aggregate → label dependencies lost | Loan child-aggregate retention 0.04 ± 0.13 (14 of 28 carriers reversed); card 0.005 ± 0.031; card `trans_balance_mean` AUROC 0.838 → 0.498 | VERIFIED-data; matches HISTORICAL | Consequence of F1: aggregates have almost no between-parent variance; window | Harness bugs (no: harness byte-identical across worktrees; the date window was fixed 5 Oct) | Same metric on E1/E2 arms | Yes, via F1 | H |
| F3 | Bottom-up aggregates restore sibling and loan structure but over-couple | Aggregate scratch: OWNER 0.991 (prediction 0.897); loan retention 0.74; but trans.balance ε² 0.695 vs 0.553, order 0.218 vs 0.003, California INCTOT 0.113 vs 0.076; loan default rate 0.139 vs 0.111 | VERIFIED-data | Positive feedback: parent statistics computed from the children's current state, children conditioned on them (sampling-time exposure bias) | Normalisation or clipping choices; statistic choice (order amounts included) | Trace ε² along the reverse chain; freeze aggregates after t*; compare a top-down generated latent (E1) | Coupling must be **calibrated**, not raw feedback | H |
| F4 | Card ownership (a structure-defined label) is not recovered by any configuration | Card child-aggregate retention −0.037 to +0.013 in all six; P(card \| balance octile) flat at ~0.16 vs real 0.00 → 0.44 | VERIFIED-data | Card existence is set by the attribute-agnostic sampler; transactions are 3 hops from card; no variant gives `account`/`trans` the count of cards among the account's dispositions (VERIFIED-code: aggregate spec and degree layout) | The association is partly post-issue in real data (task is a dependency probe, not a forecast). This does not change what a faithful generator must reproduce | Two-level aggregates or latent-conditioned structure; real-structure sampling (E0-b) | Yes: multi-level coordination or structure–attribute coupling | H |
| F5 | Cross-relation structure uncorrelated | ρ(#trans, #order) 0.26 → 0.000; ρ(#order, #loan) 0.29 → 0.006; P(loan account has order) 1.00 → 0.85; identical in all configurations | VERIFIED-code and data | Independent per-relation degree sampling | — | Joint or latent-conditioned degree model (D3) | Yes: the structure generator | M–H |
| F6 | Off-support structure breaks the attribute model | 15.5% of synthetic loans sit on accounts with no orders, never seen in training; on those rows the loan column score is 0.46 (degree FT) and 0.41 (degree scratch) vs 0.91 (original) | REPORTED-prior (`java_d_degft/.../metrics/diagnosis_degree_shift.md`) | F5 produces out-of-support inputs for any structure-aware conditioning | — | Re-score with a structure-consistency filter | Couples D3 and D4 | M |
| F7 | Marginal and within-row fidelity preserved while relational dependence is lost | Column Shapes 97.0, intra-table 98.3; California within-row gaps 0.213 (floor 0.158) and 0.0066 (floor 0.0049); `trans_balance_last` KS 0.045 with its label link gone (HISTORICAL) | VERIFIED-data / prior | Denoiser heads model rows well; the failure is lateral | — | — | Evaluation must add lateral and group metrics | M |
| F8 | California: relational fidelity up, downstream flat or slightly down | Co-member rollups add +0.002 R² even with real data; household (1-hop) adds +0.051 real vs +0.030 synthetic, unchanged by aggregates (+0.028); aggregate FT worsens within-row conditionals (0.21 → 0.32; 0.0066 → 0.016) and shifts mean INCTOT by 0.21 (floor 0.011) | VERIFIED-data; ablation PRELIMINARY (1 seed) | Task mismatch, plus FT side effects on own-row conditionals | XGBoost seed noise for differences below ~0.003 | 3 × 3-seed ablation; California aggregate-from-scratch arm (E1) | The gained dependencies are not the ones the task uses | M |
| F9 | California household → member gap is partly categorical representation | η²(INCTOT; PUMA) 0.062 real vs 0.004 synthetic; E[INCTOT \| PUMA] gap 2.96 vs floor 0.52; synthetic PUMA lowers XGBoost R² (−0.010); dropping PUMA cuts the gap from 0.021 to 0.013 | VERIFIED-data (no-model part, 3 seeds); XGBoost PRELIMINARY | A 233-code nominal column diffused as a Gaussian ordinal: a one-code rounding error lands on an unrelated area | Household attributes otherwise mis-generated (the other 60% of the gap) | Embedding or discrete representation arm (E3) | Yes, representation (not relational) | M–H |
| F10 | Berka: fidelity and TSTR gains vanish once real data is present | Aggregate scratch TSTR loan 0.826 vs real+syn 0.936 (original 0.928) and blend_75 0.916 (0.924) | VERIFIED-prior | Data-processing bound: a generator fitted to the same real rows adds no information to them | Leakage inflates every absolute level equally | Leakage-free rerun (E2) | None. Do not use mixed regimes as the generator-quality criterion | L |
| F11 | No distance-graded loss in the paper's metrics | 1-/2-/3-HOP 91.5 / 95.6 / 92.7 (non-monotone); 3-HOP flat across configurations (92.6–93.1) | VERIFIED-prior and code | k-HOP scores only ancestor-top × descendant-bottom pairs; both 3-HOP and 5 of 7 2-HOP paths start at the fixed `district` | — | Lateral (sibling/cousin) metrics by distance | The loss is organised by **dependency type**, not hop count | M |
| F12 | Joint fine-tuning collapses at sampling | Rows with \|x\| > 6: loan 0.544, trans 0.674, account 0.559 (original 0.015 / 0.001 / 0.009); Column Shapes 84.5 | VERIFIED-data; cause partly HYPOTHESIS | High-t ε error × 31.6 first-step amplification, unclipped x̂₀, and spread through aggregates | Fresh optimiser or LR; normalisation | x̂₀-clipped resampling (E0-a) | Sampler and coupling robustness are prerequisites | H |
| F13 | Degree fine-tuning collapses without any aggregate pathway | Loan rows out of range 0.509, at-bound excess 0.505; it saturates on the **real** structure too (REPORTED-prior) | VERIFIED-data | Same high-t mechanism, with off-support degree inputs | — | E0-a on a degree-FT checkpoint | Aggregates amplify collapse; they are not its cause | H |
| F14 | Small tables are fragile and their marginals distorted | Loan default rate 0.065 (original), 0.208 (degree scratch), 0.139 (aggregate scratch) vs 0.111; loan is the first table to blow up in every unstable arm | VERIFIED-data | 2.6 seeds per batch; 259 samples at t = 1999; ordinal decoding of `status` (HISTORICAL) | — | Balanced seeding (HISTORICAL S3 helped loan) | Optimisation and representation | M |
| F15 | Group size invisible to members | California E[INCTOT \| age × size] gap 1.85 vs floor 0.14; aggregate FT 0.58 | VERIFIED-data | Sum then per-node LayerNorm removes magnitude; members never see the parent's degree | — | Degree-feature arm on California | Minor (fixable by count inputs) | L |

---

## 6. Conditional-information audit

**Method.** Every statistic is computed identically on:

* the real database, which for Berka is the full `data/berka/preprocessed` that every Berka generator
  was trained on, and for California the training split;
* every configuration × seed's existing synthetic database. For California the real **test** split
  supplies a real-vs-real noise floor.

Berka tables report mean ± population SD (ddof=0) and California tables report mean ± sample SD
(ddof=1), each dataset's established convention. All values come from
`results/stage1_limitations/outputs/summary_tables.md`, which `diagnostics/summarize.py` regenerates.

No held-out labels were used to train anything. The audit compares associations *inside* each
database, so it is not a utility estimate, and the Berka leakage caveat does not affect it.

### 6.1 Berka

**Structure (VERIFIED-data).** The structure is identical across all six configurations per seed
(digests `31c7add6…`, `5eda58bc…`, `99c53aac…`).

| dependency | real | every configuration |
|---|---|---|
| ρ(#trans, #order) per account | 0.260 | 0.000 ± 0.005 |
| ρ(#order, #loan) | 0.286 | 0.006 ± 0.022 |
| ρ(#trans, #loan) | 0.143 | −0.022 ± 0.006 |
| ρ(#disp, #card) | −0.030 | **0.144 ± 0.007** (spurious: cards drawn per disposition) |
| P(loan account has ≥ 1 order) | 1.000 | 0.847 ± 0.011 |
| AUROC(#trans → card holder) | 0.539 | 0.495 ± 0.008 |

**Sibling roles: the independence test (VERIFIED-data).** "Indep." is the share predicted if each
account's dispositions drew their type independently given the account's disposition count, using
that database's own rates.

| | real | original | degree scratch | degree FT | aggregate scratch | aggregate joint FT | aggregate frozen FT |
|---|---|---|---|---|---|---|---|
| accounts with exactly one OWNER | 1.000 | 0.727 ± 0.006 | 0.889 ± 0.005 | 0.874 ± 0.026 | **0.991 ± 0.004** | 0.764 ± 0.045 | 0.732 ± 0.005 |
| … independence prediction | 0.903 | 0.727 ± 0.005 | 0.895 ± 0.004 | 0.879 ± 0.023 | 0.897 ± 0.003 | 0.776 ± 0.036 | 0.731 ± 0.003 |

The original matches the independence prediction to 0.001. Degree conditioning improves the
marginal but stays independent. Only the from-scratch aggregate model creates genuine sibling
dependence.

**Sibling coherence (VERIFIED-data).** Values are in pooled rank space. ε² is the chance-corrected
between-account share of variance. "Residual" removes the part of the account means that the
account's own columns and district explain.

| | real ε² [95% CI of raw η²] | original | degree scratch | degree FT | aggregate scratch | aggregate joint FT | aggregate frozen FT |
|---|---|---|---|---|---|---|---|
| trans.balance ε² | 0.553 [0.542, 0.566] | **0.020 ± 0.001** | 0.239 ± 0.032 | 0.356 ± 0.125 | 0.695 ± 0.043 | 0.952 ± 0.008 | 0.202 ± 0.029 |
| trans.balance residual η² | 0.518 | **0.006 ± 0.001** | 0.167 ± 0.028 | 0.212 ± 0.041 | 0.630 ± 0.048 | 0.393 ± 0.116 | 0.119 ± 0.012 |
| trans.amount ε² | 0.095 [0.096, 0.102]* | 0.011 ± 0.001 | 0.099 ± 0.019 | 0.338 ± 0.100 | 0.160 ± 0.030 | 0.565 ± 0.225 | 0.037 ± 0.008 |
| order.amount ε² | 0.003 | 0.009 ± 0.012 | 0.142 ± 0.015 | 0.451 ± 0.091 | **0.218 ± 0.011** | 0.387 ± 0.034 | 0.007 ± 0.009 |

\* The CI is for raw η² (0.099). Real balances are 55% between-account variance. The original
explains 74% of its tiny between-account variance (0.020) by the account's two columns, leaving
residual coherence of 0.006. Aggregate scratch and both fine-tunes **overshoot**.

**Feature → label associations inside each database (VERIFIED-data).** AUROC, with the real 95% row
bootstrap. "Retained" is the mean of (synthetic − 0.5)/(real − 0.5) over real carriers
(|AUROC − 0.5| > 0.05).

| | real | original | degree scratch | degree FT | aggregate scratch | aggregate joint FT | aggregate frozen FT |
|---|---|---|---|---|---|---|---|
| loan default base rate | 0.111 | 0.065 | 0.208 | 0.355 | 0.139 | 0.237 | 0.078 |
| loan: `trans_bank_0_frac` | 0.690 [.625, .764] | 0.493 | 0.668 | 0.513 | 0.646 | 0.509 | 0.471 |
| loan: `trans_balance_last` | 0.284 [.212, .357] | 0.528 | 0.513 | 0.360 | 0.267 | 0.469 | 0.533 |
| loan: `order_k_symbol_3_frac` | 0.238 [.199, .280] | 0.551 | 0.400 | 0.489 | 0.372 | 0.469 | 0.549 |
| loan: retained, child aggregates (28 carriers) | — | 0.04 ± 0.13 | 0.16 ± 0.05 | −0.07 ± 0.23 | **0.74 ± 0.18** | 0.27 ± 0.15 | −0.08 ± 0.10 |
| loan: retained, own row / 1-hop (6) | — | 0.56 ± 0.16 | 0.81 ± 0.58 | −0.37 ± 0.48 | 1.04 ± 0.23 | −0.02 ± 0.22 | 0.58 ± 0.14 |
| card: `trans_balance_mean` | 0.838 [.826, .850] | 0.498 | 0.492 | 0.498 | 0.497 | 0.504 | 0.501 |
| card: retained, child aggregates (19 carriers) | — | 0.005 ± 0.031 | −0.037 ± 0.021 | −0.005 ± 0.043 | −0.000 ± 0.044 | 0.013 ± 0.008 | −0.009 ± 0.002 |
| card: retained, own row / 1-hop (2) | — | 0.54 ± 0.05 | 0.85 ± 0.03 | 0.47 ± 0.04 | 0.95 ± 0.08 | 0.51 ± 0.03 | 0.57 ± 0.08 |

**P(card | `trans_balance_mean` octile, real edges):** real 0.000, 0.001, 0.018, 0.051, 0.154, 0.277,
0.385, 0.443. Aggregate scratch 0.156 – 0.182, flat. Every other configuration is flat at ~0.16, apart
from the original's sparsely populated low octiles.

### 6.2 California

| dependency class | metric | real train | floor (real test) | original | control | aggregate FT | label-free aggregate FT |
|---|---|---|---|---|---|---|---|
| **within row** | INCTOT \| age band × EDUC, cell-mean gap ↓ | — | 0.158 | 0.213 ± 0.005 | 0.224 ± 0.017 | **0.325 ± 0.007** | 0.329 ± 0.018 |
| | EMPSTAT \| age band × SEX, TV ↓ | — | 0.0049 | 0.0066 ± 0.0003 | 0.0067 ± 0.0004 | **0.0162 ± 0.0007** | 0.0157 ± 0.0007 |
| **parent → child** | INCTOT \| VALUEH decile ↓ | — | 0.120 | 1.121 ± 0.043 | 1.091 ± 0.048 | 1.061 ± 0.032 | 1.077 ± 0.035 |
| | INCTOT \| OWNERSHP × ROOMS ↓ | — | 0.093 | 0.468 ± 0.027 | 0.468 ± 0.024 | 0.505 ± 0.039 | 0.507 ± 0.032 |
| | INCTOT \| PUMA ↓ (per seed) | — | 0.524 | 2.96 / 2.98 / 2.91 | — | 2.95 / 2.92 / 2.91 | — |
| | η²(INCTOT; PUMA) (per seed) | 0.062 | 0.062 | 0.004 / 0.004 / 0.004 | — | 0.006 / 0.006 / 0.006 | — |
| **group size** | INCTOT \| age band × household size ↓ | — | 0.144 | **1.846 ± 0.014** | 1.864 ± 0.017 | 0.578 ± 0.041 | 0.581 ± 0.041 |
| **siblings** | AGE ε² | 0.291 | 0.295 | **0.047 ± 0.001** | 0.046 ± 0.001 | 0.273 ± 0.008 | 0.263 ± 0.011 |
| | EDUC ε² | 0.238 | 0.241 | 0.037 ± 0.001 | 0.035 ± 0.000 | 0.179 ± 0.008 | 0.179 ± 0.006 |
| | INCTOT ε² | 0.076 | 0.077 | 0.024 ± 0.000 | 0.022 ± 0.001 | **0.113 ± 0.004** (over) | 0.119 ± 0.004 |
| | head–spouse AGE corr | 0.928 | 0.928 | **0.172 ± 0.008** | 0.175 ± 0.008 | 0.612 ± 0.017 | 0.595 ± 0.039 |
| | head–spouse EDUC corr | 0.681 | 0.685 | 0.078 ± 0.005 | 0.082 ± 0.003 | 0.286 ± 0.015 | 0.275 ± 0.013 |
| **roles** | households with exactly one head | 0.933 | 0.935 | **0.377 ± 0.001** | 0.376 ± 0.000 | 0.696 ± 0.015 | 0.690 ± 0.015 |
| | … independence prediction | 0.541 | 0.544 | **0.376 ± 0.000** | 0.376 ± 0.001 | 0.541 ± 0.001 | 0.540 ± 0.001 |
| | households with no head | 0.067 | 0.065 | 0.397 ± 0.001 | 0.397 ± 0.001 | 0.195 ± 0.006 | 0.199 ± 0.007 |

The head–spouse correlations here use households with exactly one head and one spouse. They differ
slightly from `java_d-calagg-ft/REPORT.md` (0.925 / 0.154), which selected pairs differently.

**Relationship importance for the downstream tasks (PRELIMINARY: one generator seed, one XGBoost seed,
300k training rows, real test split).** Harness feature builder and XGBoost settings unchanged.

| trained on | INCTOT R²: own | + household | + co-member rollups | EMPSTAT F1: own | + household | + co-member rollups |
|---|---|---|---|---|---|---|
| real | 0.482 | 0.533 (+0.051) | 0.535 (+0.002) | 0.678 | 0.684 (+0.006) | 0.684 |
| original, seed 0 | 0.479 | 0.508 (+0.030) | 0.507 | 0.675 | 0.674 | 0.673 |
| aggregate FT, seed 0 | 0.478 | 0.506 (+0.028) | 0.507 | 0.673 | 0.672 | 0.672 |

| trained on | own | own + PUMA | own + household without PUMA | own + household |
|---|---|---|---|---|
| real | 0.482 | 0.493 | **0.539** | 0.533 |
| original, seed 0 | 0.479 | **0.469** | 0.522 | 0.508 |
| aggregate FT, seed 0 | 0.478 | 0.469 | 0.521 | 0.506 |

### 6.3 Audit verdicts

| relationship | real | synthetic (original) | discrepancy vs floor | uncertainty | task affected | status |
|---|---|---|---|---|---|---|
| within-row (California income and employment given age, education, sex) | — | gap 0.213 / TV 0.0066 | 1.3–1.4× floor | ddof=1 SD ≤ 0.005 | both California tasks | **well preserved** (degraded by aggregate FT: 2–3× floor) |
| marginals (Berka Column Shapes 97.0; California 98.6) | — | — | small | — | — | well preserved |
| parent → child, numeric household attributes → income | — | 1.12 (VALUEH) | 9× floor | ± 0.04 | California income | **partially preserved** (60% of household R² increment) |
| parent → child, nominal PUMA → income | η² 0.062 | 0.004 | 5.7× floor | per-seed range 0.0039–0.0043 | California income | **poorly preserved** (representation) |
| group size → member | — | 1.85 | 13× floor | ± 0.014 | California (`hh_size` is a feature) | **poorly preserved**; fixed by counts (0.58) |
| siblings, group factor (ε², co-member correlation) | 0.29 / 0.55 | 0.05 / 0.02 | 5–25× below | CI ± 0.01 | Berka loan and card aggregates; California rollups | **poorly preserved** |
| siblings, roles (one head, one owner) | 0.93 / 1.00 | independent draws | — | ± 0.001 | role-based features | **absent** (independence) |
| pairwise roles (head–spouse) | 0.93 | 0.17 | — | ± 0.01 | — | **absent**; partial with aggregates (0.61) |
| child aggregate → label (loan) | AUROC 0.24–0.69 | ~0.5, 14/28 reversed | — | bootstrap CI ± 0.07 (76 positives) | loan default | **absent** at baseline; restored by aggregate scratch |
| child aggregate → structural label (card) | 0.838 | 0.498 | — | CI ± 0.012 | card ownership | **absent in every configuration** |
| cross-relation structure | ρ 0.14–0.29 | 0.00 | — | ± 0.02 | loan (`order_n`, `trans_n`) | **absent by construction** |
| California 11-regime utility for 3 new arms | — | — | — | — | — | **unresolved** (Table 5 incomplete) |

---

## 7. Sampling-collapse investigation

**What collapsed (VERIFIED-data, model-space rows with any |x| > 6; the quantile transform bounds real data at ±5.2):**

| table | original | degree scratch | degree FT | aggregate scratch | aggregate joint FT | aggregate frozen FT |
|---|---|---|---|---|---|---|
| loan | 0.015 ± 0.003 | 0.162 ± 0.012 | **0.509 ± 0.015** | 0.042 ± 0.005 | **0.544 ± 0.083** | 0.016 ± 0.005 |
| account | 0.009 | 0.037 | 0.218 | 0.017 | **0.559** | 0.011 |
| order | 0.007 | 0.016 | 0.262 | 0.009 | 0.332 | 0.008 |
| trans | 0.001 | 0.028 | 0.144 | 0.015 | **0.674** | 0.003 |
| client, card | ≤ 0.002 | ≤ 0.012 | ≤ 0.004 | ≤ 0.006 | ≤ 0.003 | ≤ 0.001 |

The inverse quantile transform parks these values at the column's real min or max. The excess share
of loan rows at a bound is 0.505 (degree FT) and 0.516 (aggregate joint FT), against 0.033 (original).

**Runaway trajectories exist in every model, including the original (VERIFIED-data).** The largest
|x| in the final model-space sample, against a real bound of 5.2, per seed 0 / 1 / 2:

| configuration | client | account | loan |
|---|---|---|---|
| original | 39,668 / 2,077 / 5 | 21,928 / 4,177 / 16,095 | 5,702 / 7,951 / 2,299 |
| degree FT | — | — | 56,284 / 63,343 / 46,345 |
| aggregate joint FT | — | — | 37,181 / 47,792 / 37,734 |

Values this far outside the data range cannot arise from a well-behaved denoiser. They are
consistent with an ε error multiplied ×31.6 at the first step and then never pulled back, since x̂₀
is unclipped. Fine-tuning changes how *many* rows run away (1.5% → 51–54% of loans), not *whether*
the mechanism exists. Decoding clips all of them silently, which is why the original's fidelity
scores do not show it.

**Candidate causes, as asked in the brief:**

| candidate | evidence | verdict |
|---|---|---|
| numerical instability in training | no non-finite losses; gradient norms bounded (aggregate FT: mean 5.8–6.1, max 102–181) (REPORTED-prior) | **not supported** |
| distributional collapse at sampling | yes: the table above | **verified** (symptom) |
| first-step amplification | ∂x₁₉₉₈/∂ε̂ = 31.6 at t = 1999 (VERIFIED-code); FT ε error at t ≥ 1990 is 3–4× the original's (0.033–0.040 vs 0.010) (REPORTED-prior); x̂₀ is never clipped (VERIFIED-code) | **verified mechanism**; its sufficiency is untested (E0-a) |
| under-trained high-t behaviour of small tables | loan: 259 training samples at t = 1999 in 200k steps vs 400,766 for trans (VERIFIED); loan is the first table to blow up in every unstable arm (VERIFIED-data, plus the aggft trace: loan at t = 1999 3.5% original, 19.6% scratch, 48.9% FT) | **supported** contributing cause |
| excessive conditioning / aggregate saturation | 10.6–16.8% of accounts sit at the ±10 clip at the end; zeroing aggregates during sampling cuts trans blow-up from ~72% to ~11% (REPORTED-prior, `checks/sampling_trace.py`) | **verified amplifier** (contagion through ~235 children per account) |
| incorrect sampling-time conditioning / code mismatch | teacher-forced on the real graph through the sampler's own path, FT models beat the original at every t from 10 to 1600 (REPORTED-prior, `checks/sampling_path_probe.py`); train path equals sample path at identity noise | **not supported** |
| train–sample mismatch (exposure bias) | at sampling, aggregates and degree inputs come from generated (possibly blown) children or from off-support structure (15.5% of loans without orders) | **supported** for both conditioning types (HYPOTHESIS for size of effect) |
| normalisation of training data | the same normalisers are used everywhere (byte-checked) | **not supported** |
| loss imbalance | per-table means summed; loan's term rests on ~2.6 samples and is absent from 7.5% of batches | **contributing** (high-variance gradient for small heads) |
| fine-tuning specifics | fresh AdamW, lr 1e-4 (aggregates) or 3e-5 (degree) on a converged model; the step-5,280 frozen snapshot is stable and +20 joint epochs is already unstable (REPORTED-prior) | **supported trigger**; collapse appears early in joint training |

**Interpretation.** Collapse needs three ingredients:

1. an amplifier that every GRDM model has: ε-parameterisation, β = 0.999 at t = 1999, and unclipped x̂₀;
2. a trigger: any change to the high-noise behaviour of the small-table heads, which training barely
   constrains (fine-tuning either conditioning type, or off-support inputs);
3. optionally, a spreader: the aggregate feedback loop.

The original already has 1–2% of loan rows out of range, which decoding hides. California did not
collapse. Plausibly that is because both of its tables are large (554,503 households; 1,521,638 individuals) and households have
≤ 25 members (HYPOTHESIS). This diagnosis is **mostly about sampling and optimisation, not about
representational capacity**. Any new coupling mechanism will hit the same amplifier, so fixing it is a
precondition for Stage 2.

---

## 8. Scratch versus fine-tuning: confound analysis

| factor | aggregate scratch (Berka) | aggregate joint FT | aggregate frozen FT | degree scratch | degree FT | California aggregate FT | California control |
|---|---|---|---|---|---|---|---|
| initialisation | random | original EMA | original EMA | random | original EMA | original EMA | original EMA |
| steps with the new pathway | 200,000 | 44,880 (5,280 frozen) | 5,280 | 200,000 | 21,120 | 25,350 | — (20,280 steps, no aggregates) |
| LR schedule | 6e-4 → 0 linear | 6e-4 (new layers only) then 1e-4 peak + warmup + decay | 6e-4, new layers only | 6e-4 → 0 | 3e-5 constant, then 3e-5 → 0 | 6e-4 then 1e-4 | 1e-4 |
| optimiser state | fresh | fresh | fresh | fresh | fresh | fresh | fresh |
| training RNG seed | N | 1000+N | 1000+N | N | 1000+N (sampling 2000+N) | per experiment | per experiment |
| objective | ε-MSE | ε-MSE | ε-MSE | ε-MSE | ε-MSE | ε-MSE | ε-MSE |
| data and normalisers | full Berka | same, copied | same | same | same | California train | same |
| conditioning | 31 statistics | 31 (27 label-free) | 31 | 30 degree columns | 30 | 20 (16) | none |
| sampling | same sampler; same structure per seed | same | same | same | same | same | same |
| evaluation | same harness, re-scored | same | same | same | same | own harness | same |
| sampling health | stable | **collapsed** | stable | loan 16% out of range | **collapsed** | stable | stable |

**Conclusion (VERIFIED-code and data).** Existing experiments **cannot isolate** "training from
scratch". At least six factors change together: pathway training length (4.5–9.5× shorter), LR and
schedule, frozen phases, initialisation, RNG seed, and, for the joint arms, a sampling failure that
corrupts every downstream and fidelity comparison.

The only clean contrasts are:

* **Berka, aggregate scratch vs original:** the architecture effect at matched data, objective and
  budget.
* **California, aggregate FT vs control:** the aggregate effect at a matched fine-tuning budget.

There is no California scratch arm and no Berka fine-tune control (continued training without the
pathway).

The zeroed-aggregate probe (§4.4) shows the fine-tuned models barely use the new pathway. That
supports "this recipe did not learn the pathway", not "fine-tuning cannot".

---

## 9. Architectural versus non-architectural explanations

Category key: 1 representation · 2 information flow · 3 objective · 4 optimisation · 5 sampling ·
6 data · 7 evaluation · 8 task mismatch · 9 confounding. ● primary, ◐ contributing, ○ not supported,
? untested.

| failure | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| F1/F2 sibling independence; aggregate → label loss | ◐ no group state | ● | ◐ no group-level term | ○ control +20% steps unchanged | ? | ○ ample data | ◐ invisible to k-HOP | — | ○ |
| F3 over-coupling under bottom-up aggregates | ● uncalibrated feedback design | — | ◐ no consistency term | ? | ◐ exposure bias | — | — | — | — |
| F4 card not recovered | ◐ | ● 3 hops; no 2-level summaries | — | ? | — | ◐ dependency probe | — | — | — |
| F5/F6 structure decoupling | ● independent degree model | — | — | — | — | — | ◐ Cardinality blind to it | — | — |
| F8/F9 California fidelity ↑, utility ↓ | ● PUMA ordinal | ○ | — | ◐ FT side effect | — | — | ● wrong dependency measured | ● | ◐ FT vs scratch |
| F12/F13 collapse | ○ | — | — | ● trigger | ● amplifier | — | — | — | ◐ |
| F11 no distance gradient | — | — | — | — | — | — | ● | — | — |

**Diagnosis details**

**D-1. Missing group-level coordination: information flow (primary) and representation.**

* **Evidence:** §2.2 (no within-step path between siblings; nothing persists across steps except the
  observed columns); §6 independence test (observed = predicted to 0.001 on both datasets); the
  matched-budget intervention breaks it (OWNER 0.991 vs prediction 0.897; fixed-batch loss
  −27% on trans/loan, REPORTED-prior).
* **Competing explanations:** under-training (contradicted by the California control); the categorical
  representation (cannot produce independence); that reverse dynamics are adequate but slow to mix
  (would show as partial dependence, which is not observed).
* **Falsified if:** a K = 1 GRDM trained 2–4× longer, or with balanced seeding, shows exactly-one-head
  or exactly-one-OWNER clearly above its independence prediction. A K = 2 GRDM (siblings visible within
  a step) should break independence; if it does not, the information-flow account is wrong.
* **Smallest test:** California K = 2, one seed, 100k steps (siblings ≤ 25, cheap); or score
  intermediate checkpoints if any exist. **Reuses checkpoints:** no. **New training:** yes, one run.

**D-2. Structure decoupling: representation of the generative factorisation.**

* **Evidence:** VERIFIED-code (independent histograms, random matching) and VERIFIED-data
  (correlations zero, digest-identical).
* **Uncertain:** how much utility it costs.
* **Competing explanation:** card and loan losses are fully explained by D-1 and the window.
* **Falsified if:** sampling the existing checkpoints on the **real** structure leaves loan and card
  TSTR unchanged (diagnostic only, since the real structure contains test-account degrees).
* **Smallest test:** E0-b. **Reuses checkpoints:** yes. **New training:** no. **Sampling:** yes.

**D-3. Sampling fragility: sampling (amplifier) and optimisation (trigger).**

* **Falsified if:** re-sampling the collapsed checkpoints with x̂₀ clipped to the training range
  (|x̂₀| ≤ 5.2), or with the t = 1999 step replaced by a deterministic x̂₀ step, leaves out-of-range
  rows above 20%.
* **Smallest test:** E0-a. **Reuses checkpoints:** yes. **New training:** no.

**D-4. Nominal categoricals as Gaussian ordinals: representation.**

* **Evidence:** PUMA (VERIFIED-data); `loan.status` class B (HISTORICAL).
* **Competing explanation:** a household-level generation error unrelated to coding.
* **Falsified if:** a PUMA-embedding (or analog-bits) arm leaves η²(INCTOT; PUMA) near 0.005.
* **Smallest test:** E3, California, one seed. **New training:** yes.

**D-5. Evaluation blind spots: evaluation.** VERIFIED-code. Nothing to falsify. Fix by adding the §6
metrics to the standard protocol.

**D-6. California utility, task mismatch.**

* **Evidence:** PRELIMINARY ablation.
* **Falsified if:** at 3 × 3 seeds the co-member rollups add more than 0.005 R² on real data.
* **Smallest test:** E0-e (CPU, about one hour). **New training:** no.

---

## 10. Ranked list of credible limitations

| rank | limitation | type | strength of evidence | what would confirm or reject it |
|---|---|---|---|---|
| 1 | **No group-level (entity) state.** Siblings are conditionally independent given the parent's observed columns, so group factors, roles and child aggregates are lost | architectural (information flow and representation) | **demonstrated**: code + model-free independence test on two datasets + matched-budget intervention | Confirm: a sibling-aware arm restores the role and coherence metrics without overshoot (E1). Reject: a longer or deeper K = 1 run breaks independence |
| 2 | **Structure generated independently of attributes and of other relations** | architectural (factorisation) | demonstrated for the structure itself; its utility share is a HYPOTHESIS | real-structure sampling (E0-b) |
| 3 | **Sampling fragility**: first-step ×31.6, unclipped x̂₀, small tables untrained at high t, feedback spreading | sampling and optimisation (not representational) | mechanism verified; sufficiency is a HYPOTHESIS | x̂₀-clipped resampling (E0-a) |
| 4 | **Nominal categoricals as Gaussian ordinals** (PUMA, `loan.status`) | representation | verified on one high-impact column; utility effect PRELIMINARY | embedding or discrete arm (E3) |
| 5 | **Bottom-up aggregate coupling over-couples and destabilises** | design of the existing fix | verified (ε² overshoot; collapse spread) | trace ε² over reverse steps; top-down vs bottom-up comparison (E1) |
| 6 | **Evaluation measures the wrong dependencies** (vertical pairs; fixed `district`; Cardinality structure-only) | evaluation | verified in code | — |

---

## 11. Proposed new architectural directions

**Novelty.** The GRDM paper PDF and `docs/references/` were on `/scratch` and are unavailable. The
current workspace contains no related-work notes, and no literature search was done in this stage, so
**no novelty is claimed**.

Known related ideas that overlap and must be checked first (from memory; verify against the papers):

* **SDV's hierarchical modelling (HMA):** augments parent rows with parameters of their children's
  distributions.
* **ClavaDDPM:** coordinates parent–child generation through latent cluster variables. In this
  project's historical comparison it carried roughly twice GRDM's card signal and six times its loan
  signal.
* **Autoregressive and sequential multi-table synthesisers.**

Directions D1–D3 below overlap conceptually with these.

### D1. Generated entity latent (top-down group state). Addresses F1, F2, F3; partly F4 and F9

1. **Failure mode:** sibling independence (rank 1); over-coupling of the bottom-up fix (rank 5).
2. **Replaces or changes:** adds `d_z` latent channels to every parent row, diffused with the parent's
   attributes. Children condition on them through the existing message passing.
3. **Mechanism.** The latent's clean target is a summary of the parent's full clean child set: start
   with the existing standardised statistics, then a learned permutation-invariant encoder. It is
   generated **as part of the joint**, not recomputed each step from the children's noisy state. A
   consistency term ties the latent to the realised children's summary at x̂₀ level.
4. **More than a conditioning tweak because** it changes the generative factorisation. Each entity
   gets its own random state, sampled jointly, from which its children are generated: a hierarchical
   latent-variable model. The bottom-up fix only lets the parent read its children's current noise.
5. **Expected benefit:** restores group factors (ε², co-member correlation, aggregate → label) with
   calibrated strength, and gives grandchild information a channel (disp latent → account latent →
   transactions).
6. **Cost:** small. About d_z extra columns per parent table plus a set encoder; comparable to
   aggregate conditioning (+22% step time).
7. **Risks:** the latent ignored by children, or collapsed by the encoder; latent and realised
   children inconsistent; still no exclusivity (one owner, one head) because children remain
   independent given the latent (hence D2).
8. **Differs from the existing implementation:** the latent is generated rather than computed from
   `x_t`; it has an explicit consistency objective; it can be learned rather than hand-picked.
9. **Minimum experiment:** E1 arm C on California.
10. **Baselines and metrics:** original, bottom-up aggregates from scratch, D1+D2. Metrics: §6 role,
    coherence and over-coupling tests; household → member R² increment; 11-regime utility; fidelity
    suite; sampling health.

### D2. Within-group set interaction (sibling attention). Addresses F1 roles and pairwise structure

1. **Failure mode:** exactly-one-head 0.377 and head–spouse 0.17 (real 0.933 and 0.93); aggregates
   reach only 0.70 and 0.61.
2. **Changes:** adds one attention block over each parent's child set (with the parent as an extra
   token) after the GNN, at training and sampling.
3. **Mechanism:** within one denoising step, each child sees the others' states, which allows mutual
   exclusion and role-specific pairing. A permutation-invariant summary cannot single out a pair.
4. **Beyond a tweak:** it replaces K-hop sum aggregation for the sibling relation with explicit
   pairwise interaction.
5. **Benefit:** roles, pairs, and Berka one-owner.
6. **Cost:** O(n²) per group. California has ≤ 25 members (trivial). Berka has ~235 transactions per
   account (55k pairs per account; feasible, but batches must hold whole child sets). Training needs
   parent-seeded batches.
7. **Risks:** a different batch composition confounds the comparison; memory on Berka; over-coupling
   (the same risk as F3).
8. **Differs:** GRDM has no sibling path at K = 1, and at K = 2 only a capped (16) pooled sample.
9. **Minimum experiment:** E1 arm D.
10. **Metrics:** as D1, plus exactly-one-head and exactly-one-OWNER against their independence
    predictions, and head–spouse correlations.

### D3. Latent-conditioned structure generation. Addresses F4, F5, F6

1. **Failure mode:** independent per-relation degrees; structure-defined labels decoupled from
   attributes.
2. **Replaces:** `RandomGraphSampler`'s independent histogram draws.
3. **Mechanism:** generate each parent's child counts per relation **conditioned on the parent's
   attributes and D1 latent**, e.g. a discrete diffusion or count model per relation that reads the
   parent latent. Structure and entity state are generated together, before or interleaved with
   child attributes.
4. **Beyond a tweak:** it changes what the model generates, not only what it conditions on.
5. **Benefit:** cross-relation degree correlations; card existence that co-varies with account
   behaviour; no off-support structures.
6. **Cost:** moderate. The count models are small, but the sampling order changes.
7. **Risks:** coupling structure and attributes can make errors compound; harder to evaluate; may need
   a latent that exists before children.
8. **Differs:** today's structure is attribute-agnostic and relation-independent by construction.
9. **Minimum experiment:** E0-b first (is structure worth it?), then an E2 arm.
10. **Metrics:** §6.1 structure table, card dependency, loan `order_n` and `trans_n` associations,
    off-support rate.

### D4. Robust sampler and calibrated coupling. Addresses F12, F13, F3 (a precondition, not the novelty)

1. **Failure mode:** collapse, plus over-coupling under feedback.
2. **Changes:** the reverse-step parameterisation and the conditioning inputs.
3. **Mechanism:** x̂₀ clipping or thresholding to the per-column training range; v- or x₀-prediction;
   handling the β = 0.999 first step; small-table coverage of high t (balanced seeding or
   t-stratification); bounded, robust inputs for any coupling pathway.
4. **Beyond a tweak?** **No.** These are engineering fixes, required so that D1–D3 can be evaluated.
5. **Benefit:** stable sampling for any variant; a fair fine-tuning comparison.
6. **Cost:** negligible.
7. **Risks:** clipping can mask model errors and must be reported with out-of-range rates.
8. **Differs:** today x̂₀ is unclipped and `clip_denoised` is inert.
9. **Minimum experiment:** E0-a.
10. **Metrics:** out-of-range rows, at-bound excess, Column Shapes, plus all §6 metrics, unchanged.

### D5. Representation for nominal categoricals. Addresses F9, F14

Learned embeddings or analog bits for high-cardinality nominal columns (PUMA), and a properly scaled
discrete process for small categorical label columns. The historical one-hot and multinomial attempts
failed for implementation reasons (California R² −4.97; NaN loss) and do not settle the question.

* **Minimum experiment:** E3 (California, PUMA as an embedding, one seed).
* **Metrics:** η²(INCTOT; PUMA), E[INCTOT | PUMA] gap, own + PUMA R² increment, TSTR.

**Not proposed:**

* more conditioning vectors on GRDM's existing factorisation;
* longer or deeper K-hop GNNs as the main line. Two hops helped card historically, at 2× cost and
  about −2 Column Shapes; three hops did not help;
* combining D1–D5 into one architecture before each is tested.

---

## 12. Minimum experiments needed to validate the diagnoses

| id | question | design | reuses checkpoints | new training | cost (contended node) |
|---|---|---|---|---|---|
| **E0-a** | Is collapse driven by first-step amplification and unclipped x̂₀? | Resample aggregate-joint-FT seed 0, degree-FT seed 0 and original seed 0 with x̂₀ clipped to the training range, on the same structure. Measure out-of-range rows, Column Shapes and §6 metrics | yes | no (a sampling flag in a new worktree) | 3 sampling runs, ~2–4 GPU-h each, ~1.5 GB |
| **E0-b** | How much do loan and card lose to the structure sampler? | Sample the original and aggregate-scratch seed 0 on the **real** structure (`--fixed_structure`); downstream TSTR and §6.1. Diagnostic only, because the real structure includes test-account degrees | yes | no | 2 sampling runs + CPU scoring |
| **E0-c** | Complete Table 5 | rerun the §0 launcher | existing synthetic data | no | ~10 h CPU (needs your go-ahead; you asked not to restart it) |
| **E0-d** | Leakage-free Berka split | `python -m downstream.make_berka_split` | — | no | minutes |
| **E0-e** | Firm up the California relationship-importance and PUMA results | rerun `california_relationship_importance.py` and `california_household_gap.py` over 3 generator × 3 XGBoost seeds | existing synthetic data | no | ~1 h CPU |
| **E0-f** | When does bottom-up over-coupling appear? | Trace ε² of trans.balance along the reverse chain for aggregate scratch seed 0 (adapt `java_d-aggft/.../checks/sampling_trace.py`) | yes | no | ~2 GPU-h |
| **E1** | Does explicit group state fix sibling dependence, calibrated, from scratch? | §14 | baseline reused | yes | ~9 runs (§13) |
| **E2** | Does it carry to Berka utility, leakage-free? | §14 | — | yes | ~12 runs |
| **E3** | Is PUMA representation a separate fix? | California, PUMA embedding, 1 seed then 3 | — | yes | 1–3 runs |

---

## 13. Resource estimates for the next stage

Based on observed wall times in this project's logs (contended 8 × H200 node, shared with other users).

| unit | GPU memory | wall time | source |
|---|---|---|---|
| Berka GRDM, 200k steps + sampling | ~1.3–1.5 GB | 13–14 h + 1–1.5 h on a lightly loaded GPU; **19–38 h** contended | HISTORICAL; aggregate scratch logs (24.1 / 19.2 / 37.6 h); degree scratch (30.0 / 28.3 / 21.1 h training) |
| California GRDM, 100k steps + sampling | ~1.5 GB | sampling ≈ 8 s per reverse step ≈ 4.4 h; 25k fine-tune steps + sampling took 5.3 h → about **8–24 h per scratch run** (estimate) | `java_d-calagg-ft/.../logs/train_agg_seed0.log` |
| aggregate pathway overhead | +0.3% parameters | +22% step time (Berka) | `java_d-agg/.../architecture_notes.md` |
| two-hop depth | — | about 2× step time | HISTORICAL |
| sibling attention (D2) | California small; Berka needs whole child sets per batch (~1M nodes if naive) | **unknown**; budget +50–100% | estimate |
| California 11-regime downstream | CPU only | ~5 h CPU per arm-seed at 2 threads | baseline logs |
| Berka downstream (2 tasks, all regimes) | CPU only | minutes per arm | harness |

**Stage 2 budget (estimates):**

* **E0 block:** ~10–15 GPU-h, plus ~12 h CPU (including Table 5).
* **E1:** 3 new arms × 3 seeds = 9 runs, ~100–250 GPU-h; ~1.5–2 days wall time if run 8–9 at once.
* **E2:** 4 arms × 3 seeds = 12 runs, ~250–450 GPU-h; ~2–3 days wall time.
* **Downstream scoring:** ~50–100 CPU-h in total.

---

## 14. Recommended Stage 2 research plan

**Phase 0: no training (1–2 days, mostly parallel).**

* Run E0-a, E0-b, E0-d, E0-e and E0-f.
* Run E0-c only if you approve.
* **Gate:** adopt D4 (the sampler fix) for every Stage 2 arm only if E0-a shows it removes collapse
  without changing the original's metrics by more than seed noise. Otherwise investigate the
  aggregate feedback first.

**Phase 1: E1, the California coordination test (first training experiment).**

* **Arms**, all from scratch with the original California schedule (100k steps, lr 6e-4 → 0, batch
  4096, same normalisers, same per-seed structure, seeds 0–2), sampled with the same sampler:
  * **A.** original GRDM (existing `paper_repro`);
  * **B.** bottom-up aggregate conditioning from scratch (existing module, 20 statistics). Fills the
    missing California scratch cell and removes the FT/scratch confound;
  * **C.** D1 entity latent;
  * **D.** D1 + D2 sibling attention.
* **Ablations:** C with the consistency term off; C with a learned vs fixed summary. Run a single seed
  first and expand only if informative.
* **Pre-registered predictions:**
  * C restores coherence (ε², co-member correlation) but **not** roles (exactly-one-head stays near its
    independence prediction);
  * D restores roles;
  * B over-couples INCTOT (ε² > 0.096 = real + 0.02).
  * The PUMA dependency is unchanged in every arm (negative control).
* **Data protocol:** California train/test as established (leakage-free); 11-regime downstream with
  3 XGBoost seeds; report per seed, mean and ddof=1 SD; paired-by-seed differences with sign counts.
  No significance claim without a paired test across ≥ 3 seeds, plus a bootstrap over the 169k-row
  test set (enable it in the harness for this purpose).

**Phase 2: E2, Berka leakage-free (only if E1 meets its mechanism criteria).**

* Retrain on the regenerated `split_train`: original, B, the better of C or D, and that arm + D3 if
  E0-b showed structure matters.
* 3 seeds; report ddof=0 to match the Berka tables.
* Loan has 19 positives, so add seeds 3–4 or report test-set bootstrap CIs before any loan claim.
* Keep every existing leaky result in a separate table, labelled leaky.

**Phase 3:** E3 representation track (D5), in parallel and independently.

**Controls in every phase:** identical splits, preprocessing, harness version, XGBoost settings,
regimes, sample size (TSTR matched to the real train size) and seed handling; matched steps and
schedule across arms; any unavoidable difference (e.g. D2's parent-seeded batches) documented; Stage 1
diagnostics run on every arm (`results/stage1_limitations/diagnostics/*.py` take a config dict).

---

## 15. Limitations and unresolved questions

* **Stage 1 is observational.** The architectural diagnosis rests on code, a model-free independence
  test, and one matched-budget intervention per dataset family. No new intervention was run.
* **California ablations are PRELIMINARY.** They use one generator seed, one XGBoost seed and 300k
  rows; differences below ~0.003 R² are within noise.
* **Berka results carry the leakage caveat.** Every Berka generator saw the test accounts, so absolute
  utility is optimistic. The within-database audit does not depend on the split.
* **Loan default is noisy.** It has 76 positives in the full data and 19 in the test set.
* **Card ownership is a dependency probe, not a forecast.** Part of the real balance ↔ card
  association arises after the card is issued.
* **The independence test checks one statistic per dataset** (one head, one owner). Other forms of
  sibling dependence could exist below its resolution, but the ε² and correlation measures agree.
* **The causes of collapse are not established by intervention.** E0-a would test them.
* **Unresolved:**
  * why degree-scratch overproduces loan defaults (0.208 vs 0.111);
  * the other ~60% of California's household → member gap after PUMA;
  * whether D2 is tractable on Berka's ~235-child sets;
  * whether ClavaDDPM-style latent clustering already captures most of D1. Historical evidence
    suggests it carries more cross-table signal; this needs a current re-measurement.
* **References are unavailable** (paper PDF, AmEx report), so no comparison to the paper's own claims
  about multi-hop modelling beyond what the code shows.
* **Table 5 is incomplete** (§0).

---

## Final decision

**A. The three most credible limitations of the current GRDM**

1. No group-level or entity state. Rows sharing a parent are generated as conditionally independent
   given the parent's observed columns.
2. Graph structure is generated independently of attributes and of other relations.
3. A fragile reverse process: first-step error amplification of 31.6×, unclipped x̂₀, and small-table
   heads barely trained at high noise. This turns any perturbation (fine-tuning, new conditioning
   inputs) into collapse.

A close fourth: nominal categorical columns diffused as Gaussian ordinals (PUMA).

**B. Which are demonstrably architectural and which remain hypotheses**

* **(1) is demonstrably architectural** (information flow). The code shows no sibling path and no
  group state; the data match the independence prediction to 0.001 on two databases; and a change to
  information flow alone, at matched data, objective and budget, breaks it (one OWNER 0.991 vs
  prediction 0.897).
* **(2) is architectural by construction** (VERIFIED-code and data). Its *share of the utility loss*
  is a hypothesis.
* **(3) is not a representational limitation.** It is a sampling and optimisation property. The
  amplifier is verified; that it is *sufficient* to explain the collapse is a hypothesis.

**C. Strongest evidence for each**

* **(1)** Exactly-one-head: 0.377 observed vs 0.376 predicted under independence (real 0.933).
  Exactly-one-OWNER: 0.727 vs 0.727 (real 1.000). Transaction-balance between-account share: 0.020
  vs 0.553.
* **(2)** ρ(#order, #loan) 0.29 → 0.01 and P(loan account has an order) 1.00 → 0.85, identical in
  every configuration. The card → balance link is ~0 in all six configurations.
* **(3)** ∂x/∂ε̂ = 31.6 at t = 1999; loan gets 259 training samples there. Out-of-range loan rows:
  51% (degree FT, no aggregate pathway) and 54% (aggregate FT), against 1.5% for the original.

**D. Smallest experiment that could falsify each**

* **(1)** California GRDM with K = 2 (or the existing California control trained 2–4× longer). If
  K = 1 with more training breaks independence, or K = 2 does not, the diagnosis fails. One run, one
  seed.
* **(2)** E0-b: sample existing checkpoints on the real structure. If loan and card associations and
  TSTR do not move, structure is not a material cause. No training.
* **(3)** E0-a: resample the collapsed checkpoints with x̂₀ clipped to the training range. If
  out-of-range rows stay above 20%, amplification is not the main mechanism. No training.

**E. Most justified new capability**

Explicit, *generated* group-level state per entity, combined with within-group interaction, and
calibrated: it must restore sibling coherence and role constraints without the over-coupling that the
bottom-up aggregate fix shows (ε² 0.695 vs 0.553; orders 0.218 vs 0.003).

**F. Modify or redevelop?**

**Develop a substantially new coordination mechanism and generative factorisation, built on GRDM's
diffusion and data infrastructure, rather than adding conditioning vectors.** The reasons:

* The core deficit is in *what the model can represent jointly*, not in what it is conditioned on.
* The best existing patch (bottom-up aggregates) shows the deficit is real and repairable, but it
  over-couples, spreads sampling failures, and cannot reach structure-defined or grandchild
  dependencies (card ~0 in every configuration).
* Structure is generated independently and first, which no attribute-side conditioning can change.
* Four core components are implicated: denoiser information flow, structure generator, categorical
  representation and sampler parameterisation. That meets the brief's own threshold for a redesign.

Build it stepwise, with each component falsifiable on its own (D1 → D2 → D3, with D4 as a precondition
and D5 separate), so that every claimed gain is attributable.

**G. First Stage 2 experiment**

Start with the zero-training E0 checks (sampler clipping, real-structure sampling). The first training
experiment is **E1, the California coordination test**: original vs bottom-up aggregates from scratch
vs entity latent (D1) vs D1 + sibling attention (D2), all from scratch at the original 100k-step
schedule, 3 seeds each, with pre-registered mechanism predictions (§14).

**H. Resources required**

* **E0:** ~10–15 GPU-h of sampling, plus ~12 h CPU.
* **E1:** 9 new runs, ~1.5 GB each (D2 more); ~100–250 GPU-h; 1.5–2 days wall time on the shared
  node; ~45 CPU-h of downstream scoring. Code: D1 is about 150–300 lines (a parent-latent block,
  target computation, consistency term); D2 is an attention block plus parent-seeded batching.
* **E2 (conditional):** 12 runs, ~250–450 GPU-h, 2–3 days.

**I. Success and failure (pre-registered for E1)**

* **Success.** In every seed:
  * exactly-one-head ≥ 0.85, and ≥ 0.25 above its own independence prediction (D arm);
  * co-member AGE ε² within ±0.03 of 0.291, and INCTOT ε² ≤ 0.096 (no over-coupling) (C and D);
  * head–spouse AGE correlation ≥ 0.80 (D);
  * within-row gaps no worse than the original (INCTOT | age × EDUC ≤ 0.25; EMPSTAT TV ≤ 0.008);
  * out-of-range rows ≤ 2%; Column Shapes ≥ 98.4;
  * synthetic-only income R² ≥ original + 0.005, paired, 3/3 seeds.
* **Failure of the diagnosis.** C and D leave the role and coherence metrics at their independence
  values, or restore them only with the overshoot B shows.
* **Failure of the utility hypothesis, not the diagnosis.** Mechanism metrics are restored but the
  household → member R² increment stays ≤ 0.032 and TSTR does not move. This would confirm the
  California task mismatch and send the utility question to E2 (Berka) and E3 (PUMA representation).

---

## Appendix A. Evidence index (paths under `/storage/s25017/AMEX/`)

| evidence | path |
|---|---|
| GRDM model, loss, reverse step | `java_d/rdbdiff/model.py`; `java_d/rdbdiff/third_party/tab_ddpm/gaussian_multinomial_diffsuion.py` |
| GNN backbone | `java_d/rdbdiff/gnn.py` → relbench `HeteroGraphSAGE` (`.../site-packages/relbench/modeling/nn.py:131-179`) |
| training, EMA, checkpoints | `java_d/rdbdiff/trainer.py`; `java_d/rdbdiff/third_party/tab_ddpm/utils_train.py` |
| sampler | `java_d/rdbdiff/sampler.py`; `java_d/main.py` |
| data, graph and structure | `java_d/rdbdiff/utils/{data,graph}.py`; `java_d/rdbdiff/dataloader.py`; `java_d/rdbdiff/random_graph_sampler.py` |
| configs | `java_d/configs/{berka,california}_fastsample.json`; `java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/config*.json`; `java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/config*.json`; `java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/config.json`; `java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/config.json` |
| fidelity metric code | `java_d/rdbdiff/third_party/clava_ddpm/gen_multi_report.py:28-153` |
| downstream harness | `java_d/downstream/{run,tasks,evaluate,data}.py` |
| aggregate conditioning | `java_d-agg/rdbdiff/aggregates.py`; `java_d-agg/experiments/GRDM_AGGREGATE_CONDITIONED_SCRATCH/{architecture_notes.md,code_snapshot.patch}` |
| degree conditioning | `java_d_degcond/rdbdiff/{degree_conditioning,model_degree}.py`; `.../GRDM_DEGREE_CONDITIONED_SCRATCH/architecture_notes.md` |
| baseline checkpoints and synthetic data | `java_d/rdbdiff_workspace/{berka,california}/paper_repro/seed_{0,1,2}/` |
| other configurations' synthetic data | the `*_CONFIGS` dictionaries in `java_d/results/stage1_limitations/diagnostics/common.py` |
| training logs | `java_d/logs/train/paper_repro_*`; `java_d-agg/.../training_seed*.log`; `java_d-aggft/.../logs/finetune_*`; `java_d-calagg-ft/.../logs/train_*`; `java_d_degcond/.../training*.log` |
| collapse diagnostics | `java_d-aggft/.../checks/{sampling_trace,sampling_path_probe,fixed_batch_loss}.py`, `.../metrics/{sampling_trace_seed0,sampling_path_probe_seed0*,fixed_batch_loss}.json`, `.../REPORT.html` §6; `java_d_degft/.../{README.md,REPORT.html,metrics/diagnosis_degree_shift.md}` |
| verified tables | `java_d/results/relational_fidelity_consolidated/`; `java_d/results/paper_repro/downstream/berka_prediction_tables/`; `java_d/results/consolidated_audit/` |
| California results | `java_d-calagg-ft/.../{REPORT.md,metrics/comparison.json,downstream/*.json,eleven_regime_addendum/}`; `java_d/results/paper_repro/downstream/california__*.json` |
| historical study | `java_d/reports/grdm-final-report.html`; `java_d/reports/archive/*.html`; `java_d/docs/downstream/{DIAGNOSIS,FIXES}.md` |
| **this stage** | `java_d/results/stage1_limitations/diagnostics/*.py`; `java_d/results/stage1_limitations/outputs/{berka_diagnostics,california_diagnostics,california_relationship_importance,california_household_gap,schedule_amplification}.json`, `summary_tables.md` |

## Appendix B. Reproducible commands (everything this stage ran; CPU only)

```bash
cd /storage/s25017/AMEX/java_d/results/stage1_limitations/diagnostics
export CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=4 conda run -n tabsyn python berka_diagnostics.py                 # ~1 min  → outputs/berka_diagnostics.json
OMP_NUM_THREADS=4 conda run -n tabsyn python california_diagnostics.py            # ~1 min  → outputs/california_diagnostics.json
OMP_NUM_THREADS=8 RDBDIFF_XGB_JOBS=8 conda run -n tabsyn python california_relationship_importance.py   # ~3 min
OMP_NUM_THREADS=8 RDBDIFF_XGB_JOBS=8 conda run -n tabsyn python california_household_gap.py             # ~1.5 min
conda run -n tabsyn python schedule_amplification.py                              # seconds
conda run -n tabsyn python summarize.py                                           # → outputs/summary_tables.md
```

The scripts only read existing CSVs and pickles. They write only to
`results/stage1_limitations/outputs/`.
