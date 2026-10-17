# Stage 2 — Phase A repository and experiment audit

2026-10-14 · main repo `/storage/s25017/AMEX/java_d` (branch `paper-repro`, HEAD `a2f961e`) ·
new Stage 2 code in worktree `/storage/s25017/AMEX/java_d-stage2` (branch `stage2-hrd`, from `a2f961e`)

**Scope.** Read-only audit. It builds on the Stage 1 report (`reports/stage1_grdm_limitations.md`,
§2–3), and re-checked each claim used here against the source on 2026-10-14. Line numbers refer to
`java_d-stage2`, which is byte-identical to `java_d` HEAD for `rdbdiff/` and `main.py`. The only new
computation in this phase is `stage2/e0/out_of_range.py`, a CPU pass over existing `node_type_to_X_syn.pkl` files
(output `results/stage2/e0a/out_of_range.json`).

Companion files: `stage2_experiment_inventory.csv` (41 rows, built by `inventory/build_inventory.py`
from on-disk checks) and `stage2_current_job_status.md` (Table 5).

---

## A1. Component map

| component | implementation | key lines |
|---|---|---|
| model | `GRDM(GaussianMultinomialDiffusion)` + `GNNDenoiser` | `rdbdiff/model.py:21` (GRDM), `:362-547` (denoiser) |
| diffusion maths | TabDDPM Gaussian/multinomial diffusion | `rdbdiff/third_party/tab_ddpm/gaussian_multinomial_diffsuion.py` |
| beta schedule | cosine, `betas_for_alpha_bar(max_beta=0.999)`, T = 2000 | `gaussian_multinomial_diffsuion.py:20-62` |
| GNN | relbench `HeteroGraphSAGE` (default path), or `RelationalGraphSAGE` when flags ask; `SAGEConv(aggr=sum)` → `LayerNorm(mode="node")` → ReLU | `rdbdiff/gnn.py:26-158` |
| graph construction | tables → node types; FKs → `f2p`/`rev_f2p` edges; non-key columns → `x` (numerical first) | `rdbdiff/utils/graph.py`, `rdbdiff/utils/data.py:30` `sort_columns` |
| structure generation | `RandomGraphSampler`: per-relation degree histograms, independent draws, random stub matching | `rdbdiff/random_graph_sampler.py:106-130, 199-210, 245-255, 305-306` |
| attribute denoiser | per-table input projection + time embedding → 1 SAGE layer → per-table MLP head (ε̂) | `model.py:478-547` |
| training loss | ε-MSE, Σ over tables of per-table mean over seed rows; one t per seed subgraph | `model.py:128-314` |
| trainer | AdamW lr 6e-4 → 0 linear, wd 1e-5, EMA 0.999, final EMA kept | `rdbdiff/trainer.py` |
| sampler | 2000 reverse steps over the whole database; full 1-hop neighbourhoods; checkpoint every 25 steps | `rdbdiff/sampler.py:58-170` |
| reverse step | `relational_reverse_step` → `gaussian_p_sample(clip_denoised=False)` | `model.py:336-358`; `gaussian_multinomial_diffsuion.py:233-281, 362-387` |
| preprocessing | per-table `QuantileTransformer(normal, n_quantiles=min(N/30,1000))` over **all** non-id columns, categoricals included (`ordinal` default) | `rdbdiff/utils/data.py:95-160` |
| decoding | inverse quantile transform; round + clip categorical codes; snap small-integer columns | `data.py:280-372`, `main.py:640-680` |
| fidelity evaluation | SDV report (Column Shapes, Cardinality) + intra-table and k-HOP column-pair trends | `rdbdiff/third_party/clava_ddpm/gen_multi_report.py`; called at `main.py:690-700` |
| downstream | XGBoost, fixed hyperparameters (`n_estimators=400, max_depth=6, lr=0.1, …`), no tuning, no early stopping; 11 regimes; real held-out test | `downstream/{run,tasks,evaluate,data}.py`; params `downstream/evaluate.py:28-41` |
| aggregate conditioning | bottom-up child statistics added to parent embedding; recomputed from `x_t` every step at sampling | `java_d-agg/rdbdiff/aggregates.py` (also `java_d-aggft`, `java_d-calagg-ft`) |
| fine-tuning | `--init_from_checkpoint_dir`, frozen then joint phases | `java_d-aggft/rdbdiff/finetune.py`, `main.py` diff; `java_d_degft/.../run.py`; `java_d-calagg-ft/.../finetune.py` |
| degree conditioning | 30 per-relation log-degree columns, zero-initialised encoder | `java_d_degcond/rdbdiff/{degree_conditioning,model_degree}.py` |
| real-structure sampling | `main.py --fixed_structure` (uses the training graph's structure) | `main.py:604-616` |
| leakage-free Berka split | account-level 75/25 split, seed 0, with disjointness asserts | `downstream/splits.py:16-70`, `downstream/make_berka_split.py` |

**Configs and seeds.** Baselines: `configs/{berka,california}_fastsample.json`, `--seed {0,1,2}`, defaults
`n_hops=1`, `num_neighbors=[-1]`, `train_batch_size=4096` (confirmed in `logs/train/paper_repro_berka_seed0.log`).
Berka 200k steps, California 100k steps. Fine-tunes use training seed `1000+N`; degree FT samples with
`manual_seed(2000+N)`. All checkpoints listed in the inventory exist on disk (41/41 rows checked,
except the historical `/scratch` runs, which are lost).

## A1. Data flow (noisy input → generated database)

```
TRAINING (per step)
  real DB ──QuantileTransformer(normal), ALL non-id columns incl. categorical codes──► graph_real.x  (every column in [-5.1993, 5.1993])
  4096 seed rows ~ U(all non-dimension rows) ──NeighborLoader(K=1, all neighbours, disjoint)──► subgraphs
  t ~ U{0..1999} per subgraph;  x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε   for EVERY node in the subgraph (parent noised too)
  h0 = W_table x_t + MLP(sin(t))            [agg variant: + AggMLP(child stats of x_t)]  [degree variant: + DegEnc]
  h1 = ReLU(LayerNorm_node(Σ_rel SAGE_rel(h0)))       ← the ONLY cross-row path: one hop, sum, same t
  ε̂_seed = MLP_table(h1_seed);   loss = Σ_tables mean_seeds ‖ε − ε̂‖²

GENERATION
  (1) structure: RandomGraphSampler(seed) — per relation, child counts ~ real histogram, i.i.d.; random matching
  (2) x_T ~ N(0, I) for every row of every non-dimension table
  (3) for t = 1999 … 0, every row once per step (batches of 100k seeds, full 1-hop neighbourhoods):
        ε̂ = denoiser(x_t of row and its 1-hop neighbours, t)
        x̂0 = √(1/ᾱ_t) x_t − √(1/ᾱ_t − 1) ε̂          ← never clipped
        x_{t−1} = c1 x̂0 + c2 x_t + √β_t z           (β_1999 = 0.999)
  (4) decode: inverse quantile (parks |x|>5.2 at column min/max) → round/clip categorical codes → CSV
```

What a row sees within one step: its own `x_t`, and its 1-hop neighbours' `x_t`. An `individual` sees
its household; a household sees all its members, summed. Two siblings are two hops apart, so they have
**no within-step path**. Across steps there is an indirect relay: sibling *i* at step *t* → the
household's `x_{t−1}` → sibling *j* at step *t−1*. This relay can carry only what the household's own
columns can hold (California: 10 columns; Berka `account`: 2).

---

## A2. Verification of the Stage 1 hypotheses

| | hypothesis | verdict | code evidence (this audit) | data evidence |
|---|---|---|---|---|
| **H1** | no generated group-level state | **Confirmed** | `GNNDenoiser` has a projection, time embedding, GNN and heads, and nothing else: no latent channel, no table/group token (`model.py:362-547`). The only per-entity state is the parent row's observed columns. Coordination: direct-neighbour conditioning (1 hop) plus the cross-step parent relay above | California exactly-one-head 0.377 vs independence prediction 0.376 (real 0.933); Berka exactly-one-OWNER 0.727 vs 0.727 (Stage 1 §6, `results/stage1_limitations/outputs/`) |
| **H2** | structure generated independently of attributes and of other relations | **Confirmed** | `RandomGraphSampler` draws each relation's in-degrees with its own `rng.choice` from that relation's marginal histogram (`random_graph_sampler.py:106-118, 245-255`). The code's own comments mark joint and parent-conditioned sampling as not implemented (`:106, :199, :245, :305-306`). The sampler never receives attributes. Structure is drawn before step (3) | ρ(#order, #loan) 0.29 → 0.01; structure digest-identical across all configs per seed (Stage 1 §6.1). California has one relation, so H2 there reduces to **household size ⟂ household attributes** in the generator, which is true by construction |
| **H3** | siblings cannot exchange information during denoising | **Confirmed (within a step); a weak cross-step relay exists** | K = 1 (`main.py` default `--n_hops 1`; GNN `num_layers = n_hops`). Siblings are 2 hops apart. The relay through the parent's `x_t` exists but is bounded by the parent's column count. Nothing in the architecture can express "exactly one member is head" or "spouses have correlated ages" except via that relay | head–spouse AGE corr 0.17 vs 0.93; AGE ε² 0.047 vs 0.291 (Stage 1 §6.2). The relay is empirically negligible: the independence prediction matches to 0.001 |
| **H4** | sampling instability | **Confirmed mechanism; California also affected, but rarely** | (a) `β_t = min(1 − ᾱ(t+1)/ᾱ(t), 0.999)` gives β₁₉₉₉ = 0.999 (`gaussian_multinomial_diffsuion.py:46-62`). (b) `∂x_{t−1}/∂ε̂ = β_t /(√(1−ᾱ_t)√α_t)` = 31.6 at t = 1999 (Stage 1 `schedule_amplification.json`). (c) **x̂0 is never clipped**: `relational_reverse_step` passes `clip_denoised=False` (`model.py:347, 353`), and `gaussian_p_mean_variance` ignores the argument anyway (`:233-281`). (d) Model-space training range is exactly ±5.1993 in every column (captured from `graph_real` by `stage2/e0/x0clip.py`). (e) Decoding hides out-of-range values (`data.py:280-372`). (f) Sampling variance is β_t, not the posterior variance (`:250-256`). (g) **Exposure at high t:** one t per seed subgraph, uniform; Berka `loan` gets ~259 seed samples at t = 1999 in 200k steps vs ~400k for `trans` (Stage 1). California: ~205k seed samples per t over 100k steps, ~55k of them household seeds, so the small-table issue does not apply | Rows with any \|x\| > 6, seed 0/1/2. Berka original loan 1.9 / 1.2 / 1.3%; aggregate joint FT 45 / 53 / 65%; degree FT 49 / 50 / 53%. California original 0.00% (all seeds, all tables), **but max \|x\| = 269 / 12,585 / 5.9 (household / individual / seed 2)**; aggregate FT max 19,545 (household, seed 1). So the runaway trajectories exist on California too, as isolated rows (`results/stage2/e0a/out_of_range.json`) |
| **H5** | nominal high-cardinality categoricals diffused as ordinals | **Confirmed** | `PUMA` is `{"size": 233, "type": "discrete"}` in `data/california/train/household_domain.json`. With the default `categorical_encoding="ordinal"`, `preprocess_database` quantile-transforms **every** non-id column, codes included (`data.py:95-130`). Decoding rounds the continuous value back to the nearest code (`data.py:280-330`), so a ±1 error lands on an arbitrary neighbouring area code. One-hot and multinomial paths exist behind flags and are off | η²(INCTOT; PUMA) 0.062 real vs 0.004 synthetic (Stage 1 §6.2). **Not the only cause:** PUMA explains ~40% of the household → member R² gap; the other 60% is unexplained (Stage 1 F9) |

**Additional findings from this audit**

1. **Out-of-range threshold matters.** Discrete columns put many training rows exactly at ±5.1993,
   so final samples overshoot it slightly. With a 5.2 threshold, 33% of the *original* Berka loan rows
   count as out of range; with 6 it is 1.9%. Every collapse number in Stage 2 is therefore reported at
   5.3 / 6 / 10 / 100 (`out_of_range.py`).
2. **`--fixed_structure` uses the training graph** (`main.py:604-616`). For California this is the
   training split's own structure, which contains no test information. For Berka (all existing
   generators) it is the full database's structure, test accounts included.
3. **A ready-made real-structure harness exists but was never run.** `experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/`
   (committed in `30a6e66`) copies weights into an isolated workspace and calls `main.py --fixed_structure`.
   It is **deleted (uncommitted) in the `java_d` working tree** and intact in the worktrees and in
   `java_d-stage2`. No `T1_E1_real_structure` workspace exists anywhere.
4. **A Berka real-structure sample already exists** for the original GRDM and the degree FT
   (seed 0): `java_d_degft/.../workspace/seed_0/diagnostic_real_structure/{baseline,finetune}/`. It has
   per-column fidelity only, with no downstream scoring yet.

---

## A3. Experiment inventory (summary of `stage2_experiment_inventory.csv`)

| class | experiments (seeds) |
|---|---|
| **1 complete and reusable** | paper_repro original GRDM, Berka (0-2) and California (0-2); Berka aggregate scratch (0-2); Berka aggregate frozen FT, both specs (0-2); Berka degree scratch (0-2); Berka real-structure diagnostics, original and degree FT (seed 0) |
| **2 incomplete but recoverable** | California FT `agg`, `agg_label_free`, `control` (0-2): fidelity complete; 7 of 11 downstream regimes **running now** (Table 5) |
| **3 requires resampling only** | Berka aggregate **joint** FT, both specs (0-2), and Berka degree FT (0-2): weights intact, samples collapsed → E0-a. `T1_E1_REAL_GRAPH_GRDM_ATTR`: tooling only, never run → E0-b |
| **4 requires retraining** | any **leakage-free Berka** generator (`berka_split`): the historical checkpoints were on `/scratch` and are lost |
| **5 invalid or confounded** | Berka smoke test (not an experiment). Additionally, **every Berka downstream number is under the leaky protocol** (generator trained on `data/berka/preprocessed`, which includes `split_test` accounts), and **scratch-vs-FT contrasts are confounded** (steps, LR, init, seed, collapse; Stage 1 §8). Both remain usable for fidelity and within-protocol comparisons |

**Recorded compute** (from each experiment's `training_summary.json`; contended node, wall hours on
one GPU): aggregate scratch 24.1 / 19.2 / 37.6 h; degree scratch 30.0 / 28.3 / 21.1 h; aggregate joint
FT 9.1 h per seed; label-free joint FT 7.1 h; degree FT 3.2 / 3.3 / 2.7 h. Not recorded:
paper_repro baselines, California FT arms, frozen arms.

**Reproducibility from the current repository.** The baseline code path is on `a2f961e`. All
conditioned variants live as **uncommitted** diffs in their worktrees (`java_d-agg`, `java_d-aggft`,
`java_d_degcond`, `java_d_degft`, `java_d-calagg-ft`). They are reproducible from disk, not from git.
Several launchers live only in Claude session scratchpads under `/tmp`. The Table 5 launcher and table
builder have been copied to `results/stage2/table5_preserved/`.

## Repository-state risks noted (none changed by this audit)

* The `java_d` working tree has an uncommitted deletion of `experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/*`.
* `data/berka/split_{train,test}` exist in four worktrees (verified identical with `diff -rq`), **not**
  in `java_d`. `java_d/configs/berka_split_fastsample.json` points at `data/berka/split_train`.
* `results/consolidated_audit/AUDIT_REPORT.md` and the Stage 1 report both describe Table 5's earlier
  state; the job has since been relaunched (see the job-status file).
