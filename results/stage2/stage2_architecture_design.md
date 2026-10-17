# Stage 2 — HRD architecture design (D1 entity state, D2 sibling coordination)

2026-10-14 · code: `/storage/s25017/AMEX/java_d-stage2/stage2/hrd/` (branch `stage2-hrd`, uncommitted)
· **status: implemented and unit-tested; smoke-tested end to end on California (300 steps); not
trained at full budget; no results.**

"Hierarchical Relational Diffusion (HRD)" is a provisional name. `stage2_related_work.md` shows
that D1 closely overlaps ClavaDDPM's parent-latent mechanism. No novelty is claimed here.

## 1. What GRDM cannot represent directly, and what HRD adds

| dependency | GRDM (K = 1, as run) | HRD-D1 | HRD-D1+D2 |
|---|---|---|---|
| member ← own parent's columns | yes (1 hop) | yes | yes |
| shared group factor beyond the parent's observed columns (e.g. household age level, account balance level) | **no generated variable carries it.** Only the parent's own 10 (California) or 2 (Berka `account`) columns relay it across steps; empirically negligible (independence test) | **yes**: a generated per-entity state `z` (30-d), and members are generated conditioned on the finished `z` | yes |
| group size → member | invisible to members (sum → LayerNorm) | entity level conditioned on log(1+size); members receive it through `z` | plus attention sees the set directly |
| exclusivity / roles (exactly one head) and pairwise structure (head–spouse) | **not representable**: members are conditionally independent given the parent | still conditionally independent given (parent, `z`), so D1 is predicted **not** to fix roles | **representable**: within each denoising step every member attends to every co-member, so "someone else is already head" can be expressed |
| coupling strength | — | learned through denoising of a generated variable, not a feedback loop | same |
| bottom-up feedback (existing aggregate fix) | aggregates recomputed from members' current `x_t` at every step → positive feedback, over-coupling (ε² 0.695 vs 0.553) and collapse spreading | **no feedback path**: `z` is generated like any other column, never recomputed from members | same |

## 2. Generative process (entity-first, `timestep_mode="hierarchical"`)

```
structure  (unchanged: paper_repro seed-N sampled structure, i.e. GRDM's RandomGraphSampler output)
    │   group sizes n_g are known before any attribute
    ▼
PHASE 1 — entities          for t = T-1 … 0  (members held at pure noise, t_C = T-1)
    [x_P , z]_t ──proj_P + temb(t) + Lin(log(1+n_g))──┐
    x_C^T (noise) ──proj_C + temb(T-1)────────────────┤
                                                     1 × SAGE(sum) → LayerNorm → ReLU   (GRDM's module)
                                                      └─► MLP_P → ε̂ for [x_P, z] → reverse step
    ⇒ finished parent rows and finished entity state z_g
    ▼
PHASE 2 — members           for t = T-1 … 0  (parent rows fixed and clean, t_P = 0)
    [x_P , z]_0 ──proj_P + temb(0) + size──┐
    x_C,t ──proj_C + temb(t)───────────────┤
                                          1 × SAGE(sum) → LayerNorm → ReLU
                                          [D2] member tokens + parent token → LN → MHA (no positions,
                                               padding-masked, zero-init out-proj) → residual
                                           └─► MLP_C → ε̂ for x_C → reverse step
    ▼
decode exactly as main.py (inverse quantile, round categorical codes, CSV); z is dropped
```

`timestep_mode="shared"` runs both levels together at one t, which is GRDM's scheme. HRD-H0 (shared,
no D1, no D2) is the same function as GRDM's `GNNDenoiser`. **Verified:** it loads GRDM's California
seed-0 EMA weights with `strict=True` and reproduces GRDM's ε̂ through GRDM's own sampler dataloader
(`test_hrd_flags_off_equals_grdm_checkpoint`, atol 1e-4).

## 3. D1 — generated entity state

| question (work order C2) | answer |
|---|---|
| dimensionality | `d_z = 2 × d_C` = 30 on California: the mean and the population std of each of the 15 member columns over the group's members, in model (quantile-normal) space |
| initialisation / target | the **clean training target** is that summary of the group's **training** members, standardised with μ, σ fitted once on the training graph and clipped to ±5.1993 (the data range). Stored in `entity_state_target.json` |
| how it is generated | by the **same diffusion** as the parent attributes: appended as 30 extra parent columns, noised and denoised with them. At sampling it starts from N(0, I) and is never computed from, or copied from, any real row |
| gradient flow | one network. The ε-loss on `z`'s columns trains the parent head to generate `z`. The ε-loss on members trains everything to use `z` (proj_P → GNN message → member head, plus D2). The target is a fixed function of data, so it carries no gradient and cannot collapse. Loss = mean over parent attribute columns + mean over member columns + `state_weight` × mean over `z` columns (weight 1). The parent-attribute term keeps exactly GRDM's weight |
| variable member counts | the summary is a masked sum over members (`index_add`), with std = 0 for single-member groups. log(1+n_g) enters the entity embedding through a zero-initialised linear map. D2 pads to the largest group in the batch and masks the padding |
| missing / optional relations | California: every household has ≥ 1 member. A parent with 0 children would get the training mean as its summary target (count 0 is clamped to 1 in the mean). **Not yet exercised.** Berka's optional relations (loan, card) need a per-relation presence flag in `z` before HRD can run there (open item) |
| new entities vs memorisation | `z` is generated from noise jointly with the parent's own columns. Training rows enter only as the denoising target, exactly as GRDM's attributes do. No nearest-neighbour or copy step exists. Memorisation is to be checked like GRDM's (DCR) if a privacy claim is ever made; none is made |
| no label leakage | `z` is computed from **training-split** members only. California's test split is never read by `train.py` / `sample.py` (only by the downstream harness). INCTOT and EMPSTAT are member columns, so their training-member means are part of `z`, exactly as they are part of what GRDM itself generates. A label-free variant (drop those two columns from `z`) is a one-line config change if a reviewer asks |
| hard-coded relationships | none. `z` is generic first and second moments of every member column. It encodes no roles, pairs or domain rules |

## 4. D2 — sibling coordination

* **Mechanism.** One `nn.MultiheadAttention` block (4 heads, 128 channels) over [parent token, member₁, …, member_n]
  per group, pre-LayerNorm, residual. Its output projection is zero-initialised, so an untrained D2 is
  exactly D1 (tested).
* **Properties (each unit-tested):** permutation-equivariant in members, with no positional encoding;
  padding-masked, so a group's output is identical alone or batched with groups of other sizes;
  groups never see each other.
* **No clean-target leakage:** attention reads only the members' **noisy** embeddings at the current
  step (and the parent's input at its own noise level). The clean `x_0` never enters the denoiser.
* **Cost:** O(Σ_g n_g²) per step. California n_g ≤ 25, mean 2.74, so this is negligible. Berka
  accounts have ~235 transactions: 55k pairs per account, feasible but untested.

## 5. Variants and configs (`stage2/hrd/configs/`)

| config | timestep mode | D1 entity state | size input | D2 attention | role in the plan | params | Δ vs GRDM |
|---|---|---|---|---|---|---|---|
| (GRDM paper_repro) | shared | – | – | – | **A** reference (existing, 3 seeds) | 8,645,529 | — |
| `hrd_h0.json` | shared | – | – | – | control: GRDM function on whole-group batches | 8,645,529 | 0 |
| `hrd_h0h.json` | hierarchical | – | – | – | ablation: entity-first ordering only | 8,645,529 | 0 |
| `hrd_d1.json` | hierarchical | ✓ (30-d) | ✓ | – | **C** HRD state only | 8,665,015 | +0.23% |
| `hrd_d2.json` | hierarchical | ✓ (30-d) | ✓ | ✓ | **D** HRD state + sibling | 8,731,319 | +0.99% |

Adjacent main arms C → D differ **only** in D2. A → C bundles three choices that together define
"generate the entity first": ordering, the state, and the size input to the entity level. `hrd_h0h`
isolates ordering, and `hrd_h0` isolates the batching and re-implementation (expected ≈ A, by
construction). All arms share data, normalizers (byte-verified copies of paper_repro seed N), sampled
structure per seed, steps (100k), LR schedule (6e-4 → 0 linear), AdamW (wd 1e-5), EMA 0.999, rows per
step (1,094 groups ≈ GRDM's 4,096 seeds), sampler length (T = 2000 per phase) and evaluation code.

**Known, documented differences from GRDM's training distribution:** members of one group share
one t_C, and parent and members draw independent t's in the hierarchical modes. In GRDM every 1-hop
subgraph shares one t. Sampler x̂₀ clipping is **off** in every arm, unless the E0-a gate (stage2_diagnostic_results.md) says to adopt it for all.

## 6. Computational cost (measured / estimated)

| item | GRDM (paper_repro, California) | HRD |
|---|---|---|
| params | 8.65 M | 8.65–8.73 M (table above) |
| training step | not recorded for the baseline; the Stage 1 estimate is 8–24 h per 100k-step run on this node (PyG NeighborLoader) | **0.128 s/step measured** (D2, 300-step smoke, GPU 6 at 100% utilisation from other users) → **≈ 3.6 h per 100k steps** |
| sampling | E0-b measured 4.5 h for 2000 steps on California (≈ 8 s/step) | entity-first doubles the step count (2 × 2000). Smoke (20 + 20 steps, full 554k households / 1.52M members): phase 1 2.8 s/step (before the unused member head was skipped), phase 2 2.65 s/step → **≈ 3 h per full sample** (estimate) |
| GPU memory | ~1.5 GB | < 3 GB (whole training graph held on the GPU) |

The speed-up in training comes from dense group batching instead of PyG's per-batch homogeneous ↔
heterogeneous conversion, not from the model. It is an implementation difference, not a claim.

## 7. Changed / added files (all new; nothing in `rdbdiff/` or `main.py` is modified)

```
stage2/hrd/model.py         EntityStateTarget, pack_groups, SiblingAttention, HRDDenoiser
stage2/hrd/diffusion.py     HRDDiffusion (GRDM schedule; p_step with optional x0 clip), hrd_loss
stage2/hrd/data.py          load_graph (main.py's preprocessing), GroupedTables, GroupBatcher
stage2/hrd/train.py         training (resumable; --max_hours; exit 42 = budget)
stage2/hrd/sample.py        entity-first / shared sampling, main.py decoding, fidelity report
stage2/hrd/configs/*.json   hrd_h0, hrd_h0h, hrd_d1, hrd_d2
stage2/eval/cal_mechanism.py    Stage 1 California mechanism metrics + pre-registered criteria
stage2/eval/downstream_eval.py  unchanged downstream harness on any synthetic folder
stage2/e0/{x0clip,run_e0a,run_e0b,out_of_range}.py   E0 diagnostics
stage2/tests/{run_tests,test_hrd,test_x0clip}.py      14 tests, all passing
```

## 8. Unit tests (14/14 pass; `conda run -n tabsyn python stage2/tests/run_tests.py`)

* GRDM equivalence on the real checkpoint (strict load + identical ε̂).
* `p_step` == `GRDM.gaussian_p_sample` at t ∈ {1999, 1000, 1, 0}.
* Entity-state values (mean/std; single-member std = 0) and group locality (no cross-entity leakage).
* `pack_groups` slots and masks; attention padding invariance and group isolation; permutation
  equivariance of attention and of the full model; untrained D2 == D1.
* Batcher: an epoch covers every group exactly once, and gathering returns whole groups.
* Train/sample compatibility: loss + backward and both sampling modes run, with finite, correctly shaped outputs.
* x0clip (E0-a): mode "off" bit-identical; "column" equals the analytic clamped step; inactive outside the reverse step.

## 9. Open items before Berka (D3 follow-up)

Berka has multiple child relations, optional relations (loan, card) and a two-level path (account →
disp → card). HRD as written handles one parent → one child relation. Extending it needs per-relation
state blocks, presence flags, and either a per-relation member set for D2 or restricting D2 to `disp`.
This is not implemented.
