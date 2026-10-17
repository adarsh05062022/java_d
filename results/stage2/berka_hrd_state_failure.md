# Berka HRD d2: the entity state was never learned (diagnosis, 2026-10-16)

**Status: verified failure of the as-run Berka HRD d2 config.** Every Berka "full HRD" number reported
so far (loan TSTR 0.861, task battery, mechanism metrics, fidelity) comes from a model whose entity state
was NOT learned. The gains are real observations of that model, but they must not be attributed to the
entity state (D1). The ordering-only runs (h0h) are unaffected.

## Evidence

1. **Training logs** (`hrd_berka_runs/logs/berka_hrd_d2_seed{0,1,2}.log`): state loss is 1.00 from
   step 1k to the end in all three seeds (s1 200k: 0.997; s0 128k: 0.997; s2 131k: 0.978). California
   HRD's state loss falls 0.96 → 0.08 (`hrd_runs/logs/hrd_d2_seed0.log`). The account-column loss is
   also 1.00 throughout in s0 and s1; s2 learns it (1.0 → 0.10, from ~3k steps).
2. **Loss by noise level** (s2 checkpoint, 130k): state loss 1.005–1.027 at every entity timestep
   100…1900, |ε̂_state| ≈ 0.05. A trivial denoiser that only rescales its own noisy input would reach
   ≈ 0.5 on average, so the state head ignores its input entirely.
3. **Dead head** (s0 checkpoint): 100% of the last hidden layer of the account MLP is dead (ReLU 0 for
   every input) → constant output → loss ≡ 1. s2: 92% dead.
4. **Sampled state** (s1 `entity_state_samples.pt`): mean |z| 3.70 (real target: std 0.94), 23.5% of
   values at the ±5.2 clip; the 2 account columns are 23.5% at the clip too (→ account column fidelity
   0.62). With ε̂ ≈ 0 the reverse process pushes z to the clip, so every synthetic account carries a
   large random code shared by all its members.
5. **Cause — sum aggregation swamps the entity's own input.** GRDM's 1-layer HeteroGraphSAGE sums
   messages from children. Per-account L2 norms of the GNN input at initialisation (48 accounts):
   own (root) term 9, transactions 1,408 (≈ 230 trans/account), others ≤ 15 → own share **1.0%**.
   Denoising requires the entity's own noisy x_t; it is 1% of the signal. Seed 2 compensated by growing
   the root weights ~240× (own share 48% at 130k), enough for the 2 account columns but not the 34 state
   columns. California households have ~3 members, so the problem never appeared there.

## Fix (implemented, default off): `hrd.entity_skip`

The entity head gets `GNN output + LayerNorm(own pre-GNN embedding)`, so the entity sees its own input
regardless of fan-out. Schema-agnostic (any high-fan-out entity). Code: `stage2/hrd_rel/rel_model.py`;
flag-off is unchanged (GRDM strict-load equivalence test still passes; suite 29/29 incl. a batch-isolation
test with the skip on).

**CPU A/B** (`stage2/hrd_rel/skip_ab.py`, d2 config, seed 0, identical batches, 400 steps):

| step | account loss: no skip / skip | state loss: no skip / skip |
|---|---|---|
| 50 | 1.03 / 0.87 | 1.00 / 0.98 |
| 150 | 0.94 / 0.60 | 1.01 / 0.88 |
| 300 | 1.00 / 0.49 | 1.00 / 0.62 |
| 400 | 0.99 / 0.40 | 1.02 / 0.51 |

This shows the fix unblocks learning; it does not yet show the full model is better (needs a full run).

## What the existing Berka d2 results actually measure

"Sibling attention + size conditioning + a random large per-account code shared by its members", with
(s0, s1) an untrained account head. That is consistent with what was observed: strong within-account
coordination (exactly-one-OWNER 0.998; loan TSTR 0.861 via loan↔trans attention), over-coupling
(order-amount ε² 0.695 vs real ≈ 0 — a shared random code induces correlation that real data lacks), and
the account-table regression in fidelity and in the task battery.

## Implications

- The claims "D1 + D2 gives the Berka gain" and "the account regression is because 34 state columns
  swamp 2 account columns" are withdrawn. The first is untested on Berka; the second has a different,
  verified cause (dead / input-blind account head).
- California conclusions are unaffected (state loss learned there).
- d2 seeds 0 and 2 still training are the broken config; they can only serve as an ablation
  ("attention + random shared code"), and s0 additionally has a dead account head.
