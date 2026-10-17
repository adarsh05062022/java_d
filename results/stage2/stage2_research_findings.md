# Stage 2 — Research findings (interim: Phases A–C complete, Phase D not run)

2026-10-15 · evidence files: `stage2_repository_audit.md`, `stage2_diagnostic_results.md`,
`metrics/e0_results.csv`, `table5/`, `stage2_architecture_design.md`, `stage2_related_work.md`

**Read this first.** The HRD architecture is implemented and unit-tested. It has **not** been trained at
full budget, so **no HRD result exists**. Every answer below about D1/D2 says "not tested". The
findings that do exist come from re-sampling and re-scoring existing checkpoints (E0) and from
completing Table 5. All E0 results are **one generator sample (seed 0) per arm**.

## 1. The ten questions

| # | question | answer | strength |
|---|---|---|---|
| 1 | Which Stage 1 limitations were reproduced? | **Sibling independence** (every non-aggregate arm sits at its independence prediction to ≤ 0.006, now also under clipping and on the real structure). **Sampling collapse** (unclipped FT samples) and its mechanism. **Fidelity ↑ / utility ↓ on California** (Table 5 now complete: 82/84 paired cells negative). **Structure decoupling** on Berka | verified / preliminary as marked in the diagnostic file |
| 2 | Which are explained by missing architectural capacity? | Sibling independence and the loss of child-aggregate → label signal: supplying the true structure (E0-b) and removing collapse (E0-a) **both leave them unchanged**. Only an information-flow change (aggregate scratch) moves them. The card dependency is not recovered by any tested model, even on the real structure (retention ≤ 0.12) | strong for siblings (code + 2 datasets + 2 interventions); preliminary for card |
| 3 | Which are caused by sampler or optimisation issues? | The **fine-tuning collapse**: x̂₀ clipping alone takes aggregate-FT Column Shapes from 84.4 to 96.4 and degree-FT from 90.8 to 96.8, and loan default rate from 0.36/0.45 to 0.06/0.07, with unchanged weights. Separately, the FT recipes **did not learn** sibling coupling (still at independence after clipping): an optimisation/recipe failure, not collapse | one seed, very large effect |
| 4 | Did entity-level state improve sibling dependency preservation? | **Not tested** (HRD-D1 not trained) | — |
| 5 | Did sibling coordination add benefit beyond entity state? | **Not tested** (HRD-D2 not trained) | — |
| 6 | Did either improve synthetic-only downstream prediction? | **Not tested** | — |
| 7 | Did improvements transfer from California to Berka? | **Not tested.** HRD is single-relation; Berka needs per-relation state and presence flags (design §9, not implemented) | — |
| 8 | Computational cost introduced? | Parameters +0.23% (D1) and +0.99% (D1+D2). Training 0.128 s/step measured (smoke) ≈ 3.6 h per 100k steps. Entity-first sampling doubles the reverse steps: ≈ 2.7 s/step measured → ≈ 3 h per sample. Both are faster than GRDM's PyG pipeline on this node (implementation, not model) | measured on a 300-step / 20-step smoke run |
| 9 | Which findings are statistically reliable across seeds? | **Reliable:** sibling independence of GRDM (3 seeds × 2 datasets, Stage 1, and every E0 arm); Table 5 aggregate-FT deficit (3 seeds, consistent sign); California structure irrelevance (differences ≤ 0.001 against XGBoost SD ~0.0002). **Not reliable yet:** every Berka TSTR difference (19 loan positives, 1 sample); the clipping effect on the original's loan TSTR (−0.12, inconclusive) | — |
| 10 | What is still unknown? | Whether D1/D2 restore sibling structure **without** over-coupling; whether that moves California utility at all; whether D2 beats an implicit deeper-GNN baseline (K = 2) at roles; any leakage-free Berka number; how ClavaDDPM performs on the same mechanism metrics today | — |

## 2. Observed results (no interpretation)

* GRDM samples on the **real** structure keep exactly-one-OWNER at 0.740 vs independence 0.742 (Berka)
  and exactly-one-head at 0.378 vs 0.376 (California).
* Aggregate-scratch GRDM keeps 0.985–0.996 vs 0.893–0.901 with either structure.
* x̂₀-clipped resampling of collapsed checkpoints restores fidelity to the original's level. The
  original's own fidelity moves by ≤ 0.07.
* Clipped fine-tunes remain at their independence predictions. The aggregate FT keeps balance ε² 0.80
  (real 0.55).
* California with the real structure: every metric within 0.002 of the sampled structure.
* Berka with the real structure: TSTR loan +0.065 / +0.070, card +0.025 / +0.043 (two models,
  leaky, one sample).
* Table 5: aggregate FT arms −0.0001 to −0.0038 vs original, 82/84 cells negative; control ≈ 0.

## 3. Mechanistic interpretation (labelled as such)

* **Two independent defects, not one.** Collapse comes from the sampler (an amplifier present in every
  GRDM model, triggered by fine-tuning). Sibling independence comes from information flow. Fixing the
  first does not touch the second (E0-a). Fixing the structure does not touch the second either (E0-b).
  This separates Stage 1's ranked limitations 1 and 3 empirically, which Stage 1 could only do from code.
* **Coupling capacity is learnable from scratch, but not by these fine-tuning recipes.** The aggregate
  pathway created dependence only when trained from scratch. Clipping shows that the FT failure was
  not only collapse.
* **California's utility ceiling is not relational.** Structure is irrelevant there. Restored
  household fidelity (aggregate FT) lowers utility slightly. Stage 1 traced ~40% of the household →
  member gap to PUMA's ordinal encoding. Interpretation: on California, D1/D2 can at best fix the
  mechanism metrics. A utility gain is **not expected** from them. This should be stated *before* the
  Phase D run, as the pre-registered utility prediction.
* **Berka's card dependency needs a two-level path** (trans → account → disp → card). Neither the real
  structure nor one-level aggregates recover it (retention ≤ 0.12).

## 4. Unverified hypotheses

* H-D1: a generated entity state restores co-member coherence (AGE ε², balance ε²) **without** the
  over-coupling of bottom-up aggregates, because it has no feedback loop.
* H-D2: within-step set attention is needed for roles (exactly-one-head). D1 alone cannot do this,
  because members are conditionally independent given (parent, z).
* H-U: neither changes California TSTR by ≥ 0.005 (the pre-registered success threshold), because
  the task's missing signal is PUMA/household-attribute representation, not sibling structure.
* H-S: attribute-conditioned structure (D3) is worth ~+0.03–0.07 AUROC on Berka under a leakage-free
  protocol.

## 5. Future directions (in priority order)

1. HRD pilot on California (H0h, D1, D2; seed 0) to test H-D1 and H-D2 against the pre-registered
   criteria. Then 3 seeds and arm B, if the pilot is informative.
2. A K = 2 GRDM arm on California: the cheapest strong baseline for "implicit sibling interaction",
   needed before any D2 claim.
3. PUMA representation ablation (D5), the most plausible California *utility* lever.
4. Leakage-free Berka retraining (`split_train`) of the original and aggregate scratch, as a
   prerequisite for any Berka claim. Then multi-relation HRD and D3.
5. ClavaDDPM rerun with the same mechanism metrics (the closest D1-type competitor).

## 6. Novelty position (from `stage2_related_work.md`)

* D1 ≈ ClavaDDPM's parent latent (diffused jointly with the parent, latent-conditioned group size,
  children i.i.d. given the latent) and SDV-HMA's summary-statistics idea. **Not novel by itself.**
* D2's goal (joint siblings) exists autoregressively (REaLTabFormer, TabularARGN) and implicitly via
  deep GNNs (GCFM: 3-layer GIN, given structure). Explicit, non-autoregressive, permutation-equivariant
  set attention inside a relational diffusion denoiser **was not found** in the papers checked. That
  is an absence of evidence from a limited search, not a novelty claim.
* The most defensible contribution available **now** is diagnostic:
  * a model-free independence test showing that a published relational diffusion model generates
    siblings as conditionally independent given the parent, which its own k-HOP metrics cannot detect;
  * an intervention-based separation of that defect from sampler collapse (E0-a) and from structure
    generation (E0-b).
