# Stage 2 — Related work and novelty assessment for HRD

2026-10-14. **Status: preliminary.** This is based on web search of abstracts and method sections
(arXiv/NeurIPS/AAAI pages), not a full reading of each paper. Every comparison that matters for a
novelty claim is marked with what was actually checked. "HRD" is a working name; **no novelty is
claimed by this document.** Its job is to say where a claim could and could not be made.

## 1. What HRD is (for comparison)

* **D1.** A generated, continuous per-entity state `z` (30-d: per-member-column mean and std, standardised).
  It is appended to the parent row and diffused with it. Generation is entity-first: parent and `z`,
  then members conditioned on them, with group size as an input to the entity level.
* **D2.** Non-autoregressive, permutation-equivariant self-attention over a parent's member set,
  including the parent token, inside each denoising step.
* **Backbone.** GRDM's one-hop GNN denoiser, unchanged.

## 2. Closest prior work

| work | what it does | overlap with HRD | checked from |
|---|---|---|---|
| **GRDM** (Ketata et al., NeurIPS 2025, [arXiv 2505.16527](https://arxiv.org/abs/2505.16527)) | Degree-preserving random graph for structure, then joint diffusion of all rows with a K-hop GNN denoiser | HRD's backbone. Our Stage 1 shows that at K = 1 siblings are conditionally independent given the parent | paper abstract + full code |
| **ClavaDDPM** (Pang et al., NeurIPS 2024, [arXiv 2405.17724](https://arxiv.org/abs/2405.17724)) | GMM cluster latent `c` fitted on the weighted joint (parent, child) space, majority-voted per parent. **`c` is appended to the parent table and diffused jointly, p(y, c).** Group size is modelled as p(s \| c). Children are generated **i.i.d. given c** (explicitly stated) | **Very high overlap with D1:** a generated parent-level latent, entity-first ordering, and latent-conditioned group size. Differences: ClavaDDPM's latent is a discrete cluster fitted by GMM; D1's is a continuous set-summary target. ClavaDDPM's p(s \| c) is a form of attribute-conditioned structure (our D3) | method section (html v1) |
| **RelDiff** (Hudovernik et al., [arXiv 2506.00710](https://arxiv.org/abs/2506.00710), 2025) | Joint graph-conditioned diffusion over all tables, plus a 2K+SBM structure generator | Same family as GRDM. Its structure generator does not appear to condition on attributes (abstract-level check) | abstract |
| **Graph-Conditional Flow Matching** (Scassola et al., AAAI 2026, [arXiv 2505.15668](https://arxiv.org/abs/2505.15668)) | Flow matching on record features **given the real FK graph** (no structure generation). Denoiser node embeddings come from a GNN (GATv2 + residual, or **3 GIN layers**) computed once per step; any record can influence others in its connected component. Datasets: AirBnB, Walmart, Rossmann, Biodegradability, CORA, IMDb (not California, not Berka). **No sibling or within-group dependency metrics reported** | **Overlaps D2's goal**: with ≥ 2 layers, siblings reach each other within a step (child → parent → child), implicitly through pooled messages. **No set attention over children** (checked in the method section, 2026-10-15). D2's specific difference is explicit pairwise interaction among co-members in a 1-hop backbone | method section (html) |
| **REaLTabFormer** (Solatorio & Dupriez, 2023, [arXiv 2302.02041](https://arxiv.org/abs/2302.02041)) | Parent model, then all of a parent's children generated as one **autoregressive sequence** (seq2seq) | **Models sibling dependence jointly**, which is D2's goal, autoregressively. It imposes an order on members; D2 does not | survey summaries |
| **IRG** ([arXiv 2312.15187](https://arxiv.org/abs/2312.15187)), **RC-TGAN** ([arXiv 2211.07588](https://arxiv.org/abs/2211.07588)) | Incremental / row-conditional GAN generation with ancestor context | Parent → child conditioning, with grandparent context in RC-TGAN | abstracts |
| **TabularARGN** ([arXiv 2501.12012](https://arxiv.org/abs/2501.12012)) | Any-order autoregressive model with sequential (child-sequence) support | Joint child sequences | abstract |
| **SDV HMA** ([docs](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers/hmasynthesizer)) | Recursive Gaussian-copula model; parent rows are augmented with parameters of their children's distributions and modelled jointly with them | **Conceptually the same idea as D1's summary target:** child-distribution statistics carried as generated parent attributes. HMA is classical and parametric | docs only (no algorithm text read) |
| **Joint household–individual population synthesis** ([arXiv 2407.01643](https://arxiv.org/abs/2407.01643); [Comput. Urban Sci. 2025](https://link.springer.com/10.1007/s43762-025-00195-9)) | VAEs over padded fixed-size household + member vectors, or per-size models; explicitly targets within-household relations | **Same problem as California** (household ↔ persons, roles, intra-household relations), solved with joint fixed-size representations | abstracts / summaries |
| **LION / PointFlow-style hierarchical set models** ([arXiv 2210.06978](https://arxiv.org/abs/2210.06978)) | Global (shape) latent plus a set of elements generated conditioned on it; hierarchical latent diffusion | Generic precedent for "entity latent + member set" generation | abstract |

## 3. What this means for a novelty claim

1. **D1 alone is not novel.** A generated parent-level latent diffused with the parent, with children
   conditioned on it and group size conditioned on the latent, is ClavaDDPM's mechanism. Summary
   statistics as parent attributes is SDV HMA's idea. A continuous vs discrete latent is a variation,
   not a contribution, unless it is shown to matter empirically.
2. **D2 alone is not novel in purpose.** Joint modelling of siblings exists autoregressively
   (REaLTabFormer, TabularARGN), with fixed-size padding (population-synthesis VAEs), and implicitly
   through deep GNNs (GCFM). A permutation-equivariant, non-autoregressive set-attention block inside a
   graph diffusion denoiser is a specific design point. GCFM's method section (read 2026-10-15) uses plain message passing (GATv2 or 3-layer GIN), with no set
   attention. RelDiff's has **not** been read beyond the abstract.
3. **What could be defensible**, if experiments support it:
   * **Empirical/diagnostic contribution.** A model-free test showing that a published relational
     diffusion model generates siblings as conditionally independent given the parent (observed =
     independence prediction to 0.001 on two databases), and that the paper's k-HOP metrics cannot
     detect this. This does not depend on HRD.
   * **Calibrated coupling.** A controlled comparison of bottom-up feedback coupling (aggregates
     recomputed from x_t, which over-couples: ε² 0.695 vs 0.553) against top-down generated entity state.
     This holds only if HRD restores dependence *without* overshoot.
   * **Roles inside non-autoregressive relational diffusion.** If D2 restores exactly-one-head
     (0.93 real vs 0.38 GRDM) where D1 and ClavaDDPM-style i.i.d.-given-latent models cannot, by
     construction.
4. **Strongest competing methods to beat or compare against:** ClavaDDPM (D1-equivalent; the
   historical comparison here suggested it carries more cross-table signal than GRDM), GCFM (implicit
   sibling interaction), REaLTabFormer (autoregressive siblings) and RelDiff (current graph-diffusion
   state of the art). **None of them was rerun in Stage 2.** The ClavaDDPM numbers in Stage 1 are
   historical and not re-verifiable.

## 4. To do before any paper claim

* Read RelDiff's method section (GCFM done: no sibling-set attention; 3-layer GIN or GATv2).
* Add a **K = 2 GRDM** (or GCFM-style 3-layer GNN) arm: it is the strongest *implicit* sibling-coupling
  baseline for D2 and is cheap on California.
* Rerun ClavaDDPM on California with the same split and the Stage 1 mechanism metrics (exactly-one-head
  and its independence prediction, ε², head–spouse correlation). Its i.i.d.-given-latent assumption
  predicts it will not fix roles, which is a sharp, testable contrast with D2.
* Search the population-synthesis literature for diffusion-based joint household models (none found
  in this pass).
