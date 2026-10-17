# Stage 2 handoff — start a new chat from here (state as of 2026-10-17 21:30 IST)

Paste into a new chat: *"Read /storage/s25017/AMEX/java_d/results/stage2/HANDOFF.md and continue."*

## 1. Goal (user's words, condensed)

Improve GRDM (graph-based relational diffusion) so the synthetic relational database is **meaningful in
general** — useful for future, unknown downstream tasks and for other relational databases — not tuned
to one task. Berka (8 tables) is the main testbed; California (2 tables) was the first pilot.

## 2. Standing rules (from the Stage 2 work order + user)

- No large GPU/CPU training sweep without explicit user approval: present config, cost, expected
  information, alternatives first. Record approvals and pre-registered criteria in a file before results.
- Never delete/overwrite checkpoints, generated datasets, evaluation outputs or logs; new experiments
  go in new output dirs.
- Never report success without inspecting outputs; never fabricate. Keep observed results separate from
  hypotheses; report negative results; label leaky (old) vs leakage-free (new) protocols.
- Table 5 (California, another session's job) is complete; still never touch its outputs.
- Kill processes by PID/PGID only — `pkill -f <pattern>` matches and kills your own shell.
- Always check **per-term training losses** of any new multi-table config before trusting its results
  (that is how the broken-state bug below was found, late).

## 3. Where things are

| what | path |
|---|---|
| main repo (results, docs) | `/storage/s25017/AMEX/java_d` (branch `paper-repro`) |
| Stage 2 code (worktree, **uncommitted**) | `/storage/s25017/AMEX/java_d-stage2` (branch `stage2-hrd`), code under `stage2/` |
| python env | `conda run --no-capture-output -n tabsyn python ...` |
| tests | `cd java_d-stage2 && CUDA_VISIBLE_DEVICES= conda run -n tabsyn python stage2/tests/run_tests.py` → 29/29 pass |
| Berka leakage-free data | `java_d-stage2/data/berka/split_{train,test}` (account-level split, test 25%, seed 0) |
| leakage-free GRDM baseline (verified) | `/bhallukah/s25017/ClavaDDPM/rdb-diffusion/rdbdiff_workspace/berka_split/paper_repro/seed_{0,1,2}`; rescored in `results/stage2/berka_split_baseline/` |
| multi-table HRD code | `stage2/hrd_rel/` (`rel_data.py` grouping+state target, `rel_model.py` denoiser, `core.py` build/loss/reverse, `train.py`, `sample.py`, `skip_ab.py`) |
| configs | `stage2/hrd_rel/configs/berka_hrd_{h0h,attn,d2skip,d2}.json` (d2 = broken, don't reuse) |
| launcher (CPU train → freest-GPU sample → all scoring) | `ARM=berka_hrd_d2skip SEED=0 THREADS=12 stage2/launch/berka_hrd_cpu.sh` |
| runs | `results/stage2/hrd_berka_runs/<arm>/seed_<s>/`, logs in `hrd_berka_runs/logs/` |
| metrics | `results/stage2/metrics/berka_hrd/` (mechanism JSON, downstream JSON), `metrics/task_battery/berka/` |
| task battery (general-utility eval) | `stage2/eval/task_battery.py` + `task_battery_summary.py <dir>`; method in `results/stage2/task_battery_results.md` |
| California HRD (2-table) | `stage2/hrd/`, results `results/stage2/phase_d_pilot_results.md` |

Nothing is running now. CPU training is faster than the shared GPUs (~0.15–0.26 s/step vs ~0.3);
sampling (~25 min alone, hours when GPUs are contended) runs on a GPU.

## 4. Method (HRD = Hierarchical Relational Diffusion) vs GRDM

GRDM: per-table quantile-normalised columns, Gaussian DDPM (cosine, T=2000, ε-prediction), denoiser =
per-table projection + time embedding → 1-layer HeteroGraphSAGE (SUM aggregation, LayerNorm, ReLU) →
per-table MLP heads; structure from an independent per-relation degree sampler; all tables denoised
jointly at the same t. Known failure (Stage 1, verified): siblings/parents generated nearly independently.

HRD additions (all flags, flag-off == GRDM, strict-loads GRDM checkpoints):
- entity-first ("hierarchical") sampling: entity (account) generated first, then its members
- size input: log1p child counts per direct child relation (zero-init)
- typed sibling attention over all of an entity's disp/loan/order/trans rows + entity token (zero-init out)
- entity state z (D1): standardised mean/std of member columns, diffused with the entity row
- **entity skip** (new): entity head also sees LayerNorm(own pre-GNN embedding)
- x̂₀ clipped to ±5.1993 (the model-space training range)

## 5. Key results so far

**Bug found & fixed (doc: `berka_hrd_state_failure.md`).** The first Berka "full HRD" (d2) never learned
its entity state (state loss 1.0 for 200k steps, all seeds; account head dead in s0/s1): GraphSAGE SUM over
~230 transactions/account makes the account's own input 1% of its GNN input. All earlier Berka d2 numbers
(loan 0.861 etc.) are from that broken model — withdrawn as evidence for D1. Fixed by `entity_skip`.

**Fixed 3-seed comparison (doc: `berka_hrd_fixed_results.md`)** — leakage-free, mean over 3 seeds:

| | GRDM | h0h (ordering) | attn (no state) | d2skip (full) |
|---|---|---|---|---|
| battery rel-view median retention | −0.32 | −0.24 | 0.81 | 0.83 |
| loan TSTR AUROC (real 0.939) | 0.51 | 0.58 | 0.75 | 0.76 |
| card TSTR AUROC (real 0.874) | 0.60 | 0.59 | 0.70 | 0.65 |
| exactly-one-OWNER (real 1.000) | 0.73 | 0.73 | 1.00 | 1.00 |
| Column Shapes | 96.8 | 97.1 | 96.0 | 96.2 |
| AVG 2-WAY | 94.7 | 95.2 | 94.6 | 94.6 |
| account table column fidelity | 91.3 | 97.6 | 97.8 | 97.3 |
| order.amount within-account ε² (real ≈ 0) | −0.02 | 0.02 | 0.41 | 0.39 |

Pre-registered criteria for d2skip (`hrd_berka_runs/APPROVAL_skip.txt`): 4/5 met; **order over-coupling
not fixed**. Interpretation: the gain over GRDM is large and general; it is driven by sibling attention
(+ size input + skip); a working entity state adds only a small sign-consistent battery gain (+0.02),
matches the loan default rate (0.114 vs real 0.112; attn 0.085), but costs card TSTR (−0.05, 3/3 seeds).
Remaining fidelity cost: order/loan/trans column shapes.

Other established facts: California D2 fixes household roles; excluding the label from z is best there;
state dropout failed; card ownership in synthetic data is limited by the structure sampler (card existence
is random w.r.t. attributes) and by the entity seeing only 1 hop.

## 6. Proposed next steps (awaiting user approval — nothing launched)

1. **Held-out database test** (the user's real goal): run GRDM, attn, d2skip unchanged on `ccs` or
   `instacart_05` (data in `/storage/s25017/AMEX/complex_data/` and `/bhallukah/s25017/ClavaDDPM/rdb-diffusion/data/`;
   also f1, movie_lens) and score with the schema-driven task battery. Needs GRDM baselines there first
   (check whether they exist before training).
2. **Fix order over-coupling** (attention makes an account's 1–5 orders too alike; trans coupling is right).
3. **Structure–attribute co-generation**: generate child counts from the entity state instead of the
   independent degree sampler (addresses card-type dependencies generally).
4. Decide whether to keep D1 (entity state) given its small marginal value on Berka.
5. Commit the `java_d-stage2` worktree (currently untracked `stage2/`) when the user asks.

## 7. Other docs in `results/stage2/`

`stage2_repository_audit.md`, `stage2_diagnostic_results.md` (E0-a/E0-b), `stage2_architecture_design.md`,
`stage2_related_work.md` (novelty is weak: ClavaDDPM ≈ D1), `stage2_research_findings.md`,
`stage2_final_recommendation.md`, `phase_d_pilot_results.md` (California), `table5/` (California Table 5),
`task_battery_results.md`, `berka_hrd_state_failure.md`, `berka_hrd_fixed_results.md`.
Claude's persistent memory for this project: `~/.claude/projects/-storage-s25017-AMEX-java-d/memory/stage2-hrd-work.md`.
