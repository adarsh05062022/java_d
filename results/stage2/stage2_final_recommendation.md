# Stage 2 — Final recommendation (interim, before any HRD training)

2026-10-15

## Recommendation

**Run one controlled California pilot of HRD (seed 0, three arms) next, and nothing larger yet.** The
pilot needs explicit approval: about 20 GPU-hours on the shared node.

| arm | config | what it isolates | est. cost (contended GPU) |
|---|---|---|---|
| H0h | `hrd_h0h.json` | entity-first ordering alone (no state, no attention) | 3.6 h train + ~3 h sample |
| D1 | `hrd_d1.json` | + generated entity state (30-d) + size input | same |
| D1+D2 | `hrd_d2.json` | + sibling set attention | same (+1% params) |

The reference is the existing original GRDM seed 0, already sampled and scored on the same structure,
normalizers and split.

**Command (prepared, not run):**

```
ARM=hrd_h0h SEED=0 GPU=<g> stage2/launch/phase_d_california.sh
ARM=hrd_d1  SEED=0 GPU=<g> ...
ARM=hrd_d2  SEED=0 GPU=<g> ...
```

Pick GPUs by free memory at launch. All 8 GPUs are at 100% utilisation from other users, but each
run needs < 3 GB. Scoring is CPU only: `stage2/eval/cal_mechanism.py`, plus `downstream_eval.py`
with `--regimes real synth` (~40 min per arm).

**Why this has the highest information value.**
* E0 established that sibling independence is neither a sampler artefact (E0-a) nor a structure
  artefact (E0-b). The architectural hypothesis is therefore now the live question, and it is cheap to
  test on California.
* The pre-registered predictions make each arm falsifiable: D1 restores coherence but not roles; D2
  restores roles; neither over-couples.
* One seed suffices for the mechanism metrics, because their seed-to-seed SD is ≤ 0.015.
* Utility differences need 3 seeds, which are only worth paying for if the mechanism moves.

## Analysis decisions fixed before any Phase D result

1. **Sampler.** Primary analysis **unclipped**, matching the GRDM California baseline. California
   never collapsed (0.00% of rows above 6 in any arm or seed). A clipped resample of each HRD arm is a
   secondary sensitivity check (`sample.py --clip_x0 5.1993 --tag _clip`, ~3 h each). It is not run
   unless the unclipped sample shows > 2% out-of-range rows.
   *Reason:* E0-a passed the fidelity gate, but clipping changed the original's Berka loan TSTR by
   −0.12 on one sample (inconclusive). Switching the sampler and the architecture together would
   confound the comparison.
2. **Success criteria:** the work order's D2 list, unchanged (exactly-one-head ≥ 0.85 and ≥ +0.25
   over independence; AGE ε² within 0.291 ± 0.03; INCTOT ε² ≤ 0.096; head–spouse AGE r ≥ 0.80;
   INCTOT | age × EDUC gap ≤ 0.25; EMPSTAT TV ≤ 0.008; out-of-range ≤ 2%; Column Shapes ≥ 98.4;
   paired TSTR R² ≥ original + 0.005). These are evaluated by `stage2/eval/cal_mechanism.py`
   `criteria()` and the fidelity file.
3. **Pre-registered utility prediction:** neither D1 nor D2 reaches +0.005 R² on California. If that
   happens, it is recorded as a utility-hypothesis failure, and the mechanism result is reported on
   its own merits.
4. **Expansion rule:** go to 3 seeds (plus arm B, aggregate scratch, and a K = 2 GRDM baseline) only if
   D1 or D2 moves a mechanism metric by more than 5× its seed SD. Otherwise stop and report.

## What not to do now

* **No Berka HRD.** HRD is single-relation, and every existing Berka generator is leaky. The first
  Berka step is the leakage-free retrain of the original and aggregate-scratch on `split_train`:
  2 × 3 seeds × 19–38 GPU-h. That needs separate approval.
* **No D3 on California.** E0-b showed structure changes nothing there.
* **No clipping inside HRD's primary comparison** (decision 1).
* **No novelty claim for D1.** It is ClavaDDPM-like. Any claim rests on D2 roles and on calibrated
  (non-feedback) coupling, and only after the pilot and a K = 2 baseline.

## Alternatives considered

| option | cost | why not first |
|---|---|---|
| Full E1 (4 arms × 3 seeds incl. B) | ~80–150 GPU-h | spends 5× the pilot's cost before knowing whether D1/D2 move the mechanism at all |
| PUMA representation ablation (D5) | 1 run, ~7 h | the most likely California *utility* lever, but it tests a different hypothesis; run it in parallel once approved |
| Leakage-free Berka retrain | 6 runs, 120–230 GPU-h | prerequisite for Berka claims; not informative about the architecture |
| Adopt x̂₀ clipping project-wide | sampling only | supported for collapsed FTs; its effect on a stable model's utility is unresolved |
