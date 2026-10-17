# Phase D pilot + D5 PUMA ablation — run record

Approved by the user 2026-10-15 14:28 IST ("go pilot + PUMA in parallel"). Launched 14:29 (HRD) and
14:41 (D5). Analysis rules were fixed beforehand in `stage2_final_recommendation.md` (unclipped primary
analysis; work-order success criteria; utility prediction: no arm reaches +0.005 R²).

| arm | what changes vs original GRDM seed 0 | GPU | code / launcher | outputs |
|---|---|---|---|---|
| `hrd_h0h` | entity-first ordering only | 7 | `stage2/hrd/{train,sample}.py`, `stage2/launch/phase_d_california.sh` | `hrd_runs/hrd_h0h/seed_0/` |
| `hrd_d1` | + generated 30-d entity state + size input | 1 | same | `hrd_runs/hrd_d1/seed_0/` |
| `hrd_d2` | + sibling set attention | 6 | same | `hrd_runs/hrd_d2/seed_0/` |
| `d5` | PUMA as 8 analog bits; **otherwise the unchanged GRDM `main.py`** | 4 | `stage2/d5/puma_bits.py`, `configs/california_puma_bits_fastsample.json`, `stage2/launch/d5_california.sh` | `d5/workspace/seed_0/` (decoded: `..._decoded/synthetic_database`) |

Every arm uses seed 0 and California train data. The three HRD arms run 100k steps (lr 6e-4 → 0) and
2000 diffusion steps per phase. The D5 arm uses the paper_repro settings: 100k steps and the same
schedule.

**Pre-launch checks (VERIFIED).**
* Unit tests pass, 16/16, including the D5 bit round trip and the mapping of invalid patterns to the
  nearest valid code.
* The D5 sampled structure (FK columns) is identical to paper_repro seed 0.
* The D5 `individual` normalizer is byte-identical to paper_repro's.
* The D5 decoder output passes the harness loaders (FK integrity, 28-column schema).
* The PUMA scorer reproduces Stage 1's baseline numbers exactly.

**Automatic scoring.** CPU-only watchers, `stage2/launch/score_phase_d.sh ARM=...` (logs in
`logs/score_phase_d_*.log`), write to `metrics/phase_d/`:
* fidelity;
* model-space out-of-range rows;
* Stage 1 mechanism metrics with the pre-registered criteria;
* PUMA dependency metrics;
* TSTR (synth regime, 3 XGBoost seeds).

**Observed speeds (14:54).**
* HRD training: 0.07–0.11 s/step.
* D5 training: ≈ 0.17 s/step.

**Estimated completion (IST, contended node).**
* HRD training ~16:45–17:15, then entity-first sampling ~3 h.
* Scored results ~20:00–21:30 on 2026-10-15.
* D5 (training + 2000-step sampling): ~01:00 on 2026-10-16.
HRD arms finished 16:08-17:56 (trained+sampled+scored); results in phase_d_pilot_results.md. D5 sampling, ETA scored ~00:15 2026-10-16.
