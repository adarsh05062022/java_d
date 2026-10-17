"""Build stage2_experiment_inventory.csv from what is on disk.

Read-only: every column is an existence check, a JSON field, or a hand-written classification
(the `classification` and `notes` columns), never a rerun. Hours are copied from each experiment's
own training_summary.json where one exists; otherwise "not recorded".

    python results/stage2/inventory/build_inventory.py
"""

import csv
import glob
import json
import os

AMEX = "/storage/s25017/AMEX"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage2_experiment_inventory.csv")
SYN = "synthetic_data_sampled_structure_size_multiplier_1.0"

CLASSES = {
    1: "complete and reusable",
    2: "incomplete but recoverable",
    3: "requires resampling only",
    4: "requires retraining",
    5: "invalid or confounded",
}


def hours_from_summary(path, *keys):
    try:
        d = json.load(open(path))
        for k in keys:
            d = d[k]
        return d
    except (OSError, KeyError, TypeError, ValueError):
        return None


def exists(p):
    return bool(p) and os.path.exists(p)


def any_glob(pattern):
    return sorted(glob.glob(pattern))


ROWS = []


def add(name, dataset, tasks, arch, init, steps, seed, run_dir, ckpt, eval_glob, downstream_glob,
        hours, reproducible, cls, notes, synth_dir=None):
    synth_dir = synth_dir or os.path.join(run_dir, SYN)
    syn_csv = any_glob(os.path.join(synth_dir, "synthetic_database", "*_synthetic.csv"))
    syn_pkl = os.path.join(synth_dir, "node_type_to_X_syn.pkl")
    evals = any_glob(eval_glob) if eval_glob else []
    ds = any_glob(downstream_glob) if downstream_glob else []
    ROWS.append(dict(
        experiment=name, dataset=dataset, task=tasks, architecture=arch, initialization=init,
        training_steps=steps, seed=seed,
        checkpoint_path=ckpt if exists(ckpt) else f"MISSING: {ckpt}",
        sampling_status=("sampled (csv+pkl)" if syn_csv and exists(syn_pkl)
                         else "sampled (csv only)" if syn_csv else "not sampled"),
        evaluation_status=(f"fidelity eval present ({len(evals)} file(s))" if evals
                           else "no fidelity eval file found"),
        downstream_status=(f"{len(ds)} downstream json(s)" if ds else "none found"),
        output_paths=synth_dir if syn_csv else "",
        gpu_hours=hours if hours is not None else "not recorded",
        reproducible_from_repo=reproducible,
        classification=f"{cls} - {CLASSES[cls]}",
        notes=notes,
    ))


# 1. paper_repro baselines (main repo) ------------------------------------------------------
for s in range(3):
    rd = f"{AMEX}/java_d/rdbdiff_workspace/berka/paper_repro/seed_{s}"
    add("paper_repro original GRDM", "berka (full preprocessed, incl. test accounts)",
        "fidelity; loan_default; card_ownership", "GRDM K=1 sum-SAGE, configs/berka_fastsample.json",
        "scratch", 200000, s, rd, f"{rd}/model_checkpoint/model_ema.pt",
        f"{rd}/{SYN}/eval_result.pkl",
        f"{AMEX}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/berka__*.json",
        None, "yes (main.py, logs/train/paper_repro_berka_seed{s}.log)", 1,
        "Fidelity reusable. Downstream is LEAKY protocol: generator trained on full Berka incl. split_test accounts.")
for s in range(3):
    rd = f"{AMEX}/java_d/rdbdiff_workspace/california/paper_repro/seed_{s}"
    add("paper_repro original GRDM", "california (train split)", "fidelity; inctot; empstat",
        "GRDM K=1 sum-SAGE, configs/california_fastsample.json", "scratch", 100000, s, rd,
        f"{rd}/model_checkpoint/model_ema.pt", f"{rd}/{SYN}/eval_result.pkl",
        f"{AMEX}/java_d/results/paper_repro/downstream/california__*.json", None,
        f"yes (main.py, logs/train/paper_repro_california_seed{s}.log)", 1,
        "Leakage-free (generator saw only data/california/train). 11 regimes x 3 xgb seeds complete.")
rd = f"{AMEX}/java_d/rdbdiff_workspace/berka/smoke_test/seed_0"
add("berka smoke test", "berka", "-", "GRDM", "scratch", "smoke", 0, rd,
    f"{rd}/model_checkpoint/model_ema.pt", None, None, None, "n/a", 5,
    "Pipeline smoke test only; not an experiment.")

# 2. Berka aggregate-conditioned from scratch (java_d-agg) ------------------------------------
E = f"{AMEX}/java_d-agg/experiments/GRDM_AGGREGATE_CONDITIONED_SCRATCH"
for s in range(3):
    rd = f"{E}/runs/seed_{s}"
    add("aggregate-conditioned scratch", "berka (full)", "fidelity; loan_default; card_ownership",
        "GRDM + 31 bottom-up child aggregates on account/disp (rdbdiff/aggregates.py)", "scratch",
        200000, s, rd, f"{rd}/model_checkpoint/model_ema.pt",
        f"{AMEX}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/evaluation_agent2_scratch_seed{s}_*.json",
        f"{AMEX}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_agg_conditioned__*.json",
        hours_from_summary(f"{E}/training_summary.json", "seeds", str(s), "wall_hours_since_first_launch"),
        "yes (worktree java_d-agg, uncommitted diff in code_snapshot.patch)", 1,
        "Wall hours since first launch (contended). Downstream LEAKY protocol. Over-couples (trans.balance eps2 0.695 vs 0.553).")

# 3. Berka aggregate fine-tunes (java_d-aggft) -------------------------------------------------
E = f"{AMEX}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE"
for arm, desc, cls, note in [
    ("same_spec", "joint FT (5,280 frozen + 39,600 joint)", 3,
     "Sampling COLLAPSED (54% loan rows out of range). Weights may be usable: E0-a target (resample with x0 clipping)."),
    ("label_free", "joint FT, 27 label-free aggregates", 3, "Same collapse pattern as same_spec; E0-a candidate."),
    ("same_spec_frozen", "frozen FT (5,280 steps, new layers only)", 1, "Stable; ~= original."),
    ("label_free_frozen", "frozen FT, label-free", 1, "Stable; ~= original."),
]:
    for s in range(3):
        rd = f"{E}/runs/{arm}/seed_{s}"
        steps = 5280 if "frozen" in arm else 44880
        h = hours_from_summary(f"{E}/training_summary.json", "runs", f"{arm}/seed{s}", "train_hours")
        add(f"aggregate-conditioned FT ({arm})", "berka (full)", "fidelity; loan_default; card_ownership",
            desc, "original paper_repro EMA", steps, s, rd, f"{rd}/model_checkpoint/model_ema.pt",
            f"{E}/metrics/evaluation_ft_{arm}_seed{s}_final.json",
            f"{E}/metrics/downstream/berka_aggft_{arm}__*.json", h,
            "yes (worktree java_d-aggft)", cls, note + " Downstream LEAKY protocol. Confounded vs scratch (steps, LR, init, seed).")

# 4. Berka degree-conditioned scratch / FT -----------------------------------------------------
E = f"{AMEX}/java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH"
for s in range(3):
    sec = hours_from_summary(f"{E}/training_summary.json", f"seed_{s}", "training_seconds_total")
    add("degree-conditioned scratch", "berka (full)", "fidelity; loan_default; card_ownership",
        "GRDM + 30 per-relation log-degree inputs (model_degree.py)", "scratch", 200000, s,
        f"{E}/workspace/seed_{s}", f"{E}/checkpoints/seed_{s}/model_ema.pt", f"{E}/metrics/*seed{s}*.json",
        f"{E}/metrics/downstream/berka_degcond__*seeds{s}*.json",
        round(sec / 3600, 2) if sec else None, "yes (worktree java_d_degcond)", 1,
        "Loan 16% rows out of range; default rate 0.208 vs 0.111. Downstream LEAKY protocol.")
E = f"{AMEX}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE"
for s in range(3):
    sec = hours_from_summary(f"{E}/training_summary.json", f"seed_{s}", "training_seconds_total")
    add("degree-conditioned FT", "berka (full)", "fidelity; loan_default; card_ownership",
        "GRDM + degree inputs; 5,280 encoder-only + 15,840 joint, lr 3e-5", "original paper_repro EMA",
        21120, s, f"{E}/workspace/seed_{s}", f"{E}/checkpoints/seed_{s}/model_ema.pt",
        f"{E}/metrics/fidelity_seed{s}.json", f"{E}/metrics/downstream/berka_degft__*seeds{s}*.json",
        round(sec / 3600, 2) if sec else None, "yes (worktree java_d_degft)", 3,
        "Sampling COLLAPSED (51% loan rows out of range) without any aggregate pathway. E0-a target.")
for model in ["baseline", "finetune"]:
    d = f"{E}/workspace/seed_0/diagnostic_real_structure/{model}"
    add(f"real-structure diagnostic ({'original GRDM' if model == 'baseline' else 'degree FT'})",
        "berka (full; REAL structure incl. test accounts)", "per-column fidelity only",
        "sampled on real graph (diag_real_structure.py)", "existing EMA (copied)", "-", 0, d,
        f"{d}/model_ema.pt", f"{E}/metrics/diag_real_structure_{model}_seed0.json", None, None,
        "yes", 1, "One seed. Diagnostic only (real structure). Reusable for E0-b on Berka (no downstream run yet).",
        synth_dir=d)

# 5. California aggregate fine-tunes (java_d-calagg-ft) ----------------------------------------
E = f"{AMEX}/java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE"
for arm, ds_name, desc, steps in [
    ("agg", "california_agg_ft", "20 household aggregates; 5,070 frozen + 20,280 joint", 25350),
    ("agg_label_free", "california_agg_ft_label_free", "16 label-free aggregates; same recipe", 25350),
    ("control", "california_ft_control", "no aggregates; 20,280 joint steps lr 1e-4", 20280),
]:
    for s in range(3):
        rd = f"{E}/runs/{arm}/seed_{s}"
        add(f"California FT ({arm})", "california (train split)", "fidelity; inctot; empstat", desc,
            "original paper_repro EMA", steps, s, rd, f"{rd}/model_checkpoint/model_ema.pt",
            f"{rd}/fidelity.json", f"{E}/downstream/{ds_name}__*.json", None,
            "yes (worktree java_d-calagg-ft)", 2,
            "Fidelity complete. Downstream: real/synth/mixed/blend_50 complete; 7 regimes RUNNING "
            "(Table 5, *_missing.json pending). Leakage-free.")

# 6. Historical / planned ------------------------------------------------------------------------
ROWS.append(dict(
    experiment="berka_split leakage-free GRDM (historical)", dataset="berka split_train",
    task="loan_default; card_ownership", architecture="GRDM (+ historical fixes S1-S13)",
    initialization="scratch", training_steps="200000", seed="0,1,2",
    checkpoint_path="LOST (/scratch unmounted 2026-10-07)", sampling_status="lost",
    evaluation_status="java_d/downstream/results/berka_split__*.json (stale)",
    downstream_status="stale json only", output_paths="", gpu_hours="not recorded",
    reproducible_from_repo="code yes; artifacts no", classification=f"4 - {CLASSES[4]}",
    notes="Only leakage-free Berka generator ever trained; must be retrained on split_train for a clean protocol."))
ROWS.append(dict(
    experiment="T1_E1_REAL_GRAPH_GRDM_ATTR", dataset="berka_split / california",
    task="downstream on real graph + GRDM attributes", architecture="GRDM (sampling only)",
    initialization="paper_repro EMA", training_steps="-", seed="0,1,2",
    checkpoint_path="n/a", sampling_status="never run (no rdbdiff_workspace/*/T1_E1_real_structure)",
    evaluation_status="none", downstream_status="none", output_paths="",
    gpu_hours="-", reproducible_from_repo="code in commit 30a6e66 (deleted from java_d working tree, present in worktrees)",
    classification=f"3 - {CLASSES[3]}",
    notes="Ready-made E0-b tooling (isolated workspace copy + main.py --fixed_structure)."))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
    w.writeheader()
    w.writerows(ROWS)
print(f"wrote {OUT} ({len(ROWS)} rows)")
