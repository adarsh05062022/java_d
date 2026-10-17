"""Compute the three cross-table dependency metrics (parent<->aggregate correlation MAD,
aggregate<->aggregate correlation MAD, strong-dependency |r|>=0.1 sign agreement) plus the
aggregate-KS metric for the two degree-conditioned configurations, using the EXACT SAME
account_view()/aggregate_metrics() code already validated against baseline and the
aggregate-conditioned models in:
    /storage/s25017/AMEX/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/evaluate_run.py

No retraining, no model code touched -- this only reads already-generated synthetic_database
CSVs that exist on disk and the real preprocessed Berka tables (identical symlink target across
every worktree, confirmed via `readlink -f` + `diff` of downstream/data.py and downstream/tasks.py
across all five worktrees before writing this script).
"""
import importlib.util
import json
import os
import sys

import numpy as np

AGGFT_ROOT = "/storage/s25017/AMEX/java_d-aggft"
BASELINE_ROOT = "/storage/s25017/AMEX/java_d"
DEGCOND_ROOT = "/storage/s25017/AMEX/java_d_degcond"
DEGFT_ROOT = "/storage/s25017/AMEX/java_d_degft"
REAL = os.path.join(BASELINE_ROOT, "data", "berka", "preprocessed")
SYN_SUB = "synthetic_data_sampled_structure_size_multiplier_1.0"

# Import the validated evaluate_run.py module (defines account_view, aggregate_metrics,
# load_syn, check_schema) without running its __main__ block.
sys.path.insert(0, AGGFT_ROOT)
os.chdir(AGGFT_ROOT)  # evaluate_run.py does `ROOT = ...; sys.path.insert(0, ROOT)` relative to itself
spec = importlib.util.spec_from_file_location(
    "aggft_evaluate_run",
    os.path.join(AGGFT_ROOT, "experiments", "GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE", "evaluate_run.py"),
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

from downstream.data import coerce_discrete, load_db, load_domains, load_meta  # noqa: E402

meta = load_meta(REAL)
tables = list(meta["tables"])
domains = load_domains(REAL, tables)
real = coerce_discrete(load_db(REAL), domains)
real_v = ev.account_view(real, domains)

CONFIGS = {
    "degree_scratch": [
        os.path.join(DEGCOND_ROOT, "experiments", "GRDM_DEGREE_CONDITIONED_SCRATCH", "workspace", f"seed_{s}")
        for s in (0, 1, 2)
    ],
    "degree_finetune": [
        os.path.join(DEGFT_ROOT, "experiments", "GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE", "workspace", f"seed_{s}")
        for s in (0, 1, 2)
    ],
}

out = {}
per_seed_records = {}
for label, run_dirs in CONFIGS.items():
    seed_results = []
    for s, run in enumerate(run_dirs):
        syn_db = os.path.join(run, SYN_SUB, "synthetic_database")
        if not os.path.exists(syn_db):
            print(f"{label} seed {s}: MISSING synthetic_database at {syn_db}")
            continue
        syn = coerce_discrete(ev.load_syn(run, tables), domains)
        schema = ev.check_schema(real, syn, meta)
        syn_v = ev.account_view(syn, domains)
        agg = ev.aggregate_metrics(real_v, syn_v)
        rec = {"seed": s, "run_dir": run, "schema_ok": schema["ok"],
               "fk_integrity": schema["fk_integrity"], "aggregates": agg}
        seed_results.append(rec)
        print(f"{label} seed {s}: schema_ok={schema['ok']} fk={schema['fk_integrity']} "
              f"aggKS={agg['aggregate_ks_mean']:.4f} xcorrMAD={agg['cross_table_corr_mad']:.4f} "
              f"aggaggMAD={agg['aggregate_aggregate_corr_mad']:.4f} "
              f"sign={agg['cross_table_sign_agreement_|r|>=0.1']}")
    per_seed_records[label] = seed_results

    keys = ["aggregate_ks_mean", "cross_table_corr_mad", "aggregate_aggregate_corr_mad",
            "cross_table_sign_agreement_frac"]
    summary = {}
    for k in keys:
        vals = [r["aggregates"][k] for r in seed_results if r["aggregates"][k] is not None]
        summary[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    out[label] = summary

OUT_DIR = "/tmp/claude-1070/-storage-s25017-AMEX-java-d/fd48dfeb-a212-4bc8-a6e6-5c7fe19830c2/scratchpad"
with open(os.path.join(OUT_DIR, "degree_dep_metrics_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
with open(os.path.join(OUT_DIR, "degree_dep_metrics_per_seed.json"), "w") as f:
    json.dump(per_seed_records, f, indent=2, default=str)

print()
print(json.dumps(out, indent=2))
