"""Aggregate downstream prediction results across the 6 GRDM Berka configurations.

Reads the raw per-record JSON files produced by downstream/run.py (schema:
{"dataset":.., "leakage_caveat":.., "records":[{regime, grdm_seed, xgb_seed, n_train_real,
n_train_syn, metrics:{roc_auc, average_precision, f1_macro, ...}}, ...]}) from each worktree,
dedupes identical records (several worktrees recomputed the same baseline/reference locally),
and aggregates mean/std across grdm_seed (averaging over xgb_seed within each grdm_seed first).
"""
import json
import glob
import statistics as stats
from collections import defaultdict

ROOT = "/storage/s25017/AMEX"

# (config_label, task, leakage_caveat_expected, [list of json file globs to merge])
SOURCES = {
    ("baseline_leaky", "berka_loan_default"): [
        f"{ROOT}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/berka__berka_loan_default__seeds0-1-2.json",
    ],
    ("baseline_leaky", "berka_card_ownership"): [
        f"{ROOT}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/berka__berka_card_ownership__seeds0-1-2.json",
    ],
    ("baseline_split_clean", "berka_loan_default"): [
        f"{ROOT}/java_d/downstream/results/berka_split__berka_loan_default.json",
    ],
    ("baseline_split_clean", "berka_card_ownership"): [
        f"{ROOT}/java_d/downstream/results/berka_split__berka_card_ownership.json",
    ],
    ("degree_scratch", "berka_loan_default"): [
        f"{ROOT}/java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/metrics/downstream/berka_degcond__berka_loan_default__seeds{s}.json" for s in (0, 1, 2)
    ],
    ("degree_scratch", "berka_card_ownership"): [
        f"{ROOT}/java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/metrics/downstream/berka_degcond__berka_card_ownership__seeds{s}.json" for s in (0, 1, 2)
    ],
    ("degree_finetune", "berka_loan_default"): [
        f"{ROOT}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/berka_degft__berka_loan_default__seeds{s}.json" for s in (0, 1, 2)
    ],
    ("degree_finetune", "berka_card_ownership"): [
        f"{ROOT}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/metrics/downstream/berka_degft__berka_card_ownership__seeds{s}.json" for s in (0, 1, 2)
    ],
    ("agg_scratch", "berka_loan_default"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_agg_conditioned__berka_loan_default.json",
    ],
    ("agg_scratch", "berka_card_ownership"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_agg_conditioned__berka_card_ownership.json",
    ],
    ("agg_finetune_joint", "berka_loan_default"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_same_spec__berka_loan_default.json",
    ],
    ("agg_finetune_joint", "berka_card_ownership"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_same_spec__berka_card_ownership.json",
    ],
    ("agg_finetune_frozen", "berka_loan_default"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_same_spec_frozen__berka_loan_default.json",
    ],
    ("agg_finetune_frozen", "berka_card_ownership"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_same_spec_frozen__berka_card_ownership.json",
    ],
    # label_free variants (not one of the 6 requested configs, kept for an appendix note)
    ("agg_finetune_joint_labelfree", "berka_loan_default"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_label_free__berka_loan_default.json",
    ],
    ("agg_finetune_joint_labelfree", "berka_card_ownership"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_label_free__berka_card_ownership.json",
    ],
    ("agg_finetune_frozen_labelfree", "berka_loan_default"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_label_free_frozen__berka_loan_default.json",
    ],
    ("agg_finetune_frozen_labelfree", "berka_card_ownership"): [
        f"{ROOT}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/metrics/downstream/berka_aggft_label_free_frozen__berka_card_ownership.json",
    ],
}

REGIME_PROPS = {
    "real": (1.0, 0.0), "synth": (0.0, 1.0),
    "mixed_25": (1.0, 0.25), "mixed_50": (1.0, 0.50), "mixed": (1.0, 1.00),
    "real_50": (0.5, 0.0), "blend_50": (0.5, 0.5),
    "real_25": (0.25, 0.0), "blend_25": (0.25, 0.75),
    "real_75": (0.75, 0.0), "blend_75": (0.75, 0.25),
}

def load_records(paths):
    seen = set()
    out = []
    leak_vals = set()
    dataset_vals = set()
    for p in paths:
        try:
            d = json.load(open(p))
        except FileNotFoundError:
            print(f"  [missing] {p}")
            continue
        leak_vals.add(d.get("leakage_caveat"))
        dataset_vals.add(d.get("dataset"))
        for r in d["records"]:
            key = (r["regime"], r["grdm_seed"], r["xgb_seed"], tuple(sorted(r["metrics"].items())))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out, leak_vals, dataset_vals


def aggregate(records):
    """regime -> {metric -> (mean, std, n_seeds)} plus n_train_real/syn example."""
    by_regime = defaultdict(list)
    for r in records:
        by_regime[r["regime"]].append(r)

    result = {}
    for regime, recs in by_regime.items():
        by_seed = defaultdict(list)
        for r in recs:
            by_seed[r["grdm_seed"]].append(r)
        metric_names = sorted(recs[0]["metrics"].keys())
        per_seed_metric = defaultdict(list)  # metric -> [mean_over_xgbseeds_for_this_grdmseed, ...]
        for seed, seed_recs in by_seed.items():
            for m in metric_names:
                vals = [sr["metrics"][m] for sr in seed_recs]
                per_seed_metric[m].append(sum(vals) / len(vals))
        n_seeds = len(by_seed)
        metric_stats = {}
        for m in metric_names:
            vals = per_seed_metric[m]
            mean = sum(vals) / len(vals)
            std = stats.pstdev(vals) if len(vals) > 1 else 0.0
            metric_stats[m] = (mean, std, n_seeds)
        n_real_ex = recs[0].get("n_train_real")
        n_syn_ex = recs[0].get("n_train_syn")
        result[regime] = dict(metrics=metric_stats, n_train_real=n_real_ex, n_train_syn=n_syn_ex)
    return result


def main():
    all_out = {}
    for (config, task), paths in SOURCES.items():
        records, leak_vals, dataset_vals = load_records(paths)
        if not records:
            print(f"SKIP {config}/{task}: no records found")
            continue
        agg = aggregate(records)
        all_out[(config, task)] = dict(agg=agg, leakage_caveat=leak_vals, dataset=dataset_vals,
                                        n_records=len(records))
        print(f"{config:35s} {task:22s} leak={leak_vals} dataset={dataset_vals} n_records={len(records)} regimes={sorted(agg.keys())}")

    json.dump(
        {f"{c}|{t}": {
            "leakage_caveat": sorted(v["leakage_caveat"], key=str),
            "dataset": sorted(v["dataset"]),
            "n_records": v["n_records"],
            "regimes": {
                regime: {
                    "n_train_real": rv["n_train_real"],
                    "n_train_syn": rv["n_train_syn"],
                    "metrics": {m: {"mean": mean, "std": std, "n_seeds": n} for m, (mean, std, n) in rv["metrics"].items()},
                }
                for regime, rv in v["agg"].items()
            },
        } for (c, t), v in all_out.items()},
        open("/tmp/claude-1070/-storage-s25017-AMEX-java-d/c49eb6b2-780b-45aa-8ea3-7578831ae3b3/scratchpad/aggregated.json", "w"),
        indent=2,
    )
    print("\nwrote aggregated.json")


if __name__ == "__main__":
    main()
