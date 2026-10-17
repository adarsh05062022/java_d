import json, csv, statistics, os

MAIN = "/storage/s25017/AMEX/java_d"
WT = "/storage/s25017/AMEX/java_d-calagg-ft"
EXP = f"{WT}/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE"

REGIMES = ["real", "synth", "mixed_25", "mixed_50", "mixed",
           "real_50", "blend_50", "real_25", "blend_25", "real_75", "blend_75"]
PREVIOUSLY_SCORED_NEW_ARMS = {"synth", "mixed", "blend_50"}
NEWLY_RUN = {"real_25", "real_50", "real_75", "blend_25", "blend_75", "mixed_25", "mixed_50"}
REUSED_REAL = {"real"}

TASKS = [("california_inctot", "r2", "income R2"),
         ("california_empstat", "f1_macro", "employment macro-F1")]

CONFIGS = [
    ("baseline", "Original GRDM", "california",
     [f"{MAIN}/results/paper_repro/downstream/california__{{task}}.json",
      f"{MAIN}/results/paper_repro/downstream/california__{{task}}_seeds12.json"]),
    ("control", "Control", "california_ft_control",
     [f"{EXP}/downstream/california_ft_control__{{task}}.json",
      f"{EXP}/downstream/california_ft_control__{{task}}_missing.json"]),
    ("agg", "Aggregate-conditioned", "california_agg_ft",
     [f"{EXP}/downstream/california_agg_ft__{{task}}.json",
      f"{EXP}/downstream/california_agg_ft__{{task}}_missing.json"]),
    ("agg_label_free", "Label-free aggregate-conditioned", "california_agg_ft_label_free",
     [f"{EXP}/downstream/california_agg_ft_label_free__{{task}}.json",
      f"{EXP}/downstream/california_agg_ft_label_free__{{task}}_missing.json"]),
]

def load_records(paths, task):
    recs = []
    seen_real_xgb = set()
    for p in paths:
        p = p.format(task=task)
        if not os.path.exists(p):
            continue
        data = json.load(open(p))
        for r in data["records"]:
            if r["regime"] == "real":
                key = r["xgb_seed"]
                if key in seen_real_xgb:
                    continue  # baseline's two files both recompute "real"; keep first copy only
                seen_real_xgb.add(key)
            recs.append(r)
    return recs

def aggregate(records, regime, metric_key):
    rows = [r for r in records if r["regime"] == regime]
    if not rows:
        return None
    if regime == "real":
        vals = [r["metrics"][metric_key] for r in rows]
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return dict(mean=mean, std=std, n=len(vals), seed_kind="xgb_seeds(real data has no GRDM dependence)")
    seeds = sorted(set(r["grdm_seed"] for r in rows))
    seed_means = []
    for s in seeds:
        vals = [r["metrics"][metric_key] for r in rows if r["grdm_seed"] == s]
        seed_means.append(sum(vals) / len(vals))
    mean = sum(seed_means) / len(seed_means)
    std = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
    return dict(mean=mean, std=std, n=len(seed_means), seed_kind="GRDM seeds")

def status_for(regime, config_name):
    if config_name == "baseline":
        return "previously evaluated (paper_repro baseline)"
    if regime in REUSED_REAL:
        return "reused from baseline (real regime has no GRDM dependence; verified arm-invariant in code)"
    if regime in PREVIOUSLY_SCORED_NEW_ARMS:
        return "previously evaluated"
    if regime in NEWLY_RUN:
        return "newly evaluated (this task)"
    return "unknown"

rows_out = []
for cfg_key, cfg_label, dataset_name, path_templates in CONFIGS:
    for task_file, metric_key, metric_label in TASKS:
        records = load_records(path_templates, task_file)
        for regime in REGIMES:
            if cfg_key != "baseline" and regime == "real":
                # pull "real" from baseline directly instead of the arm's own (possibly absent) copy
                base_paths = CONFIGS[0][3]
                base_records = load_records(base_paths, task_file)
                agg = aggregate(base_records, "real", metric_key)
            else:
                agg = aggregate(records, regime, metric_key)
            status = status_for(regime, cfg_key)
            if agg is None:
                rows_out.append(dict(regime=regime, configuration=cfg_label, metric=metric_label,
                                      mean="", std="", seeds="", status="PENDING (background run in progress)"))
            else:
                rows_out.append(dict(regime=regime, configuration=cfg_label, metric=metric_label,
                                      mean=round(agg["mean"], 4), std=round(agg["std"], 4),
                                      seeds=agg["n"], status=status))

os.makedirs(f"{EXP}/eleven_regime_addendum", exist_ok=True)
with open(f"/storage/s25017/AMEX/java_d/results/stage2/table5/TABLE5_eleven_regimes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["regime", "configuration", "metric", "mean", "std", "seeds", "status"])
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

md = ["| Evaluation regime | Configuration | Metric | Mean | Standard deviation | Seeds | Status |",
      "|---|---|---|---|---|---|---|"]
for r in rows_out:
    md.append(f"| {r['regime']} | {r['configuration']} | {r['metric']} | {r['mean']} | {r['std']} | {r['seeds']} | {r['status']} |")
with open(f"/storage/s25017/AMEX/java_d/results/stage2/table5/TABLE5_eleven_regimes.md", "w") as f:
    f.write("\n".join(md) + "\n")

n_pending = sum(1 for r in rows_out if "PENDING" in r["status"])
n_total = len(rows_out)
print(f"wrote {n_total} rows, {n_pending} still pending ({n_total - n_pending} resolved)")
