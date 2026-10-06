"""Aggregate downstream result JSON into the markdown tables in RESULTS_downstream.md.

    conda run --no-capture-output -n tabsyn python -m downstream.report
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT
from downstream.tasks import TASKS

RESULTS_DIR = os.path.join(REPO_ROOT, "downstream", "results")
OUT_PATH = os.path.join(REPO_ROOT, "docs", "downstream", "RESULTS_downstream.md")

REGIME_LABEL = {
    "real": "real only",
    "synth": "synthetic only",
    "mixed_25": "real + 25% syn",
    "mixed_50": "real + 50% syn",
    "mixed": "real + 100% syn",
}
REGIME_ORDER = ["real", "synth", "mixed_25", "mixed_50", "mixed"]

# Metrics where a higher number is worse, so the delta-vs-real column flips sign.
LOWER_IS_BETTER = {"rmse", "mae"}

# Report order: flagship task first per dataset, not alphabetical.
TASK_ORDER = [
    "california_inctot",
    "california_empstat",
    "berka_loan_default",
    "berka_card_ownership",
]
# berka_split is the canonical leakage-free protocol, so it precedes the interim berka run.
DATASET_ORDER = ["california", "berka_split", "berka"]

# Fix experiments (docs/REPRODUCTION.md §9). Each changes exactly one thing against its base run.
FIX_LABELS = {
    "berka_split": "baseline (n_hops=1, ordinal codes)",
    "berka_split_s1": "S1 · distribution-matched decoding",
    "berka_split_s2": "S2 · n_hops=2 (deeper conditioning)",
    "berka_split_s3": "S3 · balanced seed sampling",
    "berka_split_s4": "S4 · one-hot categoricals",
    "berka_split_s5": "S5 · n_hops=2 + balanced seeds",
    "berka_split_s6": "S6 · n_hops=3 (depth only)",
    "berka_split_s7": "S7 · n_hops=3 + balanced seeds",
    # S8-S11 change the denoiser itself rather than its inputs or its sampling; each is
    # layered on S5, the best configuration the earlier fixes produced.
    "berka_split_s8": "S8 · multi-statistic aggregation",
    "berka_split_s9": "S9 · residual GNN",
    "berka_split_s10": "S10 · exact cardinality features",
    "berka_split_s11": "S11 · self-conditioning",
    "berka_split_s12": "S12 · S8+S9+S10 (architecture package)",
    "berka_split_s13": "S13 · S8+S9+S10+S11",
    # Date representation and the combined-direction sweep (2026-09).
    "berka_split_reparam": "reparam · log1p-lag child dates",
    "berka_split_reparam_s12": "reparam + S12 architecture",
    "berka_split_reparam_s10": "reparam + exact degree features",
    "berka_split_s12_wide": "S12 + wide hop-2 fan-out (-1 64)",
    "berka_split_discrete": "discrete/multinomial categoricals",
    # Post-hoc constraint repair of the baseline samples (dep_fidelity/run_repair.py).
    "berka_split_repair": "repair · all eight dependency checks restored",
    "berka_split_repair_finalonly": "repair (endpoints) · without the balance rewrite",
    # Task 6: six child aggregates added as parent columns, published architecture.
    "berka_split_agg": "agg · six child aggregates as parent columns",
    "california": "baseline",
    "california_s2": "S2 · n_hops=2",
    "california_s3": "S3 · balanced seed sampling",
    "california_s4": "S4 · one-hot categoricals",
    "california_s5": "S5 · n_hops=2 + balanced seeds",
    "california_s7": "S7 · n_hops=3 + balanced seeds",
    "california_s12": "S12 · S8+S9+S10 (architecture package)",
    "california_s13": "S13 · S8+S9+S10+S11",
    "california_discrete": "discrete/multinomial categoricals",
}
FIX_ORDER = list(FIX_LABELS)


def fix_comparison(results):
    """Per task: baseline vs each fix on the primary metric, with signal retained."""
    lines = []
    tasks = []
    for key in results:
        t = key.split("__")[1]
        if t not in tasks:
            tasks.append(t)
    for task_name in [t for t in TASK_ORDER if t in tasks]:
        task = TASKS[task_name]
        present = [
            d for d in FIX_ORDER if f"{d}__{task_name}" in results
        ]
        if len(present) < 2:
            continue
        floor = 0.5 if task.primary == "roc_auc" else 0.0
        lines.append(f"### {task_name} — `{task.primary}`")
        lines.append("")
        lines.append("| variant | seeds | real | synthetic only | real + 100% syn | retained | vs baseline |")
        lines.append("|---|---|---|---|---|---|---|")
        base_syn = None
        for ds in present:
            recs = results[f"{ds}__{task_name}"]["records"]
            n_seeds = len({r["grdm_seed"] for r in recs if r["grdm_seed"] is not None})
            real = summarise(recs, "real", task.primary)[0]
            syn = summarise(recs, "synth", task.primary)
            mix = summarise(recs, "mixed", task.primary)
            retained = (syn[0] - floor) / (real - floor)
            if base_syn is None:
                base_syn = syn[0]
                delta = "—"
            else:
                delta = f"{syn[0] - base_syn:+.4f}"
            mix_cell = f"{mix[0]:.4f}" if mix else "—"  # repair variants score real + synth only
            lines.append(
                f"| {FIX_LABELS[ds]} | {n_seeds} | {real:.4f} | {syn[0]:.4f} ± {syn[1]:.4f} | "
                f"{mix_cell} | {retained:.1%} | {delta} |"
            )
        lines.append("")
    return lines


def sort_key(key):
    dataset, task = key.split("__")
    d = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
    t = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
    return (d, t)


def load_results(pattern="*.json"):
    """Load `<dataset>__<task>.json` result files.

    Other artefacts live here too (`diagnosis.json`), so require the dataset/task separator
    rather than taking every json in the directory.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        key = os.path.basename(path)[: -len(".json")]
        if "__" not in key:
            continue
        with open(path) as f:
            out[key] = json.load(f)
    return out


def summarise(records, regime, metric):
    vals = [r["metrics"][metric] for r in records if r["regime"] == regime]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def generator_spread(records, regime, metric):
    """Std of the per-GRDM-seed means: generator variance with fit noise averaged out."""
    seeds = sorted({r["grdm_seed"] for r in records if r["regime"] == regime})
    if len(seeds) < 2 or seeds == [None]:
        return None
    means = [
        np.mean([r["metrics"][metric] for r in records if r["regime"] == regime and r["grdm_seed"] == s])
        for s in seeds
    ]
    return float(np.std(means))


def task_table(payload, task):
    records = payload["records"]
    lines = []
    header = ["regime", "n train"] + task.metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    baseline = {m: summarise(records, "real", m)[0] for m in task.metrics}
    for regime in REGIME_ORDER:
        first = next((r for r in records if r["regime"] == regime), None)
        if first is None:
            continue
        cells = [REGIME_LABEL[regime], f"{first['n_train']:,}"]
        for metric in task.metrics:
            mean, std, _ = summarise(records, regime, metric)
            cell = f"{mean:.4f} ± {std:.4f}"
            if regime != "real":
                delta = mean - baseline[metric]
                if metric in LOWER_IS_BETTER:
                    delta = -delta
                cell += f" ({delta:+.4f})"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    notes = []
    spread = generator_spread(records, "synth", task.primary)
    if spread is not None:
        _, fit_std, _ = summarise(records, "synth", task.primary)
        notes.append(
            f"Synthetic-only {task.primary}: spread across GRDM seeds ±{spread:.4f}, "
            f"across all repeats ±{fit_std:.4f}."
        )
    ci = next(
        (r["ci95"][task.primary] for r in records if r["regime"] == "real" and "ci95" in r), None
    )
    if ci and ci[0] is not None:
        notes.append(
            f"Real-only {task.primary} bootstrap 95% CI over the test set: "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]."
        )
    return lines, notes


def berka_label_diagnostic(grdm_dataset="berka", seeds=(0, 1, 2), source_rel="data/berka/preprocessed"):
    """Real vs synthetic prevalence of the two Berka labels.

    This is the mechanism behind the synthetic-only collapse, and it is invisible to the
    fidelity metrics: Column Shapes scored 96.89 on the same tables. `source_rel` selects the
    real database the generator was trained on, so the same function serves the full-data run
    and the leakage-free `berka_split` rerun.
    """
    from downstream.data import (
        coerce_discrete,
        grdm_run_dir,
        load_db,
        load_domains,
        load_meta,
        load_synthetic_db,
    )
    from downstream.splits import DEFAULT_STATUS

    source = os.path.join(REPO_ROOT, source_rel)
    meta = load_meta(source)
    domains = load_domains(source, list(meta["tables"]))
    real = coerce_discrete(load_db(source), domains)

    def counts(tables):
        loan = tables["loan"]
        by_status = {int(k): int(v) for k, v in loan["status"].value_counts().items()}
        return {
            "loan_n": len(loan),
            "status": {c: by_status.get(c, 0) for c in range(4)},
            "default_rate": float(loan["status"].isin(DEFAULT_STATUS).mean()),
            "card_rate": float(
                tables["disp"]["disp_id"].isin(set(tables["card"]["disp_id"])).mean()
            ),
        }

    rows = [("real", counts(real))]
    for s in seeds:
        syn = coerce_discrete(
            load_synthetic_db(grdm_run_dir(grdm_dataset, s), list(meta["tables"])), domains
        )
        rows.append((f"synthetic seed {s}", counts(syn)))

    lines = [
        "| source | loan rows | A (repaid) | B (unpaid) | C (running ok) | D (in debt) | "
        "default rate | card rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, c in rows:
        lines.append(
            f"| {label} | {c['loan_n']} | {c['status'][0]} | {c['status'][1]} | "
            f"{c['status'][2]} | {c['status'][3]} | {c['default_rate']:.3f} | "
            f"{c['card_rate']:.3f} |"
        )
    return lines


FINDINGS = [
    "### What the numbers say",
    "",
    "**The two databases behave completely differently, and the fidelity metrics did not",
    "predict the size of the gap.** California synthetic data is a near-drop-in replacement for",
    "real data; Berka synthetic data is close to useless on its own. GRDM\u2019s reproduced",
    "fidelity scores differ far less dramatically between the two (Column Shapes 98.60 vs 96.89,",
    "1-HOP 96.48 vs 91.43), so a few points of fidelity separate regimes that downstream utility",
    "separates by an order of magnitude. California is 2 tables and 1 relationship; Berka is 8",
    "tables and 8 foreign-key links with a multi-parent `disp`.",
    "",
    "_The prose in this section was last checked against the tables on 5 Oct 2026, after the",
    "loan-default window was corrected (`downstream/tasks.py`); it is not regenerated with them._",
    "",
    "**The leakage-free rerun confirms the interim Berka result rather than overturning it.**",
    "Retraining the generator on `split_train` alone moves synthetic-only AUROC from 0.5316 to",
    "0.5143 on loan default and from 0.6094 to 0.6021 on card ownership -- both inside the",
    "seed spread. Training on the test rows was not what made this synthetic data look bad; the",
    "generator simply does not carry these relationships. Note the real-only baselines are",
    "identical across the two protocols by construction: both evaluate the same fixed split, and",
    "only what the generator was allowed to see differs.",
    "",
    "**Mixing is safe but not useful.** Adding synthetic rows on top of the full real training",
    "set moves every primary metric by at most 0.011 in either direction, on both databases and",
    "at every mixing ratio. There is no augmentation win. On loan default every measure falls a",
    "little at the full ratio (AUROC -0.011, macro-F1 -0.037, balanced accuracy -0.048).",
    "",
    "**Berka fails through the dependency between tables, and only part of that shows in the",
    "marginals.** For loan default the marginal breaks -- the default rate falls from 0.112 to",
    "about 0.075 and class B collapses from 24 real rows to 1-10 synthetic -- but restoring it",
    "exactly (S1) lifts loan only from 3% to 14% of the real signal. For card ownership the",
    "marginal is nearly exact (0.168 real vs 0.160-0.167 synthetic) and the task still collapses",
    "from 0.874 to 0.602 AUROC, because what is lost is the *cross-table dependency* between",
    "account behaviour and card ownership, not the rate. Matching per-column distributions is",
    "therefore not sufficient for downstream utility, which is precisely the claim a",
    "column-shape metric cannot test.",
    "",
    "**Watch accuracy.** On card ownership the synthetic-only model matches the real model on",
    "plain accuracy (0.836 vs 0.836) while sitting at chance on balanced accuracy (0.500). On loan",
    "default accuracy does fall (0.846 vs 0.926), but it still rates a model whose balanced",
    "accuracy is 0.507 as 85% correct. Both models predict the majority class almost everywhere.",
    "",
    "**Root cause and fixes.** See `docs/downstream/DIAGNOSIS.md` and `docs/downstream/FIXES.md`",
    "(August) and `reports/grdm-final-report.html` (current). The best three-seed results are",
    "reparam + S10 for loan default (43.6% of the real signal), S5 for card ownership (50.4%)",
    "and agg for both together (35.8% and 48.7%); none reaches 50% on both tasks.",
]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    results = load_results()
    # other experiments drop their own <dataset>__<name>.json here (blend ratios, cross-domain
    # TSTR, feature comparison); keep only the four canonical downstream tasks.
    results = {k: v for k, v in results.items() if k.split("__", 1)[1] in TASKS}
    if not results:
        raise SystemExit(f"no result files in {RESULTS_DIR}")

    doc = [
        "# Downstream utility of GRDM synthetic data (XGBoost)",
        "",
        "Generated by `downstream/report.py`. Every regime is trained with identical, untuned",
        "XGBoost hyperparameters and evaluated on the **same real held-out test set**. The",
        "synthetic-only regime is subsampled to the real training row count, so `real` vs",
        "`synthetic only` compares data *quality*; the mixed rows then add quantity on top of the",
        "full real set. Parenthesised numbers are the change against the real-only baseline,",
        "sign-corrected so positive always means better.",
        "",
    ]

    doc.append("## Summary \u2014 primary metric per task")
    doc.append("")
    doc.append(
        "| dataset | task | metric | real | synthetic only | real + 100% syn | delta synth | "
        "synth signal retained |"
    )
    doc.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(results, key=sort_key):
        dataset, task_name = key.split("__")
        task = TASKS[task_name]
        recs = results[key]["records"]
        # A partial-regime run (downstream.run --regimes real synth) omits the mixed sweep,
        # so a cell may be absent; render it as "n/a" rather than crashing the whole report.
        real_s = summarise(recs, "real", task.primary)
        syn_s = summarise(recs, "synth", task.primary)
        mixed_s = summarise(recs, "mixed", task.primary)
        real_mean = real_s[0] if real_s else None
        syn_mean = syn_s[0] if syn_s else None
        mixed_mean = mixed_s[0] if mixed_s else None
        flag = " (!)" if results[key].get("leakage_caveat") else ""
        # Fraction of the real baseline's usable signal that survives training on synthetic
        # data. AUROC is measured above its 0.5 chance floor; R^2 and macro-F1 already have a
        # meaningful zero, so they are compared as a plain ratio.
        floor = 0.5 if task.primary == "roc_auc" else 0.0
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        if real_mean is not None and syn_mean is not None and real_mean != floor:
            retained = (syn_mean - floor) / (real_mean - floor)
            retained_cell = f"{retained:.1%}"
            delta_cell = f"{syn_mean - real_mean:+.4f}"
        else:
            retained_cell = "n/a"
            delta_cell = "n/a"
        doc.append(
            f"| {dataset}{flag} | {task_name} | `{task.primary}` | {fmt(real_mean)} | "
            f"{fmt(syn_mean)} | {fmt(mixed_mean)} | {delta_cell} | "
            f"{retained_cell} |"
        )
    doc.append("")
    doc.append(
        "(!) = generator trained on the rows used here as the test set; see the task caveat. "
        "*Signal retained* is measured above the metric's floor (0.5 for AUROC, 0 otherwise)."
    )
    doc.append("")
    doc.extend(FINDINGS)
    doc.append("")

    for key in sorted(results, key=sort_key):
        dataset, task_name = key.split("__")
        task = TASKS[task_name]
        payload = results[key]
        doc.append(f"## {dataset} — {task_name}")
        doc.append("")
        doc.append(f"*{task.kind}, target: {task.target}. Primary metric: `{task.primary}`.*")
        doc.append("")
        if payload.get("leakage_caveat"):
            doc.append(
                "> **Leakage caveat.** This generator was trained on the whole database, "
                "including the rows used here as the test set, so the synthetic and mixed "
                "numbers are optimistically biased. See the `berka_split` tables for the "
                "leakage-free rerun."
            )
            doc.append("")
        lines, notes = task_table(payload, task)
        doc.extend(lines)
        doc.append("")
        for n in notes:
            doc.append(f"- {n}")
        doc.append("")

    fix_lines = fix_comparison(results)
    if fix_lines:
        doc.append("## Fix experiments")
        doc.append("")
        doc.append(
            "Each variant changes exactly one thing against its baseline and is evaluated on the "
            "same fixed split and test set, so a delta is attributable to that change alone. "
            "Root causes and the evidence behind them are in `DIAGNOSIS.md`; per-fix design is in "
            "`FIXES.md`."
        )
        doc.append("")
        doc.extend(fix_lines)

    if any(k.startswith("berka") for k in results):
        doc.append("## Label marginals in the synthetic data")
        doc.append("")
        doc.append(
            "The fidelity metrics do not see this. Berka Column Shapes scored 96.89 on the very "
            "tables below, while `loan.status` class B -- the rarer of the two default outcomes "
            "-- is all but erased in the generated data. It is a symptom rather than the cause of "
            "the collapse: restoring the marginal exactly (S1) recovers little loan signal, and "
            "card ownership collapses with its marginal intact."
        )
        doc.append("")
        doc.append("**Full-data generator** (interim protocol):")
        doc.append("")
        doc.extend(berka_label_diagnostic())
        doc.append("")
        if any(k.startswith("berka_split") for k in results):
            doc.append("**Leakage-free generator** trained on `split_train` only:")
            doc.append("")
            doc.extend(
                berka_label_diagnostic(
                    grdm_dataset="berka_split", source_rel="data/berka/split_train"
                )
            )
            doc.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {args.out}")
    print("\n".join(doc))


if __name__ == "__main__":
    main()
