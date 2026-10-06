"""Repair a finished GRDM sample's dependency violations and measure what it cost.

    conda run -n tabsyn python -m dep_fidelity.run_repair \
        --dataset berka_split --seeds 0 1 2 --label berka_split_repair --write

For every seed this runs the full check suite before repair, applies the operators
in `repair.py`, re-runs the *whole* suite afterwards (not just the repaired checks --
that is how an operator breaking another constraint would show up), and scores the
distortion each repaired column suffered against the real marginal.

`--write` additionally emits the repaired tables as a `_repaired` variant of the run
directory, which `downstream.run` can consume through the same `variant` mechanism
fix S1 used. That is what makes it possible to ask whether repairing the constraints
moves downstream utility -- a separate question from whether it satisfies the checks,
and one the S1 result says should not be assumed.

Before touching synthetic data the script applies every operator to the *real*
database. A repair that is not near-identity there is mis-specified, and the run
aborts rather than reporting numbers built on it.
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from dep_fidelity.checks import run_all
from dep_fidelity.load import TABLES, load_real, load_synthetic
from dep_fidelity.repair import repair_all

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BASE_DIR_NAME = "synthetic_data_sampled_structure_size_multiplier_1.0"

# The columns any operator may rewrite, and how to score their distortion.
TOUCHED_COLUMNS = (
    ("loan", "loan_date", "continuous"),
    ("card", "issued", "continuous"),
    ("trans", "trans_date", "continuous"),
    ("trans", "balance", "continuous"),
    ("trans", "operation", "discrete"),
    ("trans", "k_symbol", "discrete"),
    ("order", "k_symbol", "discrete"),
)

METRIC_KEYS = ("violation_rate", "median_lag_days", "p10_lag_days", "consistency_rate",
               "median_abs_error", "implication_rate")


def column_distance(real_df, df, col, kind):
    """Distance from the real marginal. Wasserstein in units of the real sd for a
    continuous column, total variation for a discrete one -- both 0 when identical
    and both comparable across columns."""
    if col not in df.columns or col not in real_df.columns:
        return float("nan")
    if kind == "continuous":
        r = real_df[col].dropna().to_numpy(dtype=float)
        s = df[col].dropna().to_numpy(dtype=float)
        if not len(r) or not len(s):
            return float("nan")
        scale = float(np.std(r)) or 1.0
        return float(wasserstein_distance(r, s) / scale)
    cats = sorted(set(real_df[col].dropna().unique()) | set(df[col].dropna().unique()))
    pr = real_df[col].value_counts(normalize=True).reindex(cats, fill_value=0.0)
    ps = df[col].value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return float(0.5 * np.abs(pr.to_numpy() - ps.to_numpy()).sum())


def distortion_table(real_tables, before, after):
    rows = []
    for table, col, kind in TOUCHED_COLUMNS:
        rows.append(dict(
            column=f"{table}.{col}", kind=kind,
            dist_before=column_distance(real_tables[table], before[table], col, kind),
            dist_after=column_distance(real_tables[table], after[table], col, kind),
        ))
    return rows


def fk_integrity_ok(tables):
    """No operator writes an id column; this asserts that stayed true."""
    pairs = [("account", "trans", "account_id"), ("account", "loan", "account_id"),
             ("account", "order", "account_id"), ("account", "disp", "account_id"),
             ("disp", "card", "disp_id")]
    for parent, child, key in pairs:
        if not tables[child][key].isin(set(tables[parent][key])).all():
            return False, f"{child}.{key} -> {parent}"
    return True, ""


def validate_on_real(real_tables, real_epochs, args):
    """Apply every operator to the real database, where it should be near-identity."""
    repaired, _ = repair_all(
        real_tables, real_epochs, real_tables, real_epochs,
        temporal=args.temporal, ledger=args.ledger, implication=args.implication, seed=0,
    )
    rows = distortion_table(real_tables, real_tables, repaired)
    print("\n--- guard: operators applied to REAL data (should be ~0) ---")
    print(f"{'column':22s} {'kind':11s} {'distance from real':>18s}")
    for r in rows:
        print(f"{r['column']:22s} {r['kind']:11s} {r['dist_after']:18.4f}")
    return rows


def summarize(name, results):
    return {r["check"]: {k: r[k] for k in METRIC_KEYS if k in r} for r in results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="berka_split")
    p.add_argument("--exp-name", default="paper_repro")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--real-dir", default=os.path.join(REPO_ROOT, "data/berka/split_train"),
                   help="the real database the generator was trained on")
    p.add_argument("--temporal", default="resample", choices=["clamp", "resample", "shift"])
    p.add_argument("--ledger", default="derive", choices=["derive", "final_only"])
    p.add_argument("--implication", default="real", choices=["real", "full"])
    p.add_argument("--label", default=None)
    p.add_argument("--write", action="store_true",
                   help="emit the repaired tables as a _repaired variant run directory")
    p.add_argument("--variant-suffix", default="_repaired")
    p.add_argument("--skip-guard", action="store_true")
    args = p.parse_args()

    label = args.label or f"{args.dataset}_repair_{args.temporal}_{args.ledger}"
    real_tables, real_epochs = load_real(args.real_dir)
    real_results = run_all(real_tables, real_epochs)

    guard = None if args.skip_guard else validate_on_real(real_tables, real_epochs, args)
    if guard:
        worst = max((r["dist_after"] for r in guard if np.isfinite(r["dist_after"])), default=0.0)
        if worst > 0.25:
            raise SystemExit(
                f"guard failed: an operator moved a real column by {worst:.3f} "
                "(units of real sd / total variation). Fix the operator before trusting a "
                "synthetic result. Re-run with --skip-guard only to inspect."
            )

    seed_records = []
    for seed in args.seeds:
        run_dir = os.path.join(REPO_ROOT, "rdbdiff_workspace", args.dataset,
                               args.exp_name, f"seed_{seed}", BASE_DIR_NAME)
        syn_dir = os.path.join(run_dir, "synthetic_database")
        if not os.path.isdir(syn_dir):
            print(f"[skip] seed {seed}: no sample at {syn_dir}")
            continue

        tables, epochs = load_synthetic(syn_dir, epochs_dir=args.real_dir)
        before = run_all(tables, epochs)
        repaired, repair_stats = repair_all(
            tables, epochs, real_tables, real_epochs,
            temporal=args.temporal, ledger=args.ledger,
            implication=args.implication, seed=seed,
        )
        after = run_all(repaired, epochs)

        ok, why = fk_integrity_ok(repaired)
        if not ok:
            raise SystemExit(f"seed {seed}: repair broke referential integrity ({why})")

        rec = dict(
            seed=seed, source=syn_dir,
            before=summarize("before", before),
            after=summarize("after", after),
            repair_stats=repair_stats,
            distortion=distortion_table(real_tables, tables, repaired),
        )

        if args.write:
            out_run = run_dir + args.variant_suffix
            out_dir = os.path.join(out_run, "synthetic_database")
            os.makedirs(out_dir, exist_ok=True)
            # Copy every table the generator produced, then overwrite the repaired ones,
            # so tables this suite does not model (client, district) survive untouched.
            for fname in os.listdir(syn_dir):
                shutil.copyfile(os.path.join(syn_dir, fname), os.path.join(out_dir, fname))
            for t in TABLES:
                repaired[t].to_csv(os.path.join(out_dir, f"{t}_synthetic.csv"), index=False)
            rec["written_to"] = out_run
            print(f"  wrote {out_run}")

        seed_records.append(rec)
        print(f"[seed {seed}] repaired; "
              f"{sum(s['rows_touched'] for s in repair_stats):,} rows rewritten")

    print_report(real_results, seed_records)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(out_path, "w") as f:
        json.dump(dict(
            label=label, dataset=args.dataset, real_dir=args.real_dir,
            strategies=dict(temporal=args.temporal, ledger=args.ledger,
                            implication=args.implication),
            real=real_results, real_guard=guard, seeds=seed_records,
        ), f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


def print_report(real_results, seed_records):
    if not seed_records:
        print("no seeds processed")
        return

    print("\n=== constraint satisfaction: real vs synthetic vs repaired ===")
    print(f"{'check':46s} {'metric':20s} {'real':>10s} {'synth':>10s} {'repaired':>10s}")
    print("-" * 100)
    for real in real_results:
        if "error" in real:
            continue
        name = real["check"]
        for key in METRIC_KEYS:
            if key not in real:
                continue
            b = [s["before"].get(name, {}).get(key) for s in seed_records]
            a = [s["after"].get(name, {}).get(key) for s in seed_records]
            b = [v for v in b if v is not None and np.isfinite(v)]
            a = [v for v in a if v is not None and np.isfinite(v)]
            print(f"{name:46s} {key:20s} {real[key]:10.4f} "
                  f"{np.mean(b) if b else float('nan'):10.4f} "
                  f"{np.mean(a) if a else float('nan'):10.4f}")

    print("\n=== cost: distance of each repaired column from the real marginal ===")
    print("(Wasserstein / real sd for continuous, total variation for discrete; 0 = identical)")
    print(f"{'column':22s} {'kind':11s} {'before':>10s} {'after':>10s} {'change':>10s}")
    print("-" * 68)
    for i, row in enumerate(seed_records[0]["distortion"]):
        bs = [s["distortion"][i]["dist_before"] for s in seed_records]
        as_ = [s["distortion"][i]["dist_after"] for s in seed_records]
        bs = [v for v in bs if np.isfinite(v)]
        as_ = [v for v in as_ if np.isfinite(v)]
        mb, ma = (np.mean(bs) if bs else np.nan), (np.mean(as_) if as_ else np.nan)
        arrow = "same" if np.isclose(ma, mb) else ("better" if ma < mb else "worse")
        print(f"{row['column']:22s} {row['kind']:11s} {mb:10.4f} {ma:10.4f} {ma - mb:+10.4f}  {arrow}")

    print("\n=== rows rewritten (mean per seed) ===")
    stats = pd.DataFrame([s for rec in seed_records for s in rec["repair_stats"]])
    agg = stats.groupby(["check", "operator"], sort=False)["rows_touched"].mean()
    for (check, op), v in agg.items():
        print(f"{check:46s} {op:22s} {v:12,.0f}")


if __name__ == "__main__":
    main()
