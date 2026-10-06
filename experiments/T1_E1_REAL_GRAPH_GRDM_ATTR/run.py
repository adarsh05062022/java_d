"""T1_E1_REAL_GRAPH_GRDM_ATTR: how much downstream signal survives when only the attributes are
GRDM's and the relational graph and the target stay real?

    A  REAL graph + REAL attributes + REAL target   (the project's real-only baseline)
    B  REAL graph + GRDM attributes + REAL target   (T1_E1)

Both are scored by `downstream.run.run_task`, unchanged: same feature builders, same
`align_feature_dtypes`, same fixed XGBoost hyperparameters, same XGBoost seeds, same real
held-out test set, same metrics and bootstrap CIs. The only thing swapped is the function
`run_task` calls to load its "synthetic" tables (`downstream.run.load_syn_tables`), which here
returns the hybrid database instead of GRDM's sampled-structure output. The hybrid has exactly
as many rows as the real training set, so `run_task`'s subsample-to-real-size step is a no-op.

Two supplementary references run through the same function and are reported separately from
the A-vs-B comparison (`--no-references` skips them):

    REF_SHUFFLED_ATTR  REAL graph + real attribute rows shuffled across nodes of the same table
                       + REAL target. Per-table attribute distributions exact, every link
                       between a node's attributes and its graph position cut. What the graph
                       alone carries through this pipeline.
    REF_GRDM_TARGET    REAL graph + GRDM attributes + GRDM's own generated target (only for
                       tasks whose target is an attribute).

    conda run --no-capture-output -n tabsyn python experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/run.py \
        --dataset berka_split
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import downstream.run as dsrun  # noqa: E402
from downstream.data import (  # noqa: E402
    assert_fk_integrity,
    coerce_discrete,
    load_db,
    load_domains,
    load_meta,
    read_json,
)
from downstream.evaluate import N_JOBS, XGB_PARAMS  # noqa: E402
from downstream.tasks import TASKS  # noqa: E402
from hybrid import (  # noqa: E402
    TASK_TARGETS,
    attribute_statistics,
    build_hybrid,
    build_row_permuted,
    graph_fingerprint,
    load_grdm_output,
    sanity_checks,
)

EXP = "T1_E1_REAL_GRAPH_GRDM_ATTR"
GRDM_EXP_NAME = "T1_E1_real_structure"
GRDM_SEEDS = [0, 1, 2]  # the generator seeds of every existing downstream evaluation

CONDITION = {
    "real": "A_REAL_GRAPH_REAL_ATTR",
    "T1_E1": "B_REAL_GRAPH_GRDM_ATTR",
    "shuffled": "REF_SHUFFLED_ATTR",
    "grdm_target": "REF_GRDM_TARGET",
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def grdm_output_dir(root, seed):
    return os.path.join(root, f"seed_{seed}", "synthetic_data_real_structure", "synthetic_database")


def run_with_tables(task, cfg, tables_by_seed, seeds, xgb_seeds, domains, meta, real_tables,
                    test_tables, regimes):
    """`downstream.run.run_task`, with the given per-seed databases standing in for GRDM's."""

    def load(_cfg, seed, _domains, _meta):
        tables = {t: df.copy() for t, df in tables_by_seed[seed].items()}
        assert_fk_integrity(tables, _meta, label=f"hybrid seed {seed}")  # as load_syn_tables
        return coerce_discrete(tables, _domains)

    original = dsrun.load_syn_tables
    dsrun.load_syn_tables = load
    try:
        return dsrun.run_task(task, cfg, seeds, xgb_seeds, domains, meta, real_tables,
                              test_tables, regimes=regimes)
    finally:
        dsrun.load_syn_tables = original


def provenance(cfg, grdm_root, seeds, dry_run):
    grdm_cfg_path = os.path.join(REPO, "configs", f"{cfg['grdm_dataset']}_fastsample.json")
    grdm_cfg = read_json(grdm_cfg_path)
    prov = {
        "grdm_config_path": os.path.relpath(grdm_cfg_path, REPO),
        "grdm_config": grdm_cfg,
        "grdm_trained_on": grdm_cfg["data"]["data_path"],
        "grdm_trained_on_same_dir_as_real_baseline": os.path.samefile(
            os.path.join(REPO, grdm_cfg["data"]["data_path"]), cfg["real_train"]
        ),
        "grdm_main_py_args": "--exp_name T1_E1_real_structure --seed <N> --fixed_structure "
                             "--no_wandb --max_hours 20 (all other arguments at their defaults, "
                             "as in the paper_repro baseline: n_hops 1, num_neighbors -1, "
                             "categorical_encoding ordinal, categorical_diffusion gaussian, "
                             "categorical_decoding round)",
        "grdm_output_root": os.path.relpath(grdm_root, REPO) if not dry_run else grdm_root,
        "per_seed": {},
    }
    for s in seeds:
        entry = {"synthetic_database": grdm_output_dir(grdm_root, s)}
        if not dry_run:
            src = os.path.join(REPO, "rdbdiff_workspace", cfg["grdm_dataset"], "paper_repro",
                               f"seed_{s}")
            dst = os.path.join(grdm_root, f"seed_{s}")
            entry["model_ema_md5_paper_repro"] = md5(os.path.join(src, "model_checkpoint", "model_ema.pt"))
            entry["model_ema_md5_used"] = md5(os.path.join(dst, "model_checkpoint", "model_ema.pt"))
            entry["weights_identical_to_baseline"] = (
                entry["model_ema_md5_paper_repro"] == entry["model_ema_md5_used"]
            )
            norms = sorted(os.listdir(os.path.join(src, "normalizers")))
            entry["normalizers_identical_to_baseline"] = all(
                md5(os.path.join(src, "normalizers", n)) == md5(os.path.join(dst, "normalizers", n))
                for n in norms
            )
            entry["trained_in_this_experiment"] = False
            entry["sampling_log"] = os.path.relpath(
                os.path.join(REPO, "results", EXP, "logs",
                             f"sample_{cfg['grdm_dataset']}_seed{s}.log"), REPO)
        prov["per_seed"][s] = entry
    return prov


def code_provenance():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"unavailable: {e}"

    files = ["downstream/run.py", "downstream/tasks.py", "downstream/evaluate.py",
             "downstream/data.py", "downstream/splits.py", "main.py", "rdbdiff/sampler.py",
             "rdbdiff/model.py", "experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/hybrid.py",
             "experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/run.py",
             "experiments/T1_E1_REAL_GRAPH_GRDM_ATTR/sample_real_structure.sh"]
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_tracked_files_modified": git("status", "--porcelain", "--untracked-files=no").count("\n") + 1,
        "md5": {f: md5(os.path.join(REPO, f)) for f in files},
        "note": "downstream/*.py and main.py were used unchanged; md5s pin the exact versions",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["berka_split", "california"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--grdm-seeds", nargs="*", type=int, default=GRDM_SEEDS)
    parser.add_argument("--xgb-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--grdm-root", default=None,
                        help="override <root>/seed_<N>/synthetic_data_real_structure/"
                             "synthetic_database (dry runs only)")
    parser.add_argument("--out-dir", default=os.path.join(REPO, "results", EXP))
    parser.add_argument("--no-references", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="do not abort on failed checks (a fake GRDM output equal to the "
                             "real data is expected to fail check 6)")
    args = parser.parse_args()

    t_start = time.time()
    ds = args.dataset
    cfg = dict(dsrun.DATASETS[ds])
    cfg["_name"] = ds
    assert not cfg["leakage"], f"{ds} is a leakage-caveat dataset"
    grdm_root = args.grdm_root or os.path.join(REPO, "rdbdiff_workspace", cfg["grdm_dataset"], GRDM_EXP_NAME)
    seeds = args.grdm_seeds
    # identical rule to downstream/run.py
    xgb_seeds = args.xgb_seeds or ([0, 1, 2, 3, 4] if "berka" in ds else [0, 1, 2])
    tasks = args.tasks or cfg["tasks"]
    out = args.out_dir

    grdm_cfg = read_json(os.path.join(REPO, "configs", f"{cfg['grdm_dataset']}_fastsample.json"))
    dims = grdm_cfg["data"]["dimension_tables"] or []

    # --- data, loaded exactly as downstream/run.py::main loads it --------------------------
    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))
    real_tables = dsrun.load_tables(cfg["real_train"], domains, meta, "real train")
    test_tables = dsrun.load_tables(cfg["test"], domains, meta, "test")
    root = list(meta["tables"])[0]
    for table in meta["tables"]:
        key = f"{table}_id"
        if table == root and ds.startswith("berka"):
            continue  # district is a dimension table, deliberately shared
        overlap = set(real_tables[table][key]) & set(test_tables[table][key])
        if overlap:
            raise AssertionError(f"{table}: {len(overlap)} ids in both train and test")
    real_raw = load_db(cfg["real_train"])  # untouched CSV values: the hybrid's key source

    print(f"[{EXP}] dataset={ds} tasks={tasks} grdm_seeds={seeds} xgb_seeds={xgb_seeds}")
    print(f"  real_train={cfg['real_train']}\n  test={cfg['test']}\n  grdm_root={grdm_root}")
    print(f"  dimension tables (held real by GRDM): {dims}")

    grdm = {s: load_grdm_output(grdm_output_dir(grdm_root, s), list(meta["tables"])) for s in seeds}

    config = {
        "experiment": EXP,
        "dataset": ds,
        "conditions": {
            CONDITION["real"]: "REAL graph + REAL attributes + REAL target (downstream regime 'real')",
            CONDITION["T1_E1"]: "REAL graph + GRDM attributes + REAL target (downstream regime 'synth' fed the hybrid)",
            CONDITION["shuffled"]: "supplementary: REAL graph + real attribute rows permuted within table + REAL target",
            CONDITION["grdm_target"]: "supplementary: REAL graph + GRDM attributes + GRDM target",
        },
        "tasks": {t: {"kind": TASKS[t].kind, "target": TASKS[t].target,
                      "primary": TASKS[t].primary, "metrics": TASKS[t].metrics,
                      "target_location": TASK_TARGETS[t] or "graph structure"} for t in tasks},
        "grdm_seeds": seeds,
        "xgb_seeds": xgb_seeds,
        "real_train": os.path.relpath(cfg["real_train"], REPO),
        "test": cfg["test"],
        "domains": os.path.relpath(cfg["domains"], REPO),
        "dimension_tables": dims,
        "downstream": {
            "pipeline": "downstream.run.run_task (unchanged)",
            "regimes_used": ["real", "synth"],
            "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
            "xgb_n_jobs": N_JOBS,
            "binary_class_weight": "scale_pos_weight = n_neg / n_pos (downstream.evaluate.fit_predict)",
            "early_stopping": None,
            "bootstrap_ci": "1000 test-set resamples, first XGBoost seed of each condition/seed",
        },
        "hybrid_mapping": (
            "GRDM row i of table t <- real CSV row i of table t (relbench reindexes pk to row "
            "position; verified per table: GRDM pk == arange(n) and GRDM fk == real fk mapped "
            "through the parent's pk->position map). Real key values, row order and column "
            "order are kept; non-key columns of generated tables are replaced; dimension tables "
            "stay real; the task's target column is restored to real."
        ),
        "grdm": provenance(cfg, grdm_root, seeds, args.dry_run),
        "code": code_provenance(),
        "dry_run": args.dry_run,
    }
    write_json(config, os.path.join(out, "config", f"{ds}.json"))

    all_ok = True
    for task_name in tasks:
        task = TASKS[task_name]
        target = TASK_TARGETS[task_name]
        print(f"\n######## {task_name}: target {target or 'graph structure'}")

        hybrids, sanity = {}, {}
        for s in seeds:
            hyb, mapping = build_hybrid(real_raw, grdm[s], meta, dims, target)
            hyb_c = coerce_discrete(hyb, domains)  # what the pipeline will consume
            ok, rep = sanity_checks(real_tables, hyb_c, meta, task, domains, test_tables, dims,
                                    target, task.build)
            rep["0_row_mapping"] = mapping
            sanity[s] = {"all_passed": ok, "checks": rep}
            write_json(sanity[s], os.path.join(out, "sanity", f"{ds}__{task_name}__grdm_seed{s}.json"))
            write_json(attribute_statistics(real_tables, hyb_c, meta, domains, dims, target),
                       os.path.join(out, "dataset_stats", f"{ds}__{task_name}__grdm_seed{s}.json"))
            print(f"  [seed {s}] sanity: " + "  ".join(
                f"{k.split('_')[0]}:{'ok' if v['pass'] else 'FAIL'}"
                for k, v in sorted(rep.items()) if k != "0_row_mapping"
            ) + f"  mapping:{'ok' if all(e['ok'] for e in mapping.values()) else 'FAIL'}")
            print(f"           REAL GRAPH == HYBRID GRAPH: {rep['5_graph_topology_unchanged']['REAL_GRAPH == HYBRID_GRAPH']}"
                  f"   REAL TARGET == HYBRID TARGET: {rep['7_target_unchanged']['REAL_TARGET == HYBRID_TARGET']}")
            if not ok:
                failed = [k for k, v in rep.items() if k != "0_row_mapping" and not v["pass"]]
                print(f"  [seed {s}] FAILED checks: {failed}")
                all_ok = False
                if not args.dry_run:
                    raise AssertionError(f"{task_name} seed {s}: sanity checks failed: {failed}")
            hybrids[s] = hyb

        # ---- A and B: one run_task call, regimes real + synth --------------------------------
        records = run_with_tables(task, cfg, hybrids, seeds, xgb_seeds, domains, meta,
                                  real_tables, test_tables, regimes=["real", "synth"])
        for r in records:
            r["condition"] = CONDITION["real"] if r["regime"] == "real" else CONDITION["T1_E1"]
        del hybrids

        # ---- supplementary references --------------------------------------------------------
        if not args.no_references:
            shuffled = {s: build_row_permuted(real_raw, meta, dims, target, seed=s) for s in seeds}
            for s in seeds:
                sc = coerce_discrete(shuffled[s], domains)
                assert graph_fingerprint(sc, meta) == graph_fingerprint(real_tables, meta)
                _, yr, _ = task.build(real_tables, domains)
                _, ys, _ = task.build(sc, domains)
                assert np.array_equal(np.asarray(yr, float), np.asarray(ys, float))
            print(f"\n  --- reference {CONDITION['shuffled']} (graph + target checked equal to real)")
            ref = run_with_tables(task, cfg, shuffled, seeds, xgb_seeds, domains, meta,
                                  real_tables, test_tables, regimes=["synth"])
            for r in ref:
                r["condition"] = CONDITION["shuffled"]
            records += ref
            del shuffled

            if target is not None:
                gt = {s: build_hybrid(real_raw, grdm[s], meta, dims, None)[0] for s in seeds}
                print(f"\n  --- reference {CONDITION['grdm_target']}")
                ref = run_with_tables(task, cfg, gt, seeds, xgb_seeds, domains, meta,
                                      real_tables, test_tables, regimes=["synth"])
                for r in ref:
                    r["condition"] = CONDITION["grdm_target"]
                records += ref
                del gt

        write_json({"dataset": ds, "task": task_name, "records": records},
                   os.path.join(out, "records", f"{ds}__{task_name}.json"))
        for s in seeds:
            write_json(
                {
                    "dataset": ds, "task": task_name, "grdm_seed": s,
                    "sanity_all_passed": sanity[s]["all_passed"],
                    "records": [r for r in records if r["grdm_seed"] == s],
                },
                os.path.join(out, "per_seed", f"{ds}__{task_name}__grdm_seed{s}.json"),
            )
        print(f"  wrote {os.path.join(out, 'records', f'{ds}__{task_name}.json')}")

    print(f"\n[{EXP}] {ds} done in {time.time() - t_start:.0f}s; all sanity checks passed: {all_ok}")


if __name__ == "__main__":
    main()
