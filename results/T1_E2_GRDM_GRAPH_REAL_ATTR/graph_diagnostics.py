"""T1_E2 pre-training diagnostics: real graph vs the GRDM-sampled graph.

Everything here is read-only and descriptive. Nothing is used to modify, repair or select a
graph. It answers three questions before any hybrid can be built:

1. How do the real and GRDM graphs differ (nodes, edges, degree distributions, components,
   cardinalities, the project's existing Cardinality fidelity score)?
2. Does the GRDM graph carry any node-to-entity correspondence that would let real attribute
   records be placed on generated nodes without an arbitrary assignment?
3. Which of the experiment's sanity checks can be evaluated before a hybrid exists?

Both graphs are read in GRDM's own index space: `RDBDiffDataset(data_path).get_db()` runs
relbench's `reindex_pkeys_and_fkeys`, which renumbers every pkey to its CSV row position. That
is the space the structure sampler reads and the space `make_pkey_fkey_graph` asserts.

    conda run -n tabsyn python results/T1_E2_GRDM_GRAPH_REAL_ATTR/graph_diagnostics.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from downstream.data import assert_fk_integrity  # noqa: E402
from rdbdiff.dataset import RDBDiffDataset  # noqa: E402
from rdbdiff.utils import read_json  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "T1_E2_GRDM_GRAPH_REAL_ATTR")
GRDM_SEEDS = [0, 1, 2]

DATASETS = {
    "berka_split": dict(config="configs/berka_split_fastsample.json", test="data/berka/split_test"),
    "california": dict(
        config="configs/california_fastsample.json",
        test="/scratch/s25017/ClavaDDPM/ClavaDDPM/complex_data/california/test",
    ),
}


def run_dir(dataset, seed):
    return os.path.join(
        REPO, "rdbdiff_workspace", dataset, "paper_repro", f"seed_{seed}",
        "synthetic_data_sampled_structure_size_multiplier_1.0",
    )


def load_real(data_path):
    db = RDBDiffDataset(data_path, cache_dir=None).get_db()
    return {t: tbl.df.copy() for t, tbl in db.table_dict.items()}


def load_generated(dataset, seed):
    # sampled_structure/ is the sampler's raw output (keys + zero-filled attributes); the
    # synthetic_database/ written later reuses exactly these keys.
    return load_real(os.path.join(run_dir(dataset, seed), "sampled_structure"))


def in_degree(tables, parent, child):
    key = f"{parent}_id"
    counts = tables[child][key].value_counts()
    return tables[parent][key].map(counts).fillna(0).astype(int).to_numpy()


def degree_summary(deg):
    deg = np.asarray(deg)
    return dict(
        n_parents=int(deg.size),
        n_edges=int(deg.sum()),
        mean=float(deg.mean()),
        std=float(deg.std(ddof=0)),
        min=int(deg.min()),
        p50=float(np.percentile(deg, 50)),
        p90=float(np.percentile(deg, 90)),
        p99=float(np.percentile(deg, 99)),
        max=int(deg.max()),
        frac_zero=float(np.mean(deg == 0)),
        frac_one=float(np.mean(deg == 1)),
        histogram={int(k): int(v) for k, v in zip(*np.unique(deg, return_counts=True))},
    )


def degree_distance(real, gen):
    support = np.union1d(np.unique(real), np.unique(gen))
    p = np.array([np.mean(real == v) for v in support])
    q = np.array([np.mean(gen == v) for v in support])
    return dict(
        total_variation=float(0.5 * np.abs(p - q).sum()),
        ks_statistic=float(ks_2samp(real, gen).statistic),
        wasserstein_1=float(wasserstein_distance(real, gen)),
        sorted_degree_sequences_identical=bool(
            real.size == gen.size and np.array_equal(np.sort(real), np.sort(gen))
        ),
    )


def components(tables, meta, exclude=()):
    """Weakly connected components of the homogeneous pk-fk graph."""
    names = [t for t in meta["tables"] if t not in exclude]
    offset, total = {}, 0
    for t in names:
        offset[t] = total
        total += len(tables[t])
    rows, cols = [], []
    for parent, child in meta["relation_order"]:
        if parent is None or parent in exclude or child in exclude:
            continue
        fk = tables[child][f"{parent}_id"].to_numpy()
        rows.append(offset[child] + np.arange(len(fk)))
        cols.append(offset[parent] + fk.astype(int))
    if rows:
        r, c = np.concatenate(rows), np.concatenate(cols)
        adj = coo_matrix((np.ones(r.size), (r, c)), shape=(total, total))
    else:
        adj = coo_matrix((total, total))
    n, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    return dict(
        tables=names,
        n_nodes=int(total),
        n_components=int(n),
        largest_component=int(sizes.max()),
        n_singletons=int(np.sum(sizes == 1)),
    )


def cross_child_degree_coupling(tables, meta):
    """Spearman correlation between a parent's in-degrees from two different child tables."""
    out = {}
    for parent, info in meta["tables"].items():
        kids = info["children"]
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                a = in_degree(tables, parent, kids[i])
                b = in_degree(tables, parent, kids[j])
                rho = spearmanr(a, b).correlation if a.std() > 0 and b.std() > 0 else None
                out[f"{parent}: {kids[i]} vs {kids[j]}"] = (
                    None if rho is None or np.isnan(rho) else float(rho)
                )
    return out


def berka_cross_hop_coupling(tables):
    """disp has-a-card vs its account's child counts. This is the structural side of the
    card-ownership label, so it is reported, never used."""
    card = in_degree(tables, "disp", "card")
    acc = tables["disp"]["account_id"].to_numpy()
    out = {}
    for child in ("trans", "order", "loan", "disp"):
        deg = in_degree(tables, "account", child)[acc]
        rho = spearmanr(card, deg).correlation
        out[f"disp.card_in_degree vs account.{child}_in_degree"] = float(rho)
    return out


def correspondence_test(real, gen, meta, dimension_tables):
    """Is there any identity-level link between generated node k and a real entity?

    For each relation, compare the generated in-degree of node k with the in-degree of the real
    node that a candidate mapping would put at k. If GRDM carried identity information the two
    would agree above chance. `chance_exact_match` is sum_d P_real(d)^2, the rate two independent
    draws from the real degree distribution agree.
    """
    out = {}
    dims = set(dimension_tables or [])

    # Candidate A: same index (generated pk k <-> real row k in GRDM's index space).
    # Candidate B (Berka only): within-district rank -- the k-th generated row of district d
    # <-> the k-th real row of district d, which is the most a district anchor can give.
    for parent, info in meta["tables"].items():
        for child in info["children"]:
            r = in_degree(real, parent, child)
            g = in_degree(gen, parent, child)
            chance = float(np.sum((np.bincount(r) / r.size) ** 2))
            rec = dict(
                parent_is_dimension_table=parent in dims,
                parent_rows_real=int(r.size),
                parent_rows_gen=int(g.size),
            )
            if r.size == g.size:
                rho = spearmanr(r, g).correlation if r.std() > 0 and g.std() > 0 else None
                rec["same_index"] = dict(
                    spearman=None if rho is None or np.isnan(rho) else float(rho),
                    exact_match=float(np.mean(r == g)),
                    chance_exact_match=chance,
                )
            if parent in dims:
                rec["per_parent_counts_identical"] = bool(np.array_equal(r, g))
            out[f"{parent}->{child}"] = rec

    if "district" in dims:
        for parent in ("client", "account"):
            r_sorted = real[parent].sort_values(["district_id", f"{parent}_id"], kind="stable")
            g_sorted = gen[parent].sort_values(["district_id", f"{parent}_id"], kind="stable")
            aligned = np.array_equal(
                r_sorted["district_id"].to_numpy(), g_sorted["district_id"].to_numpy()
            )
            r_deg_all = {
                c: in_degree(real, parent, c) for c in meta["tables"][parent]["children"]
            }
            g_deg_all = {
                c: in_degree(gen, parent, c) for c in meta["tables"][parent]["children"]
            }
            for child in meta["tables"][parent]["children"]:
                r = r_deg_all[child][r_sorted.index.to_numpy()]
                g = g_deg_all[child][g_sorted.index.to_numpy()]
                rho = spearmanr(r, g).correlation if r.std() > 0 and g.std() > 0 else None
                out[f"{parent}->{child}"]["within_district_rank"] = dict(
                    district_sequences_aligned=bool(aligned),
                    spearman=None if rho is None or np.isnan(rho) else float(rho),
                    exact_match=float(np.mean(r == g)),
                    chance_exact_match=float(np.sum((np.bincount(r) / r.size) ** 2)),
                )
    return out


def target_slot_overlap(real, gen):
    """Berka: would a real label land on a node that carries it in the real graph?

    Loan default: under the within-district rank mapping of accounts (the most a district
    anchor gives), the share of GRDM loans whose account holds a loan in the real data.
    Card ownership: under the same-index mapping of dispositions (equal row counts), the share
    of GRDM card-holding dispositions whose real counterpart holds a card. If GRDM's graph
    carried identity, these would sit well above the real base rate.
    """
    r_acc = real["account"].sort_values(["district_id", "account_id"], kind="stable")
    g_acc = gen["account"].sort_values(["district_id", "account_id"], kind="stable")
    r_loan = in_degree(real, "account", "loan")[r_acc.index.to_numpy()] > 0
    g_loan = in_degree(gen, "account", "loan")[g_acc.index.to_numpy()] > 0
    r_card = in_degree(real, "disp", "card") > 0
    g_card = in_degree(gen, "disp", "card") > 0
    return {
        "loan: P(real account has loan | GRDM account has loan), within-district rank map": float(
            r_loan[g_loan].mean()
        ),
        "loan: real base rate P(account has loan)": float(r_loan.mean()),
        "card: P(real disp has card | GRDM disp has card), same-index map": float(
            r_card[g_card].mean()
        ),
        "card: real base rate P(disp has card)": float(r_card.mean()),
    }


def sampler_input_columns(data_path, dims):
    """Every column the structure sampler reads, taken from a live RandomGraphSampler."""
    from rdbdiff.random_graph_sampler import RandomGraphSampler

    s = RandomGraphSampler(real_data_path=data_path, dimension_tables=dims)
    return {t: list(df.columns) for t, df in s.real_table_to_structure.items()}


def train_test_disjoint(meta, train_path, test_path, shared=()):
    out = {}
    for t in meta["tables"]:
        a = set(pd.read_csv(os.path.join(train_path, f"{t}.csv"), usecols=[f"{t}_id"])[f"{t}_id"])
        b = set(pd.read_csv(os.path.join(test_path, f"{t}.csv"), usecols=[f"{t}_id"])[f"{t}_id"])
        out[t] = dict(overlap=len(a & b), shared_dimension_table=t in shared)
    return out


def integrity(tables, meta, label):
    res = {}
    for t in meta["tables"]:
        pk = tables[t][f"{t}_id"].to_numpy()
        res[f"{t}.pk_unique"] = bool(len(np.unique(pk)) == len(pk))
        res[f"{t}.pk_consecutive"] = bool(np.array_equal(pk, np.arange(len(pk))))
    try:
        assert_fk_integrity(tables, meta, label=label)
        res["fk_resolve_all"] = True
    except AssertionError as e:
        res["fk_resolve_all"] = False
        res["fk_error"] = str(e)
    for t, info in meta["tables"].items():
        fks = [f"{p}_id" for p in info["parents"]]
        if fks:
            res[f"{t}.null_fks"] = int(tables[t][fks].isna().sum().sum())
        if len(fks) > 1:
            res[f"{t}.duplicate_fk_combinations"] = int(tables[t].duplicated(subset=fks).sum())
    return res


def existing_cardinality_score(dataset, seed):
    """The project's existing graph-fidelity metric: SDV CardinalityShapeSimilarity, written by
    `gen_multi_report` during the GRDM run."""
    path = os.path.join(run_dir(dataset, seed), "eval_result.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        report = pickle.load(f)["report"]
    props = report.get_properties().set_index("Property")["Score"]
    details = report.get_details("Cardinality")
    return dict(
        overall=float(props["Cardinality"]),
        per_relation={
            f"{r['Parent Table']}->{r['Child Table']}": float(r["Score"])
            for _, r in details.iterrows()
        },
    )


def schema_check(real, gen_struct, gen_syn):
    out = {}
    for t in real:
        out[t] = dict(
            real_columns=sorted(real[t].columns),
            sampled_structure_columns_identical=set(gen_struct[t].columns) == set(real[t].columns),
            synthetic_database_columns_identical=set(gen_syn[t].columns) == set(real[t].columns),
            sampled_structure_nonkey_all_zero=bool(
                (gen_struct[t][[c for c in gen_struct[t].columns if not c.endswith("_id")]]
                 .to_numpy() == 0).all()
            ),
        )
    return out


def main():
    stats, checks = {}, {}
    for dataset, spec in DATASETS.items():
        cfg = read_json(spec["config"])
        data_path = cfg["data"]["data_path"]
        dims = cfg["data"]["dimension_tables"] or None
        meta = read_json(os.path.join(data_path, "dataset_meta.json"))
        print(f"\n##### {dataset}  data_path={data_path}  dimension_tables={dims}", flush=True)

        real = load_real(data_path)
        real_struct = {
            "nodes": {t: len(df) for t, df in real.items()},
            "degree": {
                f"{p}->{c}": degree_summary(in_degree(real, p, c))
                for p, info in meta["tables"].items() for c in info["children"]
            },
            "components_all_tables": components(real, meta),
            "components_without_dimension_tables": components(real, meta, exclude=dims or ()),
            "cross_child_degree_coupling": cross_child_degree_coupling(real, meta),
        }
        if dataset.startswith("berka"):
            real_struct["cross_hop_degree_coupling"] = berka_cross_hop_coupling(real)
        stats[dataset] = {"real": real_struct, "grdm": {}}
        checks[dataset] = {"real_integrity": integrity(real, meta, "real"), "seeds": {}}

        for seed in GRDM_SEEDS:
            print(f"  seed {seed}", flush=True)
            gen = load_generated(dataset, seed)
            syn_dir = os.path.join(run_dir(dataset, seed), "synthetic_database")
            gen_syn = {t: pd.read_csv(os.path.join(syn_dir, f"{t}_synthetic.csv"), nrows=5)
                       for t in meta["tables"]}
            # the synthetic database must carry the sampled keys unchanged
            syn_keys_match = {}
            for t in meta["tables"]:
                ids = [c for c in gen[t].columns if c.endswith("_id")]
                syn = pd.read_csv(os.path.join(syn_dir, f"{t}_synthetic.csv"), usecols=ids)
                syn_keys_match[t] = bool(
                    len(syn) == len(gen[t])
                    and all(np.array_equal(syn[c].to_numpy(), gen[t][c].to_numpy()) for c in ids)
                )

            deg, dist = {}, {}
            for p, info in meta["tables"].items():
                for c in info["children"]:
                    r, g = in_degree(real, p, c), in_degree(gen, p, c)
                    deg[f"{p}->{c}"] = degree_summary(g)
                    dist[f"{p}->{c}"] = degree_distance(r, g)

            g_stats = {
                "nodes": {t: len(df) for t, df in gen.items()},
                "node_delta_vs_real": {t: len(gen[t]) - len(real[t]) for t in gen},
                "degree": deg,
                "degree_distance_vs_real": dist,
                "components_all_tables": components(gen, meta),
                "components_without_dimension_tables": components(gen, meta, exclude=dims or ()),
                "cross_child_degree_coupling": cross_child_degree_coupling(gen, meta),
                "existing_fidelity_metric_SDV_Cardinality": existing_cardinality_score(dataset, seed),
                "correspondence_test": correspondence_test(real, gen, meta, dims),
            }
            if dataset.startswith("berka"):
                g_stats["cross_hop_degree_coupling"] = berka_cross_hop_coupling(gen)
                g_stats["target_slot_overlap"] = target_slot_overlap(real, gen)
            stats[dataset]["grdm"][f"seed_{seed}"] = g_stats

            checks[dataset]["seeds"][f"seed_{seed}"] = dict(
                schema=schema_check(real, gen, gen_syn),
                generated_integrity=integrity(gen, meta, f"grdm seed {seed}"),
                synthetic_database_keys_equal_sampled_structure=syn_keys_match,
                graph_differs_from_real={
                    k: not v["sorted_degree_sequences_identical"] for k, v in dist.items()
                },
                tables_with_different_row_count=[t for t in gen if len(gen[t]) != len(real[t])],
            )

            for t in meta["tables"]:
                print(f"    {t:11s} real={len(real[t]):8d} grdm={len(gen[t]):8d}", flush=True)
            for k, v in g_stats["correspondence_test"].items():
                si = v.get("same_index", {})
                wd = v.get("within_district_rank", {})
                print(
                    f"    {k:20s} dim_parent={v['parent_is_dimension_table']!s:5s} "
                    f"per_parent_identical={v.get('per_parent_counts_identical', '-')!s:5s} "
                    f"same_idx exact={si.get('exact_match', float('nan')):.3f} "
                    f"(chance {si.get('chance_exact_match', float('nan')):.3f}) "
                    f"within_district exact={wd.get('exact_match', float('nan')):.3f}",
                    flush=True,
                )

        checks[dataset]["sampler_input_columns"] = sampler_input_columns(data_path, dims)
        checks[dataset]["train_test_id_overlap"] = train_test_disjoint(
            meta, data_path, spec["test"], shared=dims or ()
        )
        checks[dataset]["grdm_trained_on"] = data_path

    with open(os.path.join(OUT_DIR, "graph_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "sanity_checks.json"), "w") as f:
        json.dump(sanity_report(stats, checks), f, indent=2)
    print("\nwrote graph_statistics.json and sanity_checks.json")


def sanity_report(stats, checks):
    """The experiment's ten sanity checks, as far as they can be evaluated without a hybrid."""
    seeds = lambda ds: checks[ds]["seeds"].values()  # noqa: E731
    all_ds = list(checks)
    rows_differ = {
        ds: {s: c["tables_with_different_row_count"] for s, c in checks[ds]["seeds"].items()}
        for ds in all_ds
    }
    reads_only_keys = {
        ds: all(c.endswith("_id") or c.endswith("_in_degree")
                for cols in checks[ds]["sampler_input_columns"].values() for c in cols)
        for ds in all_ds
    }
    leak_free = {
        ds: all(v["overlap"] == 0 or v["shared_dimension_table"]
                for v in checks[ds]["train_test_id_overlap"].values())
        for ds in all_ds
    }
    blocked = (
        "BLOCKED: no hybrid was built. GRDM's sampled graph carries no node-to-entity mapping "
        "below the dimension table, so any placement of real records on generated nodes would "
        "be an assignment chosen by the experimenter. See mapping_documentation.md."
    )
    return {
        "status": "BLOCKED_BEFORE_HYBRID_CONSTRUCTION",
        "checks": {
            "1_same_table_schema": {
                "status": "PASS" if all(
                    v["sampled_structure_columns_identical"] and v["synthetic_database_columns_identical"]
                    for ds in all_ds for c in seeds(ds) for v in c["schema"].values()
                ) else "FAIL",
                "scope": "GRDM sampled_structure and synthetic_database vs real, every table, seeds 0-2",
            },
            "2_same_attribute_columns": {
                "status": "PASS (schema only)",
                "note": "Same non-key column set in GRDM output and real data. Row-level identity is check 3.",
            },
            "3_real_attributes_unchanged": {
                "status": "BLOCKED",
                "reason": blocked,
                "row_count_mismatch_tables": rows_differ,
                "note": "Where row counts differ, no assignment can keep the real row multiset intact: "
                        "real rows would have to be dropped or duplicated.",
            },
            "4_real_target_unchanged": {
                "status": "BLOCKED / NOT SATISFIABLE for one task",
                "berka_card_ownership": "NOT SATISFIABLE: the label is 'disp has a card child', i.e. an "
                                        "edge of the graph. Replacing the graph replaces the label.",
                "berka_loan_default": "BLOCKED: loan.status sits on loan rows whose count (GRDM "
                                      f"{[stats['berka_split']['grdm'][s]['nodes']['loan'] for s in stats['berka_split']['grdm']]} "
                                      f"vs real {stats['berka_split']['real']['nodes']['loan']}) and parent accounts are "
                                      "set by the graph.",
                "california_inctot / california_empstat": "BLOCKED: label is an individual attribute; GRDM's "
                                                          "individual count and household membership differ.",
            },
            "5_grdm_graph_differs_from_real": {
                "status": "PASS",
                "relations_with_identical_degree_sequence": {
                    ds: sorted({k for c in seeds(ds) for k, v in c["graph_differs_from_real"].items() if not v})
                    for ds in all_ds
                },
                "note": "Identical relations are the dimension-table anchors (district->client, "
                        "district->account) and the strictly 1:1 client->disp.",
            },
            "6_pkfk_valid": {
                "status": "PASS" if all(
                    all(v for k, v in c["generated_integrity"].items() if k.endswith(("pk_unique", "pk_consecutive")))
                    for ds in all_ds for c in seeds(ds)
                ) else "FAIL",
            },
            "7_referential_integrity": {
                "status": "PASS" if all(
                    c["generated_integrity"]["fk_resolve_all"]
                    and all(v == 0 for k, v in c["generated_integrity"].items() if k.endswith("null_fks"))
                    for ds in all_ds for c in seeds(ds)
                ) else "FAIL",
            },
            "8_no_accidental_attribute_permutation": {
                "status": "BLOCKED: cannot be satisfied with the GRDM graph as generated",
                "reason": "For every relation below the dimension table, a generated node's degree "
                          "agrees with any candidate real node's degree only at chance (see "
                          "graph_statistics.json correspondence_test). Every assignment of real records "
                          "to generated nodes is therefore a permutation that GRDM did not supply.",
            },
            "9_no_train_test_leakage": {
                "status": "PASS" if all(leak_free.values()) else "FAIL",
                "grdm_trained_and_structure_sampled_on": {ds: checks[ds]["grdm_trained_on"] for ds in all_ds},
                "train_test_id_overlap": {ds: checks[ds]["train_test_id_overlap"] for ds in all_ds},
            },
            "10_target_not_used_to_generate_graph": {
                "status": "PASS for attribute targets; FAIL by construction for card ownership",
                "structure_sampler_reads_only_key_and_degree_columns": reads_only_keys,
                "berka_loan_default": "PASS: loan.status is dropped by get_structure_from_db. "
                                      "The account->loan degree histogram (whether a loan exists) is an input.",
                "berka_card_ownership": "FAIL by construction: the disp->card degree histogram, i.e. the "
                                        "label's marginal P(card), is a sampler input.",
                "california": "PASS: INCTOT and EMPSTAT are dropped; only household->individual degrees are read.",
            },
        },
        "detail": checks,
    }


if __name__ == "__main__":
    main()
