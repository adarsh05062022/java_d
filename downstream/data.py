"""Loading and type-alignment for real / synthetic relational databases.

Both real and synthetic tables live in the same preprocessed numeric space: discrete columns
are integer codes, continuous columns are raw values. The generator emits everything as float,
so discrete columns need an explicit round before they can be cast to a shared category dtype.
"""

import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_meta(data_path):
    return read_json(os.path.join(data_path, "dataset_meta.json"))


def load_domains(data_path, table_names):
    """{table: {column: {"type": "discrete"|"continuous", "size": int}}}"""
    return {
        t: read_json(os.path.join(data_path, f"{t}_domain.json")) for t in table_names
    }


def load_db(data_path, suffix=""):
    """Read every table named in dataset_meta.json. `suffix` is '_synthetic' for generated data."""
    meta = load_meta(data_path)
    return {
        t: pd.read_csv(os.path.join(data_path, f"{t}{suffix}.csv"))
        for t in meta["tables"]
    }


def load_synthetic_db(workspace_dir, table_names):
    """Read `<workspace>/synthetic_database/<table>_synthetic.csv` for a finished GRDM run."""
    syn_dir = os.path.join(workspace_dir, "synthetic_database")
    return {
        t: pd.read_csv(os.path.join(syn_dir, f"{t}_synthetic.csv")) for t in table_names
    }


def grdm_run_dir(dataset, seed, exp_name="paper_repro", multiplier="1.0", variant=""):
    """Locate a finished run's output. `variant` selects a re-decoded copy (fix S1)."""
    return os.path.join(
        REPO_ROOT,
        "rdbdiff_workspace",
        dataset,
        exp_name,
        f"seed_{seed}",
        f"synthetic_data_sampled_structure_size_multiplier_{multiplier}{variant}",
    )


# --------------------------------------------------------------------------------------
# type alignment
# --------------------------------------------------------------------------------------


def discrete_columns(domains):
    """{table: [column, ...]} for every column typed 'discrete' in the real domain files."""
    return {
        t: [c for c, spec in dom.items() if spec["type"] == "discrete"]
        for t, dom in domains.items()
    }


def coerce_discrete(tables, domains):
    """Round generator floats back to the integer code space, in place, returning a new dict.

    Values are left as plain integers here rather than pandas categoricals: the feature builders
    still need to do arithmetic and grouping on them. Categorical casting happens once, on the
    assembled feature matrix, in `align_feature_dtypes`.
    """
    out = {}
    for t, df in tables.items():
        df = df.copy()
        for col in domains.get(t, {}):
            if domains[t][col]["type"] == "discrete" and col in df.columns:
                df[col] = np.rint(df[col].astype(float)).astype("int64")
        out[t] = df
    return out


def align_feature_dtypes(frames, categorical_spec):
    """Cast the named columns of every frame to an identical pandas category dtype.

    `categorical_spec` maps feature-column name -> number of categories (from the real domain
    file). Codes outside [0, size) become NaN, which XGBoost handles natively under
    `enable_categorical=True`. Returns (frames, out_of_domain_rates).
    """
    out_of_domain = {}
    aligned = []
    for df in frames:
        df = df.copy()
        for col, size in categorical_spec.items():
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            valid = values.between(0, size - 1) | values.isna()
            out_of_domain[col] = out_of_domain.get(col, 0.0) + float((~valid).sum())
            values = values.where(valid)
            df[col] = pd.Categorical(values, categories=list(range(size)))
        aligned.append(df)
    return aligned, out_of_domain


# --------------------------------------------------------------------------------------
# integrity checks
# --------------------------------------------------------------------------------------


def assert_fk_integrity(tables, meta, label=""):
    """Every child FK must resolve to a row in the parent table."""
    for parent, child in meta["relation_order"]:
        if parent is None:
            continue
        key = f"{parent}_id"
        child_df, parent_df = tables[child], tables[parent]
        if key not in child_df.columns:
            raise AssertionError(f"{label}: {child} has no {key} column")
        missing = ~child_df[key].isin(set(parent_df[key]))
        if missing.any():
            raise AssertionError(
                f"{label}: {missing.sum()}/{len(child_df)} rows of {child}.{key} "
                f"do not resolve into {parent}"
            )


def assert_disjoint(train_ids, test_ids, name=""):
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(
            f"{name}: {len(overlap)} ids appear in both train and test (e.g. "
            f"{sorted(overlap)[:5]})"
        )
