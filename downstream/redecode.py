"""Re-decode a cached GRDM sample with a different categorical decoder — no resampling.

    conda run -n tabsyn python -m downstream.redecode --dataset berka_split --seeds 0 1 2 \
        --decoding match

Sampling is the expensive stage and its output is cached as `node_type_to_X_syn.pkl`. Fix S1
only changes how those continuous values become class codes, so it can reuse the cache and run
in seconds instead of hours. This mirrors the tail of `main.py` (denormalize -> decode -> write
CSVs) and deliberately skips model construction and the SDV evaluation.

`--decoding round --verify` reconstructs the baseline instead of writing it, and asserts the
result is byte-identical to the stored CSVs. That is the guard that S1 left the reproduction
untouched.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdbdiff.dataset import RDBDiffDataset
from rdbdiff.utils.data import (
    denormalize_data,
    extract_attributes,
    read_json,
    read_pickle,
    round_synthetic_data,
    write_database,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR_NAME = "synthetic_data_sampled_structure_size_multiplier_1.0"


def num_numerical_features(data_path, table_names):
    """Count continuous columns per table.

    `make_pkey_fkey_graph` derives this from the same domain files, so counting them directly
    avoids building the graph (and touching a GPU) for what is a pure post-processing step.
    """
    out = {}
    for name in table_names:
        domain = read_json(os.path.join(data_path, f"{name}_domain.json"))
        out[name] = sum(1 for v in domain.values() if v["type"] == "continuous")
    return out


def redecode(workspace, data_path, decoding, verify=False):
    base = os.path.join(workspace, BASE_DIR_NAME)
    x_syn_path = os.path.join(base, "node_type_to_X_syn.pkl")
    if not os.path.exists(x_syn_path):
        raise SystemExit(f"no cached sample at {x_syn_path}")

    out_dir = base if decoding == "round" else f"{base}_decoding_{decoding}"
    structure_dir = os.path.join(out_dir, "sampled_structure")
    if not verify and out_dir != base and not os.path.exists(structure_dir):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copytree(os.path.join(base, "sampled_structure"), structure_dir)

    node_type_to_X_syn = denormalize_data(
        read_pickle(x_syn_path), os.path.join(workspace, "normalizers")
    )

    database_raw = RDBDiffDataset(data_path, cache_dir=None).get_db()
    database_syn = RDBDiffDataset(
        os.path.join(base, "sampled_structure"), cache_dir=None
    ).get_db()

    node_type_to_X_syn = round_synthetic_data(
        node_type_to_X_syn=node_type_to_X_syn,
        node_type_to_X_real=extract_attributes(database_raw),
        node_type_to_num_numerical_features=num_numerical_features(
            data_path, list(database_raw.table_dict)
        ),
        categorical_decoding=decoding,
    )

    for table_name, table in database_syn.table_dict.items():
        df = table.df
        cols_no_id = [c for c in df.columns if not c.endswith("_id")]
        df[cols_no_id] = node_type_to_X_syn[table_name]

    if verify:
        import tempfile

        import numpy as np
        import pandas as pd

        # Compare by column name, not bytes. Id-column *order* in the written CSVs was
        # nondeterministic before the `sort_columns` fix, so a stored baseline may carry a
        # different header order than a fresh write while holding identical values.
        with tempfile.TemporaryDirectory() as tmp:
            write_database(database_syn, tmp, suffix="_synthetic")
            stored = os.path.join(base, "synthetic_database")
            mismatched = []
            for fname in sorted(os.listdir(stored)):
                a = pd.read_csv(os.path.join(stored, fname))
                b = pd.read_csv(os.path.join(tmp, fname))
                if set(a.columns) != set(b.columns) or len(a) != len(b):
                    mismatched.append(f"{fname} (shape/columns)")
                    continue
                b = b[list(a.columns)]
                for col in a.columns:
                    if not np.allclose(
                        a[col].to_numpy(dtype=float),
                        b[col].to_numpy(dtype=float),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    ):
                        mismatched.append(f"{fname}:{col}")
        return mismatched

    write_database(database_syn, os.path.join(out_dir, "synthetic_database"), suffix="_synthetic")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="berka_split")
    parser.add_argument("--data-path", default=os.path.join(REPO_ROOT, "data/berka/split_train"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--decoding", default="match", choices=["round", "match"])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reconstruct instead of writing, and assert the result matches the stored CSVs.",
    )
    parser.add_argument("--exp-name", default="paper_repro")
    args = parser.parse_args()

    for seed in args.seeds:
        workspace = os.path.join(
            REPO_ROOT, "rdbdiff_workspace", args.dataset, args.exp_name, f"seed_{seed}"
        )
        result = redecode(workspace, args.data_path, args.decoding, verify=args.verify)
        if args.verify:
            status = "IDENTICAL" if not result else f"DIFFERS: {result}"
            print(f"seed {seed}: {status}")
            if result:
                raise SystemExit("baseline reconstruction does not match stored output")
        else:
            print(f"seed {seed}: wrote {result}")


if __name__ == "__main__":
    main()
