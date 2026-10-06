"""Decode a reparameterized GRDM run back into absolute dates.

    conda run -n tabsyn python -m dep_fidelity.decode_run --dataset berka_split_reparam --seeds 0

A generator trained on `data/berka/split_train_reparam` emits child dates in the
`log1p(lag from parent)` space of `reparam.py`. This inverts that transform on a finished
sample and writes the result as a `_decoded` variant run directory, so `dep_fidelity.run`
and `downstream.run` can read it exactly like any other sample.

The parent's date is needed to resolve a child's lag, and `account_date` is not
reparameterized (account is the root), so a single pass over the sampled database
suffices -- the parent value is already absolute. Tables the transform does not touch
are copied through unchanged.
"""

import argparse
import json
import os
import shutil

from dep_fidelity.load import TABLES, load_synthetic
from dep_fidelity.reparam import decode_tables

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR_NAME = "synthetic_data_sampled_structure_size_multiplier_1.0"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="berka_split_reparam")
    p.add_argument("--exp-name", default="paper_repro")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs-dir", default=os.path.join(REPO_ROOT, "data/berka/split_train"),
                   help="source of the per-table date epochs (identical across Berka splits)")
    p.add_argument("--variant-suffix", default="_decoded")
    args = p.parse_args()

    for seed in args.seeds:
        run_dir = os.path.join(REPO_ROOT, "rdbdiff_workspace", args.dataset,
                               args.exp_name, f"seed_{seed}", BASE_DIR_NAME)
        syn_dir = os.path.join(run_dir, "synthetic_database")
        if not os.path.isdir(syn_dir):
            print(f"[skip] seed {seed}: no sample at {syn_dir}")
            continue

        tables, epochs = load_synthetic(syn_dir, epochs_dir=args.epochs_dir)
        decoded = decode_tables(tables, epochs)

        out_run = run_dir + args.variant_suffix
        out_dir = os.path.join(out_run, "synthetic_database")
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(syn_dir):
            shutil.copyfile(os.path.join(syn_dir, fname), os.path.join(out_dir, fname))
        for t in TABLES:
            decoded[t].to_csv(os.path.join(out_dir, f"{t}_synthetic.csv"), index=False)
        with open(os.path.join(out_run, "decode_manifest.json"), "w") as f:
            json.dump(dict(source=syn_dir, transform="expm1_lag_to_absolute"), f, indent=2)
        print(f"seed {seed}: wrote {out_run}")


if __name__ == "__main__":
    main()
