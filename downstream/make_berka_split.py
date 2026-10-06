"""Write data/berka/split_train and data/berka/split_test.

The train half is what a leakage-free Berka generator must be trained on: rerun GRDM with
configs/berka_split_fastsample.json so the downstream test rows are genuinely unseen. The test
half is the held-out real database the downstream harness evaluates every regime against.

Domain files are copied unchanged. `<table>_domain.json` is read only for discrete/continuous
typing (rdbdiff/utils/data.py:50-71) and never for `size`, so keeping the full-database category
space is both safe and preferable -- it stops the split from silently redefining the encoding.

    conda run --no-capture-output -n tabsyn python -m downstream.make_berka_split
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT, assert_fk_integrity, load_db, load_meta
from downstream.splits import BERKA_SPLIT_SEED, BERKA_TEST_FRAC, split_berka

SOURCE = os.path.join(REPO_ROOT, "data", "berka", "preprocessed")
OUT_TRAIN = os.path.join(REPO_ROOT, "data", "berka", "split_train")
OUT_TEST = os.path.join(REPO_ROOT, "data", "berka", "split_test")

# Not read by rdbdiff, but carried along so the split dirs stay drop-in replacements for
# `preprocessed` for anything else that reads this dataset.
SIDECAR_SUFFIXES = ("_label_encoders.pkl", "_earliest_date.txt")


def write_db(tables, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(SOURCE, "dataset_meta.json"), out_dir)
    for table, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{table}.csv"), index=False)
        shutil.copy(os.path.join(SOURCE, f"{table}_domain.json"), out_dir)
        for suffix in SIDECAR_SUFFIXES:
            src = os.path.join(SOURCE, f"{table}{suffix}")
            if os.path.exists(src):
                shutil.copy(src, out_dir)
    assert_fk_integrity(tables, meta, label=os.path.basename(out_dir))


def main():
    meta = load_meta(SOURCE)
    tables = load_db(SOURCE)
    assert_fk_integrity(tables, meta, label="berka/preprocessed")

    train, test = split_berka(tables, BERKA_TEST_FRAC, BERKA_SPLIT_SEED)

    train_accounts = set(train["account"]["account_id"])
    test_accounts = set(test["account"]["account_id"])
    assert not (train_accounts & test_accounts), "account leaked across the split"
    assert len(train_accounts) + len(test_accounts) == len(tables["account"])
    # 1:1 client<->disp is what makes the account cut clean; assert it rather than assume it.
    assert not (set(train["client"]["client_id"]) & set(test["client"]["client_id"]))
    for t in ("loan", "order", "trans", "card", "disp", "client"):
        assert len(train[t]) + len(test[t]) == len(tables[t]), f"{t} rows lost in the split"

    write_db(train, meta, OUT_TRAIN)
    write_db(test, meta, OUT_TEST)

    print(f"berka account split  seed={BERKA_SPLIT_SEED}  test_frac={BERKA_TEST_FRAC}")
    print(f"{'table':10s} {'full':>8s} {'train':>8s} {'test':>8s}")
    for t in meta["tables"]:
        print(f"{t:10s} {len(tables[t]):8d} {len(train[t]):8d} {len(test[t]):8d}")
    n_pos = lambda df: int(df["status"].isin([1, 3]).sum())  # noqa: E731  (B/D = defaulted)
    print(
        f"\nloan default positives: train {n_pos(train['loan'])}/{len(train['loan'])}  "
        f"test {n_pos(test['loan'])}/{len(test['loan'])}"
    )
    print(
        f"card ownership positives: train {len(train['card'])}/{len(train['disp'])}  "
        f"test {len(test['card'])}/{len(test['disp'])}"
    )
    print(f"\nwrote {OUT_TRAIN}\nwrote {OUT_TEST}")


if __name__ == "__main__":
    main()
