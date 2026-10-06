"""Minimal, standalone I/O for the dependency-fidelity checks.

Deliberately independent of `downstream/data.py`: these checks only need six
tables, three date epochs, and a handful of discrete columns rounded back to
integer codes, so keeping a separate loader means this module has no
dependency on the downstream ML-task harness.

Date columns are stored as "days since the earliest date observed in that
column" (see ClavaDDPM/preprocess_utils.py:calculate_days_since_earliest_date),
and that reference date is chosen independently per table -- account/trans
epoch to 1993-01-01, loan to 1993-07-05, card to 1993-11-07. Comparing two
tables' date columns directly (e.g. loan_date - account_date) is off by the
gap between their epochs; every check here converts to absolute calendar days
first via the epoch sidecar files before comparing across tables.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_DIR = os.path.join(REPO_ROOT, "data", "berka", "preprocessed")

TABLES = ("account", "trans", "loan", "card", "disp", "order")
EPOCH_TABLES = ("account", "trans", "loan", "card")

DISCRETE_COLS = {
    "trans": ("trans_type", "operation", "k_symbol"),
    "order": ("k_symbol",),
}
ID_COLS = {
    "account": ("account_id", "district_id"),
    "trans": ("trans_id", "account_id"),
    "loan": ("loan_id", "account_id"),
    "card": ("card_id", "disp_id"),
    "disp": ("client_id", "account_id", "disp_id"),
    "order": ("order_id", "account_id"),
}


def _read_epoch(data_dir: str, table: str) -> int:
    path = os.path.join(data_dir, f"{table}_earliest_date.txt")
    with open(path) as f:
        return datetime.strptime(f.read().strip(), "%y%m%d").toordinal()


def _round_columns(tables: dict) -> dict:
    out = {}
    for t, df in tables.items():
        df = df.copy()
        for col in (*DISCRETE_COLS.get(t, ()), *ID_COLS.get(t, ())):
            if col in df.columns:
                df[col] = np.rint(df[col].astype(float)).astype("int64")
        out[t] = df
    return out


def load_real(data_dir: str = REAL_DATA_DIR):
    tables = {t: pd.read_csv(os.path.join(data_dir, f"{t}.csv")) for t in TABLES}
    epochs = {t: _read_epoch(data_dir, t) for t in EPOCH_TABLES}
    return _round_columns(tables), epochs


def load_synthetic(synthetic_dir: str, epochs_dir: str = REAL_DATA_DIR):
    """`synthetic_dir` is a finished run's `synthetic_database/` directory.

    Synthetic date columns live in the same offset space the generator was
    trained on (real data's per-table epoch), not a re-derived synthetic one.
    """
    tables = {t: pd.read_csv(os.path.join(synthetic_dir, f"{t}_synthetic.csv")) for t in TABLES}
    epochs = {t: _read_epoch(epochs_dir, t) for t in EPOCH_TABLES}
    return _round_columns(tables), epochs
