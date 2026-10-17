"""Shared paths and loaders for the Stage 1 diagnostics.

Read-only: every function here reads synthetic CSVs that earlier experiments already wrote.
Nothing is trained, sampled or overwritten. Outputs go to ../outputs/ only.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

AMEX = "/storage/s25017/AMEX"
REPO = f"{AMEX}/java_d"
sys.path.insert(0, REPO)

from downstream import data as D  # noqa: E402  (harness loaders, unchanged)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SYN = "synthetic_data_sampled_structure_size_multiplier_1.0/synthetic_database"
SEEDS = (0, 1, 2)

# Berka: every generator below was trained on the FULL database (leakage_caveat=true).
BERKA_REAL = f"{REPO}/data/berka/preprocessed"
BERKA_CONFIGS = {
    "baseline": f"{REPO}/rdbdiff_workspace/berka/paper_repro/seed_{{s}}/{SYN}",
    "degree_scratch": f"{AMEX}/java_d_degcond/experiments/GRDM_DEGREE_CONDITIONED_SCRATCH/workspace/seed_{{s}}/{SYN}",
    "degree_ft": f"{AMEX}/java_d_degft/experiments/GRDM_BERKA_DEGREE_CONDITIONED_FINETUNE/workspace/seed_{{s}}/{SYN}",
    "agg_scratch": f"{AMEX}/java_d-agg/experiments/GRDM_AGGREGATE_CONDITIONED_SCRATCH/runs/seed_{{s}}/{SYN}",
    "agg_ft_joint": f"{AMEX}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/runs/same_spec/seed_{{s}}/{SYN}",
    "agg_ft_frozen": f"{AMEX}/java_d-aggft/experiments/GRDM_BERKA_AGGREGATE_CONDITIONED_FINETUNE/runs/same_spec_frozen/seed_{{s}}/{SYN}",
}

# California: generators trained on data/california/train only; test split is separate.
CAL_REAL_TRAIN = f"{REPO}/data/california/train"
CAL_REAL_TEST = f"{AMEX}/complex_data/california/test"
CAL_CONFIGS = {
    "baseline": f"{REPO}/rdbdiff_workspace/california/paper_repro/seed_{{s}}/{SYN}",
    "control": f"{AMEX}/java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/runs/control/seed_{{s}}/{SYN}",
    "agg": f"{AMEX}/java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/runs/agg/seed_{{s}}/{SYN}",
    "agg_label_free": f"{AMEX}/java_d-calagg-ft/experiments/GRDM_CALIFORNIA_AGGREGATE_CONDITIONED_FINETUNE/runs/agg_label_free/seed_{{s}}/{SYN}",
}


def meta_domains(real_path):
    meta = D.load_meta(real_path)
    names = list(meta["tables"])
    return meta, names, D.load_domains(real_path, names)


def load_real(real_path):
    meta, names, dom = meta_domains(real_path)
    tables = D.load_db(real_path)
    D.assert_fk_integrity(tables, meta, label=f"real {real_path}")
    return D.coerce_discrete(tables, dom), dom, meta


def load_syn(syn_dir, real_path):
    """Same steps as downstream.run.load_syn_tables, pointed at an arbitrary run folder."""
    meta, names, dom = meta_domains(real_path)
    tables = {t: pd.read_csv(os.path.join(syn_dir, f"{t}_synthetic.csv")) for t in names}
    D.assert_fk_integrity(tables, meta, label=f"synthetic {syn_dir}")
    return D.coerce_discrete(tables, dom), dom, meta


def rank01(x):
    """Pooled rank transform to (0, 1]; ties share the average rank."""
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def eta_squared(values, groups):
    """Correlation ratio: between-group share of total variance (SS_between / SS_total)."""
    df = pd.DataFrame({"v": values, "g": groups})
    gm = df.groupby("g")["v"].agg(["mean", "size"])
    grand = df["v"].mean()
    ss_b = float((gm["size"] * (gm["mean"] - grand) ** 2).sum())
    ss_t = float(((df["v"] - grand) ** 2).sum())
    return ss_b / ss_t if ss_t > 0 else float("nan"), gm


def weighted_r2(y, X, w):
    """Weighted least-squares R^2 of y on X (X gets an intercept)."""
    X = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    ybar = np.average(y, weights=w)
    return 1.0 - float(np.sum(w * resid**2) / np.sum(w * (y - ybar) ** 2))


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def dump(obj, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"wrote {path}")
    return path


def mean_sd(vals):
    """Mean and population SD (ddof=0), the Berka-table convention in this project."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "sd": None, "n": 0, "values": []}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=0)), "n": int(a.size),
            "values": [float(v) for v in a]}
