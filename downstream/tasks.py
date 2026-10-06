"""Downstream task definitions: label extraction and relational feature construction.

Every feature set spans joins and/or child aggregates. Single-table features would exercise
nothing GRDM does beyond a per-table tabular model, so they would not be a test of relational
generation.

Two leakage rules are enforced here rather than assumed:

1. An aggregate may only be built from columns that are not the task's target
   (`_check_no_target_leak`). That removes self-inclusion -- a household's mean EMPSTAT contains
   the row's own EMPSTAT -- without leave-one-out arithmetic.
2. No feature may be a deterministic consequence of the label. Berka's transaction table is full
   of these, and the label encoders name them exactly: `operation`=5 is 'VYBER KARTOU' (a credit
   card withdrawal, which cannot exist unless the disposition holds a card), `k_symbol`=4 is
   'SANKC. UROK' (sanction interest, charged only once an account is in debt) and 8 is 'UVER'
   (a loan repayment). Left in, they gave the real-data baselines AUROC 1.0000 and 0.9805 --
   ceilings that measure the schema, not the model. See BERKA_LOAN and BERKA_CARD notes below.
"""

import functools
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from downstream.splits import DEFAULT_STATUS

TRANS_CODE_COLS = ("trans_type", "operation", "k_symbol", "bank")
ORDER_CODE_COLS = ("k_symbol", "bank_to")

# Source of the per-table date origins: data/berka/preprocessed in rdb-diffusion, and
# complex_data/berka/preprocessed in the ClavaDDPM copy of this harness (same values). Every Berka
# variant (split, agg, reparam after decoding) and every generator sample keeps that encoding, so
# one set of origins serves real and synthetic tables alike.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERKA_DATE_ORIGIN_DIR = next(
    (
        d
        for d in (
            os.path.join(_REPO, "data", "berka", "preprocessed"),
            os.path.join(_REPO, "complex_data", "berka", "preprocessed"),
        )
        if os.path.isdir(d)
    ),
    os.path.join(_REPO, "data", "berka", "preprocessed"),
)


# ------------------------------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def berka_date_offset(table):
    """Days to add to `table`'s date column to put it on the transaction table's day count.

    Berka stores each date column as days since the earliest value in *that table*
    (`<table>_earliest_date.txt`): transactions and accounts count from 1 Jan 1993, loans from
    5 Jul 1993, cards from 7 Nov 1993. Dates from two tables can only be compared after this
    shift; comparing the stored values directly put the loan cutoff 185 days too early.
    """

    def origin(t):
        with open(os.path.join(BERKA_DATE_ORIGIN_DIR, f"{t}_earliest_date.txt")) as f:
            return datetime.strptime(f.read().strip(), "%y%m%d").toordinal()

    return origin(table) - origin("trans")


def _fractions_by_code(df, group_key, col, size, prefix):
    """Per-group share of each discrete code, reindexed onto the full domain.

    Reindexing matters: without it a synthetic table that never emits some code would produce a
    narrower feature matrix than the real one and the two would stop being comparable.
    """
    counts = df.groupby([df[group_key], df[col].astype("int64")]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(size), fill_value=0)
    fracs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fracs.columns = [f"{prefix}_{col}_{c}_frac" for c in fracs.columns]
    return fracs


def _berka_trans_agg(trans, domains, group_key="account_id", code_cols=TRANS_CODE_COLS):
    """Transaction rollup: 13 numeric summaries plus a code-share vector per discrete column."""
    ordered = trans.sort_values([group_key, "trans_date"])
    g = ordered.groupby(group_key)
    agg = pd.DataFrame(
        {
            "trans_n": g.size(),
            "trans_amount_mean": g["amount"].mean(),
            "trans_amount_std": g["amount"].std(),
            "trans_amount_min": g["amount"].min(),
            "trans_amount_max": g["amount"].max(),
            "trans_amount_sum": g["amount"].sum(),
            "trans_balance_mean": g["balance"].mean(),
            "trans_balance_min": g["balance"].min(),
            "trans_balance_max": g["balance"].max(),
            "trans_balance_last": g["balance"].last(),
            "trans_date_min": g["trans_date"].min(),
            "trans_date_max": g["trans_date"].max(),
        }
    )
    agg["trans_date_span"] = agg["trans_date_max"] - agg["trans_date_min"]
    parts = [agg]
    for col in code_cols:
        parts.append(
            _fractions_by_code(trans, group_key, col, domains["trans"][col]["size"], "trans")
        )
    return pd.concat(parts, axis=1)


def _berka_order_agg(order, domains, code_cols=ORDER_CODE_COLS):
    g = order.groupby("account_id")
    agg = pd.DataFrame(
        {
            "order_n": g.size(),
            "order_amount_sum": g["amount"].sum(),
            "order_amount_mean": g["amount"].mean(),
            "order_amount_max": g["amount"].max(),
        }
    )
    parts = [agg]
    for col in code_cols:
        parts.append(
            _fractions_by_code(order, "account_id", col, domains["order"][col]["size"], "order")
        )
    return pd.concat(parts, axis=1)


def _berka_client_agg(disp, client, domains):
    """Account-holder demographics, aggregated over the account's dispositions.

    Berka marks one disposition per account as OWNER, but picking it out means trusting a code
    that synthetic data is free to permute, so we aggregate over all of an account's clients
    rather than guess. The `disp_type` shares carry the same information in a form that is well
    defined on real and synthetic tables alike.
    """
    joined = disp.merge(client, on="client_id", how="left", suffixes=("", "_client"))
    g = joined.groupby("account_id")
    agg = pd.DataFrame(
        {
            "n_disps": g.size(),
            "client_year_min": g["year"].min(),
            "client_year_max": g["year"].max(),
            "client_month_mean": g["month"].mean(),
            "client_gender_mean": g["gender"].mean(),
        }
    )
    fracs = _fractions_by_code(
        disp, "account_id", "disp_type", domains["disp"]["disp_type"]["size"], "disp"
    )
    return pd.concat([agg, fracs], axis=1)


# District `A2` is the district name: 77 levels over a 511-row training set is an identity
# column, not a feature. `A3` (8 regions) carries the geography that generalises.
BERKA_DISTRICT_DROP = ["A2"]


def _berka_district_features(district, domains, prefix="dist"):
    cols = [c for c in district.columns if c != "district_id" and c not in BERKA_DISTRICT_DROP]
    out = district[["district_id"] + cols].copy()
    out.columns = ["district_id"] + [f"{prefix}_{c}" for c in cols]
    cat = {
        f"{prefix}_{c}": domains["district"][c]["size"]
        for c in cols
        if domains["district"][c]["type"] == "discrete"
    }
    return out, cat


# ------------------------------------------------------------------------------------------
# Berka A: loan default, predicted at origination
# ------------------------------------------------------------------------------------------
#
# Transaction aggregates are restricted to transactions strictly before the loan date, both
# dates on one day count (`berka_date_offset`). Without that filter the task is not a prediction
# at all: `trans_balance_min` alone separates the classes perfectly (single-feature test AUROC
# 1.0000), because an account going into debt is what `status` D means, and 'SANKC. UROK'
# interest is charged as a consequence. Filtering leaves a median of 77 prior transactions per
# loan, every loan has some history, and 'SANKC. UROK' remains in 29 of the 80 loan accounts that
# ever pay it: sanction interest charged before the loan was granted, which a lender would see.
# The strongest single feature reaches about 0.79 test AUROC. Until 5 Oct 2026 the cut compared
# the raw stored dates, which count from different days, so it fell 185 days early (median 30
# transactions, 88 of 682 loans with none).
#
# Standing orders carry no timestamp so they cannot be filtered. That is acceptable: every row
# in this task has a loan, so the 'UVER' repayment order is near-constant across rows, and its
# amount duplicates the `payments` column that is already a legitimate origination-time feature.


def build_berka_loan(tables, domains):
    loan = tables["loan"].copy()
    y = loan["status"].isin(DEFAULT_STATUS).astype(int)

    account = tables["account"][["account_id", "district_id", "account_date", "frequency"]]
    X = loan[["loan_id", "account_id", "loan_date", "amount", "duration", "payments"]].merge(
        account, on="account_id", how="left"
    )
    X["amount_per_payment"] = X["amount"] / X["payments"].replace(0, np.nan)
    X["amount_per_month"] = X["amount"] / X["duration"].replace(0, np.nan)
    X["account_age_at_loan"] = (X["loan_date"] + berka_date_offset("loan")) - (
        X["account_date"] + berka_date_offset("account")
    )

    dist, dist_cat = _berka_district_features(tables["district"], domains)
    X = X.merge(dist, on="district_id", how="left")

    # Group by loan_id, not account_id: the structure sampler is free to give a synthetic
    # account more than one loan, and each loan needs its own cutoff.
    prior = tables["trans"].merge(
        loan[["loan_id", "account_id", "loan_date"]], on="account_id", how="inner"
    )
    prior = prior[
        prior["trans_date"] + berka_date_offset("trans")
        < prior["loan_date"] + berka_date_offset("loan")
    ]
    X = X.merge(
        _berka_trans_agg(prior, domains, group_key="loan_id"),
        left_on="loan_id",
        right_index=True,
        how="left",
    )

    for agg in (
        _berka_order_agg(tables["order"], domains),
        _berka_client_agg(tables["disp"], tables["client"], domains),
    ):
        X = X.merge(agg, left_on="account_id", right_index=True, how="left")

    X = X.drop(columns=["loan_id", "account_id", "district_id"])
    cat_spec = {"frequency": domains["account"]["frequency"]["size"], **dist_cat}
    return X, y, cat_spec


# ------------------------------------------------------------------------------------------
# Berka B: card ownership -- a cross-table dependency probe, not a forecast
# ------------------------------------------------------------------------------------------
#
# `card` has no usable origination time for negatives (`issued` exists only where a card exists),
# and cards are issued a median of 658.5 days after the account opens with only 21% inside the
# first year (dates on one day count; the 348 days and 51% quoted here before 5 Oct 2026 compared
# raw stored dates), so no time window separates cause from effect. This task is therefore scored as a
# dependency-recovery probe: can a model trained on synthetic data recover an association that
# is strong in the real database? Account behaviour here is partly a consequence of holding a
# card, and that is stated rather than hidden.
#
# The `operation` share vector is still dropped, on a different ground: 'VYBER KARTOU' is a
# *deterministic* function of the label (P(card | any card withdrawal) = 1), which makes the
# task degenerate rather than informative. It alone scored 0.9278 single-feature test AUROC.
CARD_TRANS_CODE_COLS = tuple(c for c in TRANS_CODE_COLS if c != "operation")


def build_berka_card(tables, domains):
    disp = tables["disp"].copy()
    y = disp["disp_id"].isin(set(tables["card"]["disp_id"])).astype(int)

    client = tables["client"][
        ["client_id", "district_id", "year", "month", "client_date", "gender"]
    ]
    account = tables["account"][["account_id", "district_id", "account_date", "frequency"]]

    X = disp[["disp_id", "client_id", "account_id", "disp_type"]].merge(
        client, on="client_id", how="left"
    )
    X = X.rename(columns={"district_id": "client_district_id"})
    X = X.merge(account, on="account_id", how="left")
    X["same_district"] = (X["client_district_id"] == X["district_id"]).astype(int)

    dist, dist_cat = _berka_district_features(tables["district"], domains)
    X = X.merge(dist, on="district_id", how="left")

    for agg in (
        _berka_trans_agg(tables["trans"], domains, code_cols=CARD_TRANS_CODE_COLS),
        _berka_order_agg(tables["order"], domains),
        _berka_client_agg(tables["disp"], tables["client"], domains),
    ):
        X = X.merge(agg, left_on="account_id", right_index=True, how="left")

    X = X.drop(
        columns=["disp_id", "client_id", "account_id", "district_id", "client_district_id"]
    )
    cat_spec = {
        "disp_type": domains["disp"]["disp_type"]["size"],
        "gender": domains["client"]["gender"]["size"],
        "year": domains["client"]["year"]["size"],
        "month": domains["client"]["month"]["size"],
        "client_date": domains["client"]["client_date"]["size"],
        "frequency": domains["account"]["frequency"]["size"],
        **dist_cat,
    }
    return X, y, cat_spec


# ------------------------------------------------------------------------------------------
# California: individual-level tasks over a household join + sibling aggregates
# ------------------------------------------------------------------------------------------


def _check_no_target_leak(agg_sources, target):
    offenders = [name for name, src in agg_sources.items() if src == target]
    if offenders:
        raise AssertionError(
            f"aggregate(s) {offenders} are derived from the target column {target!r}"
        )


def _california_matrix(tables, domains, target):
    individual, household = tables["individual"], tables["household"]

    hh_cols = [c for c in household.columns if c != "household_id"]
    ind_cols = [c for c in individual.columns if c not in ("individual_id", "household_id")]

    g = individual.groupby("household_id")
    agg = pd.DataFrame(
        {
            "hh_size": g.size(),
            "hh_age_mean": g["AGE"].mean(),
            "hh_age_max": g["AGE"].max(),
            "hh_age_min": g["AGE"].min(),
            "hh_educ_max": g["EDUC"].max(),
            "hh_n_female": g["SEX"].sum(),
        }
    )
    agg["hh_n_minor"] = (
        individual.assign(_m=individual["AGE"] < 18).groupby("household_id")["_m"].sum()
    )
    agg_sources = {
        "hh_size": None,
        "hh_age_mean": "AGE",
        "hh_age_max": "AGE",
        "hh_age_min": "AGE",
        "hh_educ_max": "EDUC",
        "hh_n_female": "SEX",
        "hh_n_minor": "AGE",
    }
    if target != "INCTOT":
        agg["hh_inctot_mean"] = g["INCTOT"].mean()
        agg["hh_inctot_sum"] = g["INCTOT"].sum()
        agg_sources.update({"hh_inctot_mean": "INCTOT", "hh_inctot_sum": "INCTOT"})
    _check_no_target_leak(agg_sources, target)

    X = individual[["household_id"] + ind_cols].merge(
        household[["household_id"] + hh_cols], on="household_id", how="left"
    )
    X = X.merge(agg, left_on="household_id", right_index=True, how="left")

    y = X[target].copy()
    X = X.drop(columns=["household_id", target])

    cat_spec = {}
    for table, cols in (("individual", ind_cols), ("household", hh_cols)):
        for c in cols:
            if c != target and domains[table][c]["type"] == "discrete":
                cat_spec[c] = domains[table][c]["size"]
    return X, y, cat_spec


def build_california_inctot(tables, domains):
    X, y, cat = _california_matrix(tables, domains, "INCTOT")
    return X, y.astype(float), cat


def build_california_empstat(tables, domains):
    X, y, cat = _california_matrix(tables, domains, "EMPSTAT")
    return X, pd.Series(np.rint(y.astype(float)).astype(int)), cat


# ------------------------------------------------------------------------------------------
# registry
# ------------------------------------------------------------------------------------------


@dataclass
class Task:
    name: str
    dataset: str
    kind: str  # binclass | multiclass | regression
    target: str  # human-readable description
    target_col: str  # column that must never appear in X (empty when the label is derived)
    build: Callable
    primary: str
    n_classes: int = 0
    metrics: List[str] = field(default_factory=list)


BINARY_METRICS = ["roc_auc", "average_precision", "f1_macro", "balanced_accuracy", "accuracy"]

TASKS: Dict[str, Task] = {
    "berka_loan_default": Task(
        name="berka_loan_default",
        dataset="berka",
        kind="binclass",
        target="loan.status in {B,D}, predicted at origination",
        target_col="status",
        build=build_berka_loan,
        primary="roc_auc",
        n_classes=2,
        metrics=BINARY_METRICS,
    ),
    "berka_card_ownership": Task(
        name="berka_card_ownership",
        dataset="berka",
        kind="binclass",
        target="disp has a card (dependency probe)",
        target_col="",
        build=build_berka_card,
        primary="roc_auc",
        n_classes=2,
        metrics=BINARY_METRICS,
    ),
    "california_inctot": Task(
        name="california_inctot",
        dataset="california",
        kind="regression",
        target="individual.INCTOT (income quantile bin, 0-99)",
        target_col="INCTOT",
        build=build_california_inctot,
        primary="r2",
        metrics=["r2", "rmse", "mae"],
    ),
    "california_empstat": Task(
        name="california_empstat",
        dataset="california",
        kind="multiclass",
        target="individual.EMPSTAT (4 classes)",
        target_col="EMPSTAT",
        build=build_california_empstat,
        primary="f1_macro",
        n_classes=4,
        metrics=["accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr"],
    ),
}
