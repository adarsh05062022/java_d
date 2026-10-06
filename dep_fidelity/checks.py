"""The dependency-fidelity checks.

Every check is a real, validated Berka dependency -- either a temporal
ordering constraint across a foreign-key chain, an account-level financial
invariant, or a categorical value that is a near-deterministic consequence of
another table's state (the same facts `downstream/tasks.py` excludes from its
ML features as leakage; here they are the thing being measured instead of the
thing being avoided). Every function has the signature `(tables, epochs) ->
dict`, runs unchanged on real or synthetic tables, and returns numbers on the
same scale so the two are directly comparable.

Real-data ceilings, computed against `data/berka/preprocessed` before any
synthetic table was involved:
  - loan_date >= account_date:               0.0% violations   (n=682)
  - card.issued >= account_date (via disp):  0.0% violations   (n=892)
  - trans_date >= account_date:              0.0% violations   (n=1,056,320)
  - account ledger reconciliation:           98.3% within 1 unit (n=4,500)
  - operation=VYBER KARTOU -> has card:      100.0%            (n=8,036)
  - k_symbol=SANKC. UROK -> ever overdrawn:   95.9%            (n=1,577)
  - order.k_symbol=UVER -> has loan:          95.1%            (n=717)
"""

import numpy as np
import pandas as pd

# trans.operation / trans.k_symbol / order.k_symbol codes, from the fitted
# LabelEncoders in data/berka/preprocessed/{trans,order}_label_encoders.pkl.
OP_VYBER_KARTOU = 5  # trans.operation: card withdrawal
KSYM_SANKC_UROK = 4  # trans.k_symbol: sanction interest, charged only once overdrawn
ORDER_KSYM_UVER = 4  # order.k_symbol: standing loan-repayment order


def _lag_stats(lag: pd.Series, n_dangling: int, label: str) -> dict:
    lag = lag.dropna()
    n = len(lag)
    return {
        "check": label,
        "n": int(n),
        "dangling_fk_dropped": int(n_dangling),
        "violation_rate": float((lag < 0).mean()) if n else float("nan"),
        "median_lag_days": float(lag.median()) if n else float("nan"),
        "p10_lag_days": float(lag.quantile(0.10)) if n else float("nan"),
    }


def loan_after_account(tables, epochs) -> dict:
    account, loan = tables["account"], tables["loan"]
    j = loan.merge(account[["account_id", "account_date"]], on="account_id", how="inner")
    lag = (j["loan_date"] + epochs["loan"]) - (j["account_date"] + epochs["account"])
    return _lag_stats(lag, len(loan) - len(j), "loan_date >= account_date")


def card_after_account(tables, epochs) -> dict:
    account, disp, card = tables["account"], tables["disp"], tables["card"]
    j = card.merge(disp[["disp_id", "account_id"]], on="disp_id", how="inner")
    j = j.merge(account[["account_id", "account_date"]], on="account_id", how="inner")
    lag = (j["issued"] + epochs["card"]) - (j["account_date"] + epochs["account"])
    return _lag_stats(lag, len(card) - len(j), "card.issued >= account_date")


def trans_after_account(tables, epochs) -> dict:
    account, trans = tables["account"], tables["trans"]
    j = trans.merge(account[["account_id", "account_date"]], on="account_id", how="inner")
    lag = (j["trans_date"] + epochs["trans"]) - (j["account_date"] + epochs["account"])
    return _lag_stats(lag, len(trans) - len(j), "trans_date >= account_date")


def ledger_reconciles(tables, epochs, tolerance: float = 1.0) -> dict:
    """Per account: final recorded balance == sum of every signed transaction amount.

    Order-independent by construction (sums the full history rather than
    replaying transactions one at a time), which sidesteps same-day
    transactions whose relative order the day-granularity `trans_date`
    cannot resolve. trans_type code 0 is 'PRIJEM' (credit, +); codes 1/2 are
    'VYBER'/'VYDAJ' (withdrawal, -) -- two codes for the same event, a known
    quirk of the original Berka data entry.
    """
    trans = tables["trans"]
    sign = np.where(trans["trans_type"] == 0, 1.0, -1.0)
    totals = trans.assign(signed=trans["amount"] * sign).groupby("account_id")["signed"].sum()

    last_date = trans.groupby("account_id")["trans_date"].transform("max")
    last_rows = trans.loc[trans["trans_date"] == last_date, ["account_id", "balance"]].copy()
    last_rows["total"] = last_rows["account_id"].map(totals)
    last_rows["err"] = (last_rows["balance"] - last_rows["total"]).abs()
    err = last_rows.groupby("account_id")["err"].min()

    return {
        "check": "account.final_balance == sum(trans.signed_amount)",
        "n": int(len(err)),
        "consistency_rate": float((err < tolerance).mean()),
        "median_abs_error": float(err.median()),
        "mean_abs_error": float(err.mean()),
    }


def ledger_step_consistency(tables, epochs, tolerance: float = 1.0) -> dict:
    """Per transaction: does the balance move by exactly this transaction's signed amount?

    `ledger_reconciles` above states the invariant at the end of each account's history,
    which makes it satisfiable without the ledger being consistent: rewriting one row per
    account -- the last -- passes it while every intermediate balance stays wrong. That
    is not hypothetical; `repair.py`'s `final_only` operator does exactly that, at 1/235th
    the edit cost of a real repair and with 175x less distortion to the balance marginal.

    This check states the same invariant at every step, so it can only be satisfied by a
    ledger that actually accumulates. Any endpoint-only constraint on a sequence has this
    weakness; the general lesson is to state the check over the whole history.
    """
    trans = tables["trans"]
    sign = np.where(trans["trans_type"] == 0, 1.0, -1.0)
    t = trans.assign(_signed=trans["amount"] * sign)
    t = t.sort_values(["account_id", "trans_date"], kind="stable")

    prev = t.groupby("account_id", sort=False)["balance"].shift(1)
    # An account's first transaction opens it, so the expected prior balance is zero.
    expected = prev.fillna(0.0) + t["_signed"]
    err = (t["balance"] - expected).abs()

    return {
        "check": "balance[i] - balance[i-1] == signed_amount[i]",
        "n": int(len(err)),
        "consistency_rate": float((err < tolerance).mean()),
        "median_abs_error": float(err.median()),
    }


def _implication_rate(matched: pd.DataFrame, group_key: str, target_ids: set, label: str) -> dict:
    n = len(matched)
    rate = float(matched[group_key].isin(target_ids).mean()) if n else float("nan")
    return {"check": label, "n": int(n), "implication_rate": rate}


def card_withdrawal_implies_card(tables, epochs) -> dict:
    trans, disp, card = tables["trans"], tables["disp"], tables["card"]
    card_accounts = set(disp.merge(card, on="disp_id")["account_id"])
    matched = trans[trans["operation"] == OP_VYBER_KARTOU]
    return _implication_rate(matched, "account_id", card_accounts,
                              "operation=VYBER KARTOU -> account holds a card")


def sanction_interest_implies_overdrawn(tables, epochs) -> dict:
    trans = tables["trans"]
    neg_accounts = set(trans.loc[trans["balance"] < 0, "account_id"])
    matched = trans[trans["k_symbol"] == KSYM_SANKC_UROK]
    return _implication_rate(matched, "account_id", neg_accounts,
                              "k_symbol=SANKC.UROK -> account ever overdrawn")


def loan_order_implies_loan(tables, epochs) -> dict:
    order, loan = tables["order"], tables["loan"]
    loan_accounts = set(loan["account_id"])
    matched = order[order["k_symbol"] == ORDER_KSYM_UVER]
    return _implication_rate(matched, "account_id", loan_accounts,
                              "order.k_symbol=UVER -> account has a loan")


CHECKS = [
    loan_after_account,
    card_after_account,
    trans_after_account,
    ledger_reconciles,
    ledger_step_consistency,
    card_withdrawal_implies_card,
    sanction_interest_implies_overdrawn,
    loan_order_implies_loan,
]


def run_all(tables: dict, epochs: dict) -> list:
    results = []
    for fn in CHECKS:
        try:
            results.append(fn(tables, epochs))
        except Exception as e:  # a malformed synthetic table should not kill the whole report
            results.append({"check": fn.__name__, "error": str(e)})
    return results
