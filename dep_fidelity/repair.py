"""Constraint repair operators -- Tier 1 of the enforcement ladder.

`checks.py` measures whether a synthetic database satisfies Berka's temporal,
financial and logical dependencies. This module *restores* them, by projecting a
sampled database onto the constraint-satisfying set after generation. No
retraining and no resampling: every operator here runs on a finished
`synthetic_database/` directory in seconds.

Three principles, each of which is a result in its own right:

1. **Repair to the real rate, not to 100%.** Two of Berka's three logical rules
   hold at 95.9% / 95.1% in the real database, and the ledger reconciles at
   98.3%. A synthetic database repaired to a perfect 100% is *too clean* -- it
   would be distinguishable from real data by the very check meant to validate
   it. Every operator here takes its target from the measured real rate.

2. **Repairs interact, so they are ordered.** Rewriting `trans_date` changes the
   order the ledger accumulates in; recomputing `balance` changes which accounts
   are ever overdrawn, which is the antecedent of the SANKC. UROK rule. The
   fixed order is temporal -> ledger -> logical, and `run_repair.py` re-runs the
   full check suite afterwards to confirm no operator broke an earlier one.

3. **A repair is only trustworthy if it is near-identity on real data.** Every
   operator can be applied to the real database, where it should change almost
   nothing. `validate_on_real` in `run_repair.py` is that guard -- it is the same
   discipline as measuring the real-data ceiling before comparing to synthetic.

None of these operators touch an id column, so referential integrity is
preserved by construction (asserted in `run_repair.py`).
"""

import numpy as np
import pandas as pd

from dep_fidelity.checks import (
    KSYM_SANKC_UROK,
    OP_VYBER_KARTOU,
    ORDER_KSYM_UVER,
)

# --------------------------------------------------------------------------------------
# temporal ordering
# --------------------------------------------------------------------------------------

# Each child event must not precede its parent account's opening. `via` routes a
# child that reaches `account` through an intermediate table (card -> disp -> account).
TEMPORAL_SPECS = (
    dict(name="loan_date >= account_date", child="loan", child_col="loan_date", via=None),
    dict(name="card.issued >= account_date", child="card", child_col="issued",
         via=("disp", "disp_id", "account_id")),
    dict(name="trans_date >= account_date", child="trans", child_col="trans_date", via=None),
)


def _parent_abs_date(tables, epochs, spec):
    """Absolute account-opening day for every row of the child table (NaN if the FK dangles)."""
    account = tables["account"]
    pmap = pd.Series(
        (account["account_date"].to_numpy(dtype=float) + epochs["account"]),
        index=account["account_id"].to_numpy(),
    )
    child = tables[spec["child"]]
    if spec["via"] is None:
        keys = child["account_id"]
    else:
        via_table, via_key, via_fk = spec["via"]
        via = tables[via_table]
        vmap = pd.Series(via[via_fk].to_numpy(), index=via[via_key].to_numpy())
        keys = child[via_key].map(vmap)
    return keys.map(pmap).to_numpy(dtype=float)


def _real_lags(real_tables, real_epochs, spec):
    """The real database's lag distribution for this constraint, used by `resample`."""
    parent_abs = _parent_abs_date(real_tables, real_epochs, spec)
    child_abs = (
        real_tables[spec["child"]][spec["child_col"]].to_numpy(dtype=float)
        + real_epochs[spec["child"]]
    )
    lag = child_abs - parent_abs
    return lag[np.isfinite(lag) & (lag >= 0)]


def repair_temporal(tables, epochs, real_tables, real_epochs, strategy="resample", rng=None):
    """Remove child-before-parent date violations.

    strategy:
      ``clamp``     set the offending date to the parent's date (lag 0). Minimal edit,
                    but it piles every repaired row onto a single day -- the real p10
                    lags are 167/282/188 days, so the resulting spike at zero is a new
                    artefact that the `p10_lag_days` statistic will expose.
      ``resample``  draw a fresh lag for each violating row from the *real* lag
                    distribution. Same guarantee, and it preserves the shape of the
                    lag distribution rather than flattening its left tail.
      ``shift``     move the whole parent group forward by its worst violation, so that
                    within-group relative timing survives untouched. Meaningful only for
                    a child table with many rows per parent (`trans`); falls back to
                    ``resample`` elsewhere.
    """
    rng = rng or np.random.default_rng(0)
    tables = {t: df.copy() for t, df in tables.items()}
    stats = []

    for spec in TEMPORAL_SPECS:
        child_name, col = spec["child"], spec["child_col"]
        child = tables[child_name]
        parent_abs = _parent_abs_date(tables, epochs, spec)
        child_abs = child[col].to_numpy(dtype=float) + epochs[child_name]
        lag = child_abs - parent_abs

        violating = np.isfinite(lag) & (lag < 0)
        n_viol = int(violating.sum())
        before = float(np.nanmean(lag[np.isfinite(lag)] < 0)) if np.isfinite(lag).any() else np.nan

        if n_viol:
            mode = strategy
            if mode == "shift" and child_name not in ("trans",):
                mode = "resample"

            if mode == "clamp":
                new_lag = np.zeros(n_viol)
                child_abs[violating] = parent_abs[violating] + new_lag
            elif mode == "resample":
                pool = _real_lags(real_tables, real_epochs, spec)
                new_lag = rng.choice(pool, size=n_viol, replace=True)
                child_abs[violating] = parent_abs[violating] + new_lag
            elif mode == "shift":
                # Per account, the single offset that lifts the worst violation to zero.
                grp = child["account_id"].to_numpy()
                deficit = np.where(violating, -lag, 0.0)
                worst = pd.Series(deficit).groupby(grp).transform("max").to_numpy()
                moved = worst > 0
                child_abs[moved] = child_abs[moved] + worst[moved]
                n_viol = int(moved.sum())  # rows actually rewritten, not just violating ones
            else:
                raise ValueError(f"unknown strategy {strategy}")

            tables[child_name][col] = child_abs - epochs[child_name]

        lag_after = (
            tables[child_name][col].to_numpy(dtype=float) + epochs[child_name] - parent_abs
        )
        finite = np.isfinite(lag_after)
        stats.append(dict(
            check=spec["name"],
            operator=f"temporal:{strategy}",
            n=int(finite.sum()),
            rows_touched=n_viol,
            violation_rate_before=before,
            violation_rate_after=float((lag_after[finite] < 0).mean()) if finite.any() else np.nan,
        ))
    return tables, stats


# --------------------------------------------------------------------------------------
# financial consistency
# --------------------------------------------------------------------------------------


def repair_ledger(tables, epochs, strategy="derive"):
    """Restore `account.final_balance == sum(signed transaction amounts)`.

    strategy:
      ``derive``      recompute `balance` as the running total of signed amounts within
                      each account, ordered by date. This is the post-hoc form of the
                      Tier-3 fix: the invariant stops being something the model has to
                      learn and becomes an identity of the representation. It rewrites
                      every transaction's balance, so the balance marginal is no longer
                      the generator's -- it is whatever the amounts imply.
      ``final_only``  rewrite only the last transaction of each account. One row per
                      account instead of ~235, and the check passes -- but every
                      intermediate balance stays inconsistent. Included deliberately:
                      it demonstrates that a check defined at account level can be
                      satisfied without the ledger being repaired, which is an argument
                      for stating checks over the whole history rather than an endpoint.
    """
    tables = {t: df.copy() for t, df in tables.items()}
    trans = tables["trans"]

    sign = np.where(trans["trans_type"].to_numpy() == 0, 1.0, -1.0)
    signed = trans["amount"].to_numpy(dtype=float) * sign

    work = pd.DataFrame({
        "account_id": trans["account_id"].to_numpy(),
        "trans_date": trans["trans_date"].to_numpy(dtype=float),
        "signed": signed,
        "_pos": np.arange(len(trans)),
    })
    # `kind="stable"` keeps the emitted row order as the tiebreak for same-day
    # transactions, whose relative order the day-granularity date cannot resolve.
    work = work.sort_values(["account_id", "trans_date"], kind="stable")
    running = work.groupby("account_id", sort=False)["signed"].cumsum().to_numpy()

    balance_new = trans["balance"].to_numpy(dtype=float).copy()
    if strategy == "derive":
        balance_new[work["_pos"].to_numpy()] = running
        rows_touched = int(len(trans))
    elif strategy == "final_only":
        is_last = work.groupby("account_id", sort=False).cumcount(ascending=False) == 0
        pos = work.loc[is_last, "_pos"].to_numpy()
        balance_new[pos] = running[is_last.to_numpy()]
        rows_touched = int(is_last.sum())
    else:
        raise ValueError(f"unknown strategy {strategy}")

    tables["trans"]["balance"] = balance_new
    return tables, [dict(
        check="account.final_balance == sum(trans.signed_amount)",
        operator=f"ledger:{strategy}",
        n=int(trans["account_id"].nunique()),
        rows_touched=rows_touched,
    )]


# --------------------------------------------------------------------------------------
# logical implication
# --------------------------------------------------------------------------------------

IMPLICATION_SPECS = (
    dict(name="operation=VYBER KARTOU -> account holds a card",
         table="trans", col="operation", code=OP_VYBER_KARTOU, condition="has_card",
         group_col="trans_type"),
    dict(name="k_symbol=SANKC.UROK -> account ever overdrawn",
         table="trans", col="k_symbol", code=KSYM_SANKC_UROK, condition="overdrawn",
         group_col="trans_type"),
    dict(name="order.k_symbol=UVER -> account has a loan",
         table="order", col="k_symbol", code=ORDER_KSYM_UVER, condition="has_loan",
         group_col=None),
)


def _satisfying_accounts(tables, condition):
    if condition == "has_card":
        return set(tables["disp"].merge(tables["card"], on="disp_id")["account_id"])
    if condition == "overdrawn":
        return set(tables["trans"].loc[tables["trans"]["balance"] < 0, "account_id"])
    if condition == "has_loan":
        return set(tables["loan"]["account_id"])
    raise ValueError(condition)


def _replacement_codes(real_df, spec, group_values, rng):
    """Draw replacement codes from the real marginal, excluding the code being removed.

    Conditioned on `group_col` where one is given, so a withdrawal's replacement
    operation is drawn from the operations real withdrawals actually use rather
    than from the column marginal over every transaction type.
    """
    col, code, gcol = spec["col"], spec["code"], spec["group_col"]
    eligible = real_df[real_df[col] != code]
    n = len(group_values)
    out = np.empty(n, dtype="int64")

    if gcol is None or gcol not in real_df.columns:
        pool = eligible[col].to_numpy()
        return rng.choice(pool, size=n, replace=True) if len(pool) else np.full(n, code)

    global_pool = eligible[col].to_numpy()
    for g in np.unique(group_values):
        mask = group_values == g
        pool = eligible.loc[eligible[gcol] == g, col].to_numpy()
        if len(pool) == 0:
            pool = global_pool
        out[mask] = (
            rng.choice(pool, size=int(mask.sum()), replace=True)
            if len(pool) else code
        )
    return out


def repair_implications(tables, epochs, real_tables, real_epochs, rng=None, target="real"):
    """Bring each near-deterministic cross-table rule back to its real-data rate.

    A violating row is one whose code fires while the implied condition does not hold
    in another table. Two repairs are possible: remove the code, or create the implied
    entity. This takes the first -- creating a `card` row to justify a card withdrawal
    would fabricate the label of one of the downstream tasks, which is not a repair but
    a leak. Replacement codes are drawn from the real conditional marginal, so the
    column's distribution moves toward real rather than away from it.

    `target="real"` leaves exactly as many violations standing as the real database has
    (rules 2 and 3 hold at 95.9% / 95.1%, not 100%); `target="full"` repairs all of them.
    """
    rng = rng or np.random.default_rng(0)
    tables = {t: df.copy() for t, df in tables.items()}
    stats = []

    for spec in IMPLICATION_SPECS:
        tname, col, code = spec["table"], spec["col"], spec["code"]
        df = tables[tname]

        real_df = real_tables[tname]
        real_matched = real_df[real_df[col] == code]
        real_ok = _satisfying_accounts(real_tables, spec["condition"])
        real_rate = (
            float(real_matched["account_id"].isin(real_ok).mean()) if len(real_matched) else 1.0
        )

        ok_accounts = _satisfying_accounts(tables, spec["condition"])
        fires = (df[col] == code).to_numpy()
        satisfied = df["account_id"].isin(ok_accounts).to_numpy()
        violating = fires & ~satisfied
        n_matched, n_viol = int(fires.sum()), int(violating.sum())
        before = 1.0 - (n_viol / n_matched) if n_matched else np.nan

        # Leave enough violations standing that the *post-repair* rate equals the real
        # one, instead of an implausible 100%. Repairing a row removes the code, so it
        # leaves the firing set too: the surviving denominator is (satisfied + allowed),
        # not the original n_matched. Solving satisfied / (satisfied + allowed) = rate
        # gives the line below -- apportioning against n_matched instead undershoots
        # badly whenever the synthetic rate is far from real (UVER lands at 0.70, not
        # 0.94, because 86% of its firing rows are violations to begin with).
        n_sat = n_matched - n_viol
        if target == "real" and real_rate > 0:
            allowed = int(round(n_sat * (1.0 - real_rate) / real_rate))
        else:
            allowed = 0
        n_repair = max(0, n_viol - allowed)

        if n_repair:
            idx = np.flatnonzero(violating)
            chosen = rng.choice(idx, size=n_repair, replace=False)
            gvals = (
                df[spec["group_col"]].to_numpy()[chosen]
                if spec["group_col"] and spec["group_col"] in df.columns
                else np.zeros(n_repair)
            )
            df.loc[df.index[chosen], col] = _replacement_codes(real_df, spec, gvals, rng)
            tables[tname] = df

        fires_after = (tables[tname][col] == code).to_numpy()
        sat_after = tables[tname]["account_id"].isin(ok_accounts).to_numpy()
        n_after = int(fires_after.sum())
        stats.append(dict(
            check=spec["name"],
            operator=f"implication:{target}",
            n=n_after,
            rows_touched=n_repair,
            real_rate=real_rate,
            implication_rate_before=before,
            implication_rate_after=(
                float((fires_after & sat_after).sum() / n_after) if n_after else np.nan
            ),
        ))
    return tables, stats


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def repair_all(tables, epochs, real_tables, real_epochs, temporal="resample",
               ledger="derive", implication="real", seed=0):
    """Apply every operator in dependency order and return (tables, per-operator stats).

    The order is not cosmetic. `trans_date` determines the sequence the ledger
    accumulates in, so temporal repair must precede it; the recomputed `balance`
    determines which accounts are ever overdrawn, which is the antecedent of the
    SANKC. UROK rule, so ledger repair must precede the logical operators.
    """
    rng = np.random.default_rng(seed)
    stats = []
    tables, s = repair_temporal(tables, epochs, real_tables, real_epochs,
                                strategy=temporal, rng=rng)
    stats += s
    tables, s = repair_ledger(tables, epochs, strategy=ledger)
    stats += s
    tables, s = repair_implications(tables, epochs, real_tables, real_epochs,
                                    rng=rng, target=implication)
    stats += s
    return tables, stats
