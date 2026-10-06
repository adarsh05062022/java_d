"""Task 6: build a Berka variant whose parent tables carry explicit relational aggregates.

    conda run --no-capture-output -n tabsyn python -m downstream.build_agg_dataset

This changes the *input representation* only -- no architecture, training or evaluation change.
Six account-level rollups over the child tables are denormalised into the schema as ordinary
continuous columns, so GRDM diffuses them like any other attribute and conditions children on
them like any other parent feature:

    agg_trans_n              fan-out: how many transactions this account has
    agg_trans_balance_mean   the feature worth 0.838 univariate AUROC in real data
    agg_trans_balance_last   the one that keeps its marginal (KS 0.045) and loses its meaning
    agg_trans_amount_mean
    agg_order_n              fan-out of the second child table
    agg_order_amount_sum

**They are written onto `account` *and* `disp`.** GRDM runs with `n_hops=1`, and the card label
lives on `card`, whose only parent is `disp`; account is two hops away and therefore outside the
conditioning window. Denormalising the same six columns onto `disp` is what puts account
behaviour inside the window when a card row is generated. `loan` and `trans` hang off `account`
directly, so one hop already reaches them.

No operation-code shares are included. 'VYBER KARTOU' is a deterministic consequence of holding
a card, and the downstream card task drops it for that reason -- exposing it here would hand the
generator the label.

The downstream evaluation never reads these columns: its feature builders select account and disp
columns by name, and rebuild every aggregate from the *generated* child tables. So an improvement
requires GRDM to actually emit a consistent child table, which is the hypothesis under test.
"""

import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT

SRC = os.path.join(REPO_ROOT, "data/berka/split_train")
DST = os.path.join(REPO_ROOT, "data/berka/split_train_agg")

AGG_COLS = [
    "agg_trans_n",
    "agg_trans_balance_mean",
    "agg_trans_balance_last",
    "agg_trans_amount_mean",
    "agg_order_n",
    "agg_order_amount_sum",
]


def account_aggregates(trans, order, account_ids):
    ordered = trans.sort_values(["account_id", "trans_date"])
    g = ordered.groupby("account_id")
    agg = pd.DataFrame(
        {
            "agg_trans_n": g.size(),
            "agg_trans_balance_mean": g["balance"].mean(),
            "agg_trans_balance_last": g["balance"].last(),
            "agg_trans_amount_mean": g["amount"].mean(),
        }
    )
    go = order.groupby("account_id")
    agg_o = pd.DataFrame({"agg_order_n": go.size(), "agg_order_amount_sum": go["amount"].sum()})
    out = pd.DataFrame(index=pd.Index(account_ids, name="account_id"))
    out = out.join(agg).join(agg_o)
    # An account with no standing orders genuinely has zero of them; that is a value, not a gap.
    out[["agg_order_n", "agg_order_amount_sum"]] = out[
        ["agg_order_n", "agg_order_amount_sum"]
    ].fillna(0.0)
    assert out.notna().all().all(), "unexpected NaN in aggregates"
    return out.astype(float).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    args = p.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    print(f"copied {args.src} -> {args.dst}")

    account = pd.read_csv(os.path.join(args.src, "account.csv"))
    disp = pd.read_csv(os.path.join(args.src, "disp.csv"))
    trans = pd.read_csv(os.path.join(args.src, "trans.csv"))
    order = pd.read_csv(os.path.join(args.src, "order.csv"))

    agg = account_aggregates(trans, order, account["account_id"].to_numpy())
    print(f"\naggregates over {len(trans):,} trans / {len(order):,} order rows "
          f"for {len(agg):,} accounts")
    print(agg[AGG_COLS].describe().loc[["mean", "std", "min", "max"]].T.to_string())

    account_out = account.merge(agg, on="account_id", how="left", validate="one_to_one")
    disp_out = disp.merge(agg, on="account_id", how="left", validate="many_to_one")
    assert account_out[AGG_COLS].notna().all().all()
    assert disp_out[AGG_COLS].notna().all().all()
    assert len(account_out) == len(account) and len(disp_out) == len(disp)

    account_out.to_csv(os.path.join(args.dst, "account.csv"), index=False)
    disp_out.to_csv(os.path.join(args.dst, "disp.csv"), index=False)

    # Columns absent from the domain file are treated as ID columns by `utils.sort_columns`,
    # so registering them as continuous is what makes them features rather than keys.
    for table in ("account", "disp"):
        path = os.path.join(args.dst, f"{table}_domain.json")
        dom = json.load(open(path))
        for c in AGG_COLS:
            dom[c] = {"size": 0, "type": "continuous"}
        json.dump(dom, open(path, "w"))
        print(f"\n{table}_domain.json -> {list(dom)}")

    print(f"\naccount: {len(account.columns)} -> {len(account_out.columns)} columns")
    print(f"disp:    {len(disp.columns)} -> {len(disp_out.columns)} columns")
    print(f"\nwrote {args.dst}")


if __name__ == "__main__":
    main()
