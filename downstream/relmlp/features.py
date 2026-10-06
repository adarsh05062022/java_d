"""Relational sample construction for the DeepSets probe.

Unlike downstream/tasks.py, nothing here is aggregated. Each training example is a parent row
plus the raw child rows that hang off it; the model (downstream/relmlp/model.py) learns the
pooling. The only relational logic carried over from tasks.py is what guards against leakage,
not what summarises:

  * berka_loan_default restricts transaction children to those strictly before the loan date,
    per loan, with both dates on one day count (tasks.berka_date_offset). Without the cut
    `trans_balance` alone separates the classes (tasks.py: single-feature AUROC 1.0000).
  * berka_card_ownership drops the `operation` column from transaction children:
    'VYBER KARTOU' (code 5) is a deterministic function of the label, P(card | withdrawal)=1.
  * District `A2` (77-level identity column over a 511-row train set) is dropped; `A3`
    (8 regions) is kept.

Everything downstream/tasks.py hand-rolls -- the 13 numeric trans summaries, the per-code share
vectors, the client demographics rolled up to the account -- is gone, replaced by a learned set
encoder over these raw rows.

A builder takes a `tables` dict (already passed through `coerce_discrete`, so discrete columns
are int codes) and the real-domain `domains` dict, and returns a `RelData`. The column lists and
categorical cardinalities come from the real domain files, so real / synthetic / test frames are
always shape-compatible; out-of-domain synthetic codes are folded into an <unk> slot at tensor
build time (downstream/relmlp/model.py).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from downstream.splits import DEFAULT_STATUS
from downstream.tasks import berka_date_offset

DISTRICT_DROP = ("A2",)


@dataclass
class ChildBlock:
    """One child table, flattened across the whole batch.

    `group[i]` is the parent-row index (into `RelData.parent_*`, range [0, B)) that child row
    `i` belongs to. The model pools rows sharing a `group` value.
    """

    num: np.ndarray  # [T, n_num]  float32
    cat: np.ndarray  # [T, n_cat]  int64
    group: np.ndarray  # [T]        int64 in [0, B)
    num_cols: List[str]
    cat_cols: List[str]
    cat_cards: List[int]


@dataclass
class RelData:
    parent_num: np.ndarray  # [B, Pn] float32
    parent_cat: np.ndarray  # [B, Pc] int64
    parent_num_cols: List[str]
    parent_cat_cols: List[str]
    parent_cat_cards: List[int]
    children: Dict[str, ChildBlock]
    y: np.ndarray  # [B] float32, {0., 1.}
    ids: np.ndarray  # [B] parent id, for asserts / debugging

    def __len__(self):
        return len(self.y)


# ------------------------------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------------------------------


def _district_split(domains):
    """(continuous district cols, discrete district cols), with the identity column removed."""
    cols = [c for c in domains["district"] if c not in DISTRICT_DROP]
    num = [c for c in cols if domains["district"][c]["type"] == "continuous"]
    cat = [c for c in cols if domains["district"][c]["type"] == "discrete"]
    return num, cat


def _num(df, cols):
    return np.column_stack([df[c].to_numpy(dtype="float64") for c in cols]).astype("float32")


def _cat(df, cols):
    """Integer code matrix. NaN (an unresolved join) becomes -1, which the model maps to <unk>."""
    out = np.full((len(df), len(cols)), -1, dtype="int64")
    for j, c in enumerate(cols):
        v = df[c].to_numpy(dtype="float64")
        ok = np.isfinite(v)
        out[ok, j] = np.rint(v[ok]).astype("int64")
    return out


def _group_index(child_key_series, parent_ids):
    """Map each child row's parent id onto the parent's row position; drop unmatched rows."""
    pos = {pid: i for i, pid in enumerate(parent_ids.tolist())}
    g = child_key_series.map(pos).to_numpy()
    keep = np.isfinite(g.astype("float64"))
    return g, keep


# ------------------------------------------------------------------------------------------
# Berka A: loan default at origination
# ------------------------------------------------------------------------------------------


def build_berka_loan_rel(tables, domains):
    loan = tables["loan"].reset_index(drop=True)
    account = tables["account"].set_index("account_id")
    district = tables["district"].set_index("district_id")
    dist_num, dist_cat = _district_split(domains)

    parent_ids = loan["loan_id"].to_numpy()
    acc = account.reindex(loan["account_id"].to_numpy()).reset_index()
    dist = district.reindex(acc["district_id"].to_numpy()).reset_index()

    parent_num_cols = ["loan_amount", "loan_duration", "loan_payments", "loan_date",
                       "account_date"] + [f"dist_{c}" for c in dist_num]
    parent_num = np.column_stack([
        loan[["amount", "duration", "payments", "loan_date"]].to_numpy("float64"),
        acc["account_date"].to_numpy("float64"),
        dist[dist_num].to_numpy("float64"),
    ]).astype("float32")

    parent_cat_cols = ["frequency", "dist_A3"]
    parent_cat_cards = [domains["account"]["frequency"]["size"],
                        domains["district"]["A3"]["size"]]
    parent_cat = np.column_stack([
        _cat(acc, ["frequency"]), _cat(dist, dist_cat),
    ]).astype("int64")

    y = loan["status"].isin(DEFAULT_STATUS).to_numpy().astype("float32")

    children = {}

    # transactions strictly before the loan date, grouped per loan (a synthetic account may
    # carry more than one loan, each with its own cutoff).
    tr = tables["trans"].merge(
        loan[["loan_id", "account_id", "loan_date"]], on="account_id", how="inner"
    )
    tr = tr[
        tr["trans_date"] + berka_date_offset("trans") < tr["loan_date"] + berka_date_offset("loan")
    ]
    g, keep = _group_index(tr["loan_id"], parent_ids)
    tr = tr[keep]
    tcols_num, tcols_cat = ["trans_date", "amount", "balance"], ["trans_type", "operation",
                                                                 "k_symbol", "bank"]
    children["trans"] = ChildBlock(
        _num(tr, tcols_num), _cat(tr, tcols_cat), g[keep].astype("int64"),
        tcols_num, tcols_cat, [domains["trans"][c]["size"] for c in tcols_cat],
    )

    # standing orders and account dispositions carry no timestamp, so they are kept whole and
    # attached to every loan on the account.
    od = tables["order"].merge(loan[["loan_id", "account_id"]], on="account_id", how="inner")
    g, keep = _group_index(od["loan_id"], parent_ids)
    od = od[keep]
    children["order"] = ChildBlock(
        _num(od, ["amount"]), _cat(od, ["bank_to", "k_symbol"]), g[keep].astype("int64"),
        ["amount"], ["bank_to", "k_symbol"],
        [domains["order"]["bank_to"]["size"], domains["order"]["k_symbol"]["size"]],
    )

    sib = tables["disp"][["account_id", "client_id", "disp_type"]].merge(
        tables["client"][["client_id", "year", "month", "client_date", "gender"]],
        on="client_id", how="left",
    ).merge(loan[["loan_id", "account_id"]], on="account_id", how="inner")
    g, keep = _group_index(sib["loan_id"], parent_ids)
    sib = sib[keep]
    children["client"] = ChildBlock(
        _num(sib, ["year"]), _cat(sib, ["month", "gender", "client_date", "disp_type"]),
        g[keep].astype("int64"),
        ["year"], ["month", "gender", "client_date", "disp_type"],
        [domains["client"]["month"]["size"], domains["client"]["gender"]["size"],
         domains["client"]["client_date"]["size"], domains["disp"]["disp_type"]["size"]],
    )

    return RelData(parent_num, parent_cat, parent_num_cols, parent_cat_cols, parent_cat_cards,
                   children, y, parent_ids)


# ------------------------------------------------------------------------------------------
# Berka B: card ownership -- a cross-table dependency probe
# ------------------------------------------------------------------------------------------


def build_berka_card_rel(tables, domains):
    disp = tables["disp"].reset_index(drop=True)
    client = tables["client"].set_index("client_id")
    account = tables["account"].set_index("account_id")
    district = tables["district"].set_index("district_id")
    dist_num, dist_cat = _district_split(domains)

    parent_ids = disp["disp_id"].to_numpy()
    cl = client.reindex(disp["client_id"].to_numpy()).reset_index()
    acc = account.reindex(disp["account_id"].to_numpy()).reset_index()
    dist = district.reindex(acc["district_id"].to_numpy()).reset_index()
    same_district = (cl["district_id"].to_numpy() == acc["district_id"].to_numpy()).astype("float32")

    parent_num_cols = ["client_year", "account_date", "same_district"] + [f"dist_{c}" for c in dist_num]
    parent_num = np.column_stack([
        cl["year"].to_numpy("float64"),
        acc["account_date"].to_numpy("float64"),
        same_district.astype("float64"),
        dist[dist_num].to_numpy("float64"),
    ]).astype("float32")

    parent_cat_cols = ["disp_type", "client_month", "client_gender", "client_date",
                       "frequency", "dist_A3"]
    parent_cat_cards = [domains["disp"]["disp_type"]["size"], domains["client"]["month"]["size"],
                        domains["client"]["gender"]["size"], domains["client"]["client_date"]["size"],
                        domains["account"]["frequency"]["size"], domains["district"]["A3"]["size"]]
    parent_cat = np.column_stack([
        _cat(disp, ["disp_type"]),
        _cat(cl, ["month", "gender", "client_date"]),
        _cat(acc, ["frequency"]),
        _cat(dist, dist_cat),
    ]).astype("int64")

    y = disp["disp_id"].isin(set(tables["card"]["disp_id"])).to_numpy().astype("float32")

    children = {}

    # No usable origination time for negatives, so no cutoff. `operation` is dropped: it is a
    # deterministic tell of the label.
    tr = tables["trans"].merge(disp[["disp_id", "account_id"]], on="account_id", how="inner")
    g, keep = _group_index(tr["disp_id"], parent_ids)
    tr = tr[keep]
    tcols_num, tcols_cat = ["trans_date", "amount", "balance"], ["trans_type", "k_symbol", "bank"]
    children["trans"] = ChildBlock(
        _num(tr, tcols_num), _cat(tr, tcols_cat), g[keep].astype("int64"),
        tcols_num, tcols_cat, [domains["trans"][c]["size"] for c in tcols_cat],
    )

    od = tables["order"].merge(disp[["disp_id", "account_id"]], on="account_id", how="inner")
    g, keep = _group_index(od["disp_id"], parent_ids)
    od = od[keep]
    children["order"] = ChildBlock(
        _num(od, ["amount"]), _cat(od, ["bank_to", "k_symbol"]), g[keep].astype("int64"),
        ["amount"], ["bank_to", "k_symbol"],
        [domains["order"]["bank_to"]["size"], domains["order"]["k_symbol"]["size"]],
    )

    sib = tables["disp"][["account_id", "client_id", "disp_type"]].merge(
        tables["client"][["client_id", "year", "month", "client_date", "gender"]],
        on="client_id", how="left",
    ).merge(disp[["disp_id", "account_id"]], on="account_id", how="inner")
    g, keep = _group_index(sib["disp_id"], parent_ids)
    sib = sib[keep]
    children["client"] = ChildBlock(
        _num(sib, ["year"]), _cat(sib, ["month", "gender", "client_date", "disp_type"]),
        g[keep].astype("int64"),
        ["year"], ["month", "gender", "client_date", "disp_type"],
        [domains["client"]["month"]["size"], domains["client"]["gender"]["size"],
         domains["client"]["client_date"]["size"], domains["disp"]["disp_type"]["size"]],
    )

    return RelData(parent_num, parent_cat, parent_num_cols, parent_cat_cols, parent_cat_cards,
                   children, y, parent_ids)


BUILDERS: Dict[str, Callable] = {
    "berka_loan_default": build_berka_loan_rel,
    "berka_card_ownership": build_berka_card_rel,
}


# ------------------------------------------------------------------------------------------
# row-count manipulation (mirrors downstream/run.py's regime construction)
# ------------------------------------------------------------------------------------------


def subsample_rel(rel: RelData, n: int, rng) -> RelData:
    """Keep `n` parent rows and the child rows beneath them, renumbering `group`."""
    if n >= len(rel):
        return rel
    keep = np.sort(rng.choice(len(rel), size=n, replace=False))
    remap = np.full(len(rel), -1, dtype="int64")
    remap[keep] = np.arange(n)
    children = {}
    for name, cb in rel.children.items():
        m = remap[cb.group] >= 0
        children[name] = ChildBlock(cb.num[m], cb.cat[m], remap[cb.group[m]],
                                    cb.num_cols, cb.cat_cols, cb.cat_cards)
    return RelData(rel.parent_num[keep], rel.parent_cat[keep], rel.parent_num_cols,
                   rel.parent_cat_cols, rel.parent_cat_cards, children, rel.y[keep], rel.ids[keep])


def concat_rel(a: RelData, b: RelData) -> RelData:
    """Stack two RelData (real + synthetic for the `mixed` regime), offsetting b's groups."""
    off = len(a)
    children = {}
    for name in a.children:
        ca, cb = a.children[name], b.children[name]
        children[name] = ChildBlock(
            np.concatenate([ca.num, cb.num]), np.concatenate([ca.cat, cb.cat]),
            np.concatenate([ca.group, cb.group + off]), ca.num_cols, ca.cat_cols, ca.cat_cards,
        )
    return RelData(
        np.concatenate([a.parent_num, b.parent_num]),
        np.concatenate([a.parent_cat, b.parent_cat]),
        a.parent_num_cols, a.parent_cat_cols, a.parent_cat_cards, children,
        np.concatenate([a.y, b.y]), np.concatenate([a.ids, b.ids]),
    )
