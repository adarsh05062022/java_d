"""Relational train/test splits.

California already ships a household-level split (`ClavaDDPM/complex_data/california/{train,test}`)
that GRDM honoured during training, so nothing is computed here for it.

Berka has no split, so we cut it at the `account` hub. The cardinalities make this clean:
`client` <-> `disp` is strictly 1:1 (5369 each, `disp.client_id` unique), `loan` is 1-per-account,
`card` is 1-per-disp. So carrying an account's descendants never orphans a row and never leaves a
client straddling both sides. `district` is a dimension table (GRDM holds it fixed at t=0 and
conditions on it) and is kept whole on both sides.
"""

from sklearn.model_selection import train_test_split

# Fixed so that the eval-time split and the split the retrained generator saw are the same cut.
BERKA_SPLIT_SEED = 0
BERKA_TEST_FRAC = 0.25

# Berka `status` codes are label-encoded A,B,C,D -> 0,1,2,3 (counts 203/31/403/45).
# B = finished but unpaid, D = running and in debt: the two default outcomes.
DEFAULT_STATUS = (1, 3)


def berka_account_split(tables, test_frac=BERKA_TEST_FRAC, seed=BERKA_SPLIT_SEED):
    """Return (train_account_ids, test_account_ids), stratified on the loan outcome.

    Strata are no-loan / loan-repaid / loan-defaulted. Loans exist for only 682 of 4500 accounts
    and only 76 of those defaulted, so an unstratified 25% cut leaves the number of defaulted test
    loans to chance -- seed 0 gave 13 where the proportional share is 19. Stratifying on the
    outcome pins it. Splitting on the label is standard and leaks nothing: the model never sees
    test labels.
    """
    account = tables["account"]
    defaulted = set(tables["loan"].loc[tables["loan"]["status"].isin(DEFAULT_STATUS), "account_id"])
    with_loan = set(tables["loan"]["account_id"])

    stratum = account["account_id"].map(
        lambda a: 2 if a in defaulted else (1 if a in with_loan else 0)
    )
    train_ids, test_ids = train_test_split(
        account["account_id"].to_numpy(),
        test_size=test_frac,
        random_state=seed,
        stratify=stratum.to_numpy(),
    )
    return set(train_ids.tolist()), set(test_ids.tolist())


def subset_berka(tables, account_ids):
    """Slice a Berka database down to the given accounts and everything beneath them."""
    account = tables["account"][tables["account"]["account_id"].isin(account_ids)]
    disp = tables["disp"][tables["disp"]["account_id"].isin(account_ids)]
    client = tables["client"][tables["client"]["client_id"].isin(set(disp["client_id"]))]
    card = tables["card"][tables["card"]["disp_id"].isin(set(disp["disp_id"]))]

    out = {
        "district": tables["district"].copy(),  # dimension table: whole on both sides
        "account": account.reset_index(drop=True),
        "client": client.reset_index(drop=True),
        "disp": disp.reset_index(drop=True),
        "card": card.reset_index(drop=True),
    }
    for t in ("loan", "order", "trans"):
        out[t] = tables[t][tables[t]["account_id"].isin(account_ids)].reset_index(drop=True)
    return out


def split_berka(tables, test_frac=BERKA_TEST_FRAC, seed=BERKA_SPLIT_SEED):
    train_ids, test_ids = berka_account_split(tables, test_frac, seed)
    return subset_berka(tables, train_ids), subset_berka(tables, test_ids)
