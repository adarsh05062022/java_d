"""Tier 3: make temporal violations unrepresentable, instead of repairing them.

Tier 1 (`repair.py`) removes violations after generation. This module removes the
*ability to express* one. A child event's date is stored not as an absolute day but
as `log1p(days since the parent account opened)`. Decoding is

    child_date = parent_date + expm1(max(value, 0))

so every decodable value maps to a lag of at least zero. A generator working in this
space cannot emit a transaction that precedes its own account, however badly it models
the column -- the constraint is carried by the representation rather than learned from
data.

Why this is the right shape of fix for GRDM specifically: the relational structure is
sampled *before* any attribute is generated, so at decode time every row already knows
which account it belongs to and that account's date is already decoded. The parent's
value is available exactly when the child needs it. This is the same argument that makes
S10's degree features legitimate rather than leakage.

Two properties are checked in `verify()` below, both without training anything:

  * **Round-trip losslessness.** encode -> decode on the real database returns the
    original dates to floating-point tolerance. A transform that loses information
    would trade a constraint violation for a fidelity loss.
  * **Structural guarantee.** Replacing the encoded column with pure noise and decoding
    still yields a 0% violation rate, where the same noise in the current
    representation violates roughly half the time. That is the claim "the model cannot
    get this wrong", demonstrated rather than argued.

Decode order matters in general: a parent's own date must be decoded before its
children's lags are resolved. All three constraints here hang directly off `account`,
which is the root and is never reparameterized, so a single pass suffices.
"""

import numpy as np
import pandas as pd

from dep_fidelity.repair import TEMPORAL_SPECS, _parent_abs_date


def encode_tables(tables, epochs, specs=TEMPORAL_SPECS):
    """Replace each child date column with log1p(lag from its parent account)."""
    out = {t: df.copy() for t, df in tables.items()}
    report = []
    for spec in specs:
        child, col = spec["child"], spec["child_col"]
        parent_abs = _parent_abs_date(tables, epochs, spec)
        child_abs = tables[child][col].to_numpy(dtype=float) + epochs[child]
        lag = child_abs - parent_abs
        # A dangling FK has no parent to measure against; leave the raw value so the
        # transform stays total. Real Berka has none of these (verified in checks.py).
        dangling = ~np.isfinite(parent_abs)
        negative = np.isfinite(lag) & (lag < 0)
        encoded = np.where(dangling, child_abs, np.log1p(np.clip(lag, 0.0, None)))
        out[child][col] = encoded
        report.append(dict(check=spec["name"], column=f"{child}.{col}",
                           n=int(len(lag)), dangling=int(dangling.sum()),
                           clipped_negative=int(negative.sum())))
    return out, report


def decode_tables(tables, epochs, specs=TEMPORAL_SPECS, parent_tables=None):
    """Invert `encode_tables`. `parent_tables` supplies the (already decoded) parent
    rows when the child table is being decoded separately from its parent."""
    out = {t: df.copy() for t, df in tables.items()}
    src = parent_tables or tables
    for spec in specs:
        child, col = spec["child"], spec["child_col"]
        lookup = {**src, child: tables[child]}
        parent_abs = _parent_abs_date(lookup, epochs, spec)
        value = tables[child][col].to_numpy(dtype=float)
        lag = np.expm1(np.clip(value, 0.0, None))
        decoded = np.where(np.isfinite(parent_abs), parent_abs + lag, value)
        out[child][col] = decoded - epochs[child]
    return out


def _violation_rate(tables, epochs, spec):
    parent_abs = _parent_abs_date(tables, epochs, spec)
    child_abs = tables[spec["child"]][spec["child_col"]].to_numpy(dtype=float) + epochs[spec["child"]]
    lag = child_abs - parent_abs
    finite = np.isfinite(lag)
    return float((lag[finite] < 0).mean()) if finite.any() else float("nan")


def verify(tables, epochs, rng=None):
    """Round-trip losslessness and the structural guarantee, on whatever DB is passed."""
    rng = rng or np.random.default_rng(0)
    encoded, report = encode_tables(tables, epochs)
    decoded = decode_tables(encoded, epochs, parent_tables=tables)

    rows = []
    for spec in TEMPORAL_SPECS:
        child, col = spec["child"], spec["child_col"]
        a = tables[child][col].to_numpy(dtype=float)
        b = decoded[child][col].to_numpy(dtype=float)
        max_err = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0

        # Structural guarantee: pure noise in each representation, decoded, compared.
        # The noise is matched to the moments of the column it replaces, so the two
        # representations are given equally bad generators.
        noisy_reparam = {t: df.copy() for t, df in encoded.items()}
        enc = encoded[child][col].to_numpy(dtype=float)
        noisy_reparam[child][col] = rng.normal(np.nanmean(enc), np.nanstd(enc), size=len(enc))
        rate_reparam = _violation_rate(
            decode_tables(noisy_reparam, epochs, parent_tables=tables), epochs, spec
        )

        noisy_raw = {t: df.copy() for t, df in tables.items()}
        raw = tables[child][col].to_numpy(dtype=float)
        noisy_raw[child][col] = rng.normal(np.nanmean(raw), np.nanstd(raw), size=len(raw))
        rate_raw = _violation_rate(noisy_raw, epochs, spec)

        rows.append(dict(check=spec["name"], column=f"{child}.{col}",
                         roundtrip_max_abs_error=max_err,
                         noise_violation_rate_current=rate_raw,
                         noise_violation_rate_reparam=rate_reparam))
    return rows, report


def main():
    import argparse
    import json
    import os
    import shutil

    from dep_fidelity.load import TABLES, load_real

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default=os.path.join(repo, "data/berka/split_train"))
    p.add_argument("--write-dataset", default=None,
                   help="emit a reparameterized copy of the dataset directory, ready to train on")
    args = p.parse_args()

    tables, epochs = load_real(args.real_dir)
    rows, report = verify(tables, epochs)

    print("=== encode report ===")
    for r in report:
        print(f"{r['column']:18s} n={r['n']:>9,}  dangling={r['dangling']:>3}  "
              f"negative_lags_clipped={r['clipped_negative']:>3}")

    print("\n=== round-trip losslessness (real data) ===")
    for r in rows:
        print(f"{r['column']:18s} max |decode(encode(x)) - x| = {r['roundtrip_max_abs_error']:.6g} days")

    print("\n=== structural guarantee: decode pure noise in each representation ===")
    print(f"{'column':18s} {'current repr':>14s} {'reparameterized':>17s}")
    for r in rows:
        print(f"{r['column']:18s} {r['noise_violation_rate_current']:13.1%} "
              f"{r['noise_violation_rate_reparam']:16.1%}")

    if args.write_dataset:
        os.makedirs(args.write_dataset, exist_ok=True)
        for fname in os.listdir(args.real_dir):
            src = os.path.join(args.real_dir, fname)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(args.write_dataset, fname))
        encoded, _ = encode_tables(tables, epochs)
        for t in TABLES:
            encoded[t].to_csv(os.path.join(args.write_dataset, f"{t}.csv"), index=False)
        with open(os.path.join(args.write_dataset, "reparam_manifest.json"), "w") as f:
            json.dump(dict(source=args.real_dir, transform="log1p_lag_from_parent",
                           columns=[r["column"] for r in report]), f, indent=2)
        print(f"\nWrote reparameterized dataset to {args.write_dataset}")
        print("Column names, domain files and dtypes are unchanged, so it is drop-in for "
              "training; decode with `decode_tables` after sampling.")


if __name__ == "__main__":
    main()
