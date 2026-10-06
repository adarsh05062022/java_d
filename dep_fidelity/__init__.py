"""Dependency-fidelity metrics for the Berka relational database.

Standalone from `downstream/`: this measures whether GRDM's synthetic Berka
tables preserve known inter-column dependencies (temporal ordering across a
foreign-key chain, an account-level financial invariant, and categorical
values that are near-deterministic consequences of another table's state),
independent of any downstream ML task.

Two layers:

  * **Measurement** -- `checks.py` / `run.py`, written up in `docs/dep_fidelity/REPORT.md`. Every
    check runs identically on real and synthetic tables, and the real-data rate
    is the ceiling a synthetic database is read against (it is not always 100%).
  * **Enforcement** -- `repair.py` / `run_repair.py` restore a violated
    dependency on a finished sample (Tier 1, post-hoc projection), and
    `reparam.py` re-encodes child dates as lags so a temporal violation cannot
    be expressed at all (Tier 3). Written up in `docs/dep_fidelity/REPAIR.md`. Neither needs
    retraining.
"""
