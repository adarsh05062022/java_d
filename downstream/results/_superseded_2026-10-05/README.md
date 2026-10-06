# Superseded results, 5 October 2026

Copies of every result file that the 5 October 2026 fixes regenerated, kept so the numbers in the
4 October report can still be traced to their source.

Two fixes made these stale:

1. **Loan-default window.** `downstream/tasks.py` (and `downstream/relmlp/features.py`) compared
   `trans_date < loan_date` on the raw stored values. Berka stores each date column as days since
   that table's own earliest date (transactions 1 Jan 1993, loans 5 Jul 1993), so the cut fell
   185 days before each loan. Every loan-default number in these files used that shortened window.
2. **Signed retention.** `feature_compare.py`, `feature_provenance.py`, `aggregate_validation.py`
   and `diagnose_aggregates.py` measured retention as a ratio of |AUROC − 0.5| (or |corr|), which
   counts a reversed relationship as kept. They now keep the sign.

Card-ownership downstream scores did not change and were not copied. `fix_comparison_final.json`
is an August summary of S1–S5 that nothing reads; it was moved here rather than copied.
These files are not read by any script (the results globs are not recursive).
