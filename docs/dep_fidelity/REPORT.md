# Dependency Fidelity on Berka

Does GRDM's synthetic Berka database preserve the dependencies between columns
that the real database has — not just per-column marginals or pairwise
contingency tables, but the specific temporal orderings and near-deterministic
logical implications that make Berka a real bank's ledger rather than seven
independently-plausible tables? This is deliberately **not** a downstream ML
task: no classifier is trained, so there is no conditioning-depth ceiling, no
seed-noise from gradient boosting, and no leakage-avoidance tradeoff to reason
about. Every check below is a closed-form statistic computed identically on
real and synthetic tables.

Code: `dep_fidelity/` (`load.py`, `checks.py`, `run.py`). Run with:

```
conda run -n tabsyn python -m dep_fidelity.run \
    --synthetic <run>/synthetic_database --synthetic <run2>/synthetic_database ... \
    --label <name>
```

## The eight checks

**Temporal ordering** (3 checks) — a child event's date must not precede its
parent's, across a foreign-key chain:
`loan_date >= account_date`, `card.issued >= account_date` (via `disp`),
`trans_date >= account_date`.

Berka's date columns are each stored as *days since that column's own
earliest observed date* (`account`/`trans` epoch to 1993-01-01, `loan` to
1993-07-05, `card` to 1993-11-07 — see `ClavaDDPM/preprocess_utils.py`).
Subtracting two tables' date columns directly is off by the gap between their
epochs; `load.py` converts every date to an absolute calendar day via the
`*_earliest_date.txt` sidecar before any comparison. (One side effect worth
flagging: `downstream/tasks.py`'s comment that cards are issued "a median of
348 days" after account opening used the uncorrected subtraction — the
epoch-aligned figure is 658.5 days. Harmless for that file's actual purpose,
since a tree model's split thresholds are shift-invariant per feature, but
worth knowing if anyone reads that comment as a literal calendar fact.)

**Financial consistency** (2 checks) — the same invariant stated at two scales.

`account.final_balance == sum(all of that account's signed transaction
amounts)` is account-level and order-independent: same-day transactions can't
be strictly ordered from `trans_date` alone (verified — naively trusting row
order as a same-day tiebreak drifts to a mean error of 887 over the full
transaction history; collapsing to one reconciliation per account sidesteps
the ambiguity entirely and still validates cleanly on real data, at 98.3%).

`balance[i] - balance[i-1] == signed_amount[i]` states it at every step
instead. It was added later, after the repair work in `REPAIR.md` showed that
the account-level form can be satisfied without repairing anything —
rewriting one row per account, the last, passes it while every intermediate
balance stays wrong. The price of the stronger statement is exactly the
ambiguity the account-level check was designed to avoid, and it is visible in
the number: real data scores **68.7%**, with a median error of 0. The real
ledger is consistent; what the check cannot do is order transactions inside a
single day. Read the two together — the account-level check has a clean real
ceiling but is gameable, the step-level check is not gameable but has a noisy
one.

**Causal/logical dependency** (3 checks) — a categorical code that is a
near-deterministic consequence of another table's state, expressed as
`P(target holds | code fires)`:
`operation=VYBER KARTOU -> account holds a card`,
`k_symbol=SANKC. UROK -> account was ever overdrawn`,
`order.k_symbol=UVER -> account has a loan`.
These are the same facts `downstream/tasks.py` documents and excludes from its
ML features as leakage (`operation=5`, `k_symbol=4` cause real-data AUROC to
hit 1.0 / 0.93 single-feature) — here they're the thing being measured
directly, not a confound to route around.

## Results — GRDM `paper_repro`, 3 seeds

| check | metric | real | synthetic (mean ± sd) |
|---|---|---:|---:|
| loan_date ≥ account_date | violation rate | **0.0%** | **10.5% ± 0.3pt** |
| | median lag (days) | 395.5 | 402.8 ± 5.3 |
| card.issued ≥ account_date | violation rate | **0.0%** | **10.3% ± 1.3pt** |
| | median lag (days) | 658.5 | 797.0 ± 17.4 |
| trans_date ≥ account_date | violation rate | **0.0%** | **3.1% ± 0.1pt** |
| | median lag (days) | 680.0 | 718.9 ± 8.0 |
| final balance = Σ signed amounts | consistency rate | **98.3%** | **0.0% ± 0.0pt** |
| | median abs. error | 0.2 | 99,717 ± 2,580 |
| VYBER KARTOU ⟹ holds a card | implication rate | **100.0%** | **19.2% ± 0.5pt** |
| SANKC. UROK ⟹ ever overdrawn | implication rate | **95.9%** | **67.2% ± 0.5pt** |
| order UVER ⟹ has a loan | implication rate | **95.1%** | **17.1% ± 0.9pt** |

Full per-seed numbers: `dep_fidelity/results/paper_repro.json`.

## Reading the results

**Financial consistency collapses completely, not partially.** Real accounts
reconcile at 98.3% (the residual 1.7% is sub-unit floating-point noise, median
error 0.2 currency units). Synthetic reconciles at 0.0% across all three
seeds, with a median error near 100,000 units — comparable to a typical
account's total balance. `amount` and `balance` are diffused as independent
continuous columns with nothing enforcing the ledger identity between them,
and the generator has no mechanism to acquire it. This is the sharpest,
least ambiguous result in the suite: an AmEx-style Gate 4 ("Financial
Consistency") check would fail every synthetic row it's run against.

**Temporal ordering degrades but does not collapse.** Violation rates of
3–10% (vs. 0% real) mean the generator has *mostly* learned "children come
after parents," but not as a hard constraint — and the magnitude also drifts
in the surviving cases (card lag +21% high, `p10_lag_days` for card swings
from +282 real to roughly zero synthetic, meaning the shortest real gaps
disappear into the violation mass). This is a softer failure than financial
consistency: partially learned as a statistical tendency, not enforced as a
constraint.

**Logical dependencies split into two regimes.** `operation=VYBER KARTOU`
and `order.k_symbol=UVER` both collapse to near their base rates (19.2% and
17.1%, vs. 100% and 95.1% real) — the generator reproduces each column's
marginal but not its cross-table logical necessity. `k_symbol=SANKC. UROK`
degrades much more gently (95.9% → 67.2%), retaining most of the dependency.
The difference is plausibly structural: SANKC. UROK's implied condition
("ever overdrawn") is itself a `trans`-table fact reachable within GRDM's own
conditioning window, while card ownership and loan existence live in a
different table reached only through a `disp`/`account` hop — consistent
with the conditioning-window mechanism `docs/downstream/DIAGNOSIS.md` documents
for the ML-task failures, surfacing here independently through a completely
different measurement.

## Extending this

Each check function in `checks.py` takes `(tables, epochs) -> dict` and is
independent of the other seven — adding a ninth is adding one function and a
line in `CHECKS`. The suite currently targets only Berka; porting it to
another schema means re-deriving the epoch/code constants for that schema
(the `_lag_stats` / `_implication_rate` helpers already generalize).

## Enforcing these dependencies

`REPAIR.md` is the other half of this work: restoring each dependency rather than
only measuring it. Every check above can be brought back to its real-data rate on a
finished sample in seconds without retraining, and the cost of doing so is measured
per column. Two findings there feed back into this file:

- `ledger_step_consistency` was added *after* the repair work, because building the
  repair exposed that the original account-level ledger check can be satisfied by
  rewriting one row per account while the ledger stays inconsistent. An endpoint
  check on a sequence is gameable; state the invariant over the whole history.
- Repairs must target the **real** rate, not 100%. Two of the three logical rules
  hold at ~94% in the real database and the step-level ledger check at 68.7%, so a
  repaired database that scores perfectly is distinguishable from real by the very
  check meant to validate it.
