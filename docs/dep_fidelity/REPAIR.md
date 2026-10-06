# Enforcing the dependencies, not just measuring them

> **Note, 5 Oct 2026.** Loan-default numbers in this document were computed with a date window that ended 185 days before each loan (raw stored dates compared across tables, which count from different days). The corrected numbers are in `reports/grdm-final-report.html` and `docs/downstream/RESULTS_downstream.md`; card-ownership numbers are unaffected.

`REPORT.md` established that GRDM's synthetic Berka database violates the temporal,
financial and logical dependencies the real database satisfies. This file is the other
half: what it takes to **restore** them, what each repair costs, and which of the
resulting numbers can be trusted.

Two tiers are implemented. Tier 1 (`repair.py`) projects a finished sample onto the
constraint-satisfying set after generation. Tier 3 (`reparam.py`) changes the data
representation so that one whole class of violation cannot be expressed at all. Neither
requires retraining; both run on cached samples in seconds.

```
conda run -n tabsyn python -m dep_fidelity.run_repair --dataset berka_split --seeds 0 1 2 --write
conda run -n tabsyn python -m dep_fidelity.reparam
```

---

## 1. The guard: every operator is near-identity on real data

A repair that changes the real database is mis-specified, and any synthetic number it
produces is unreadable. `run_repair.py` applies every operator to the real database
first and aborts if anything moves. Distances are Wasserstein in units of the real
standard deviation (continuous) or total variation (discrete); 0 means identical.

| column | distance from real after repairing *real* data |
|---|---:|
| `loan.loan_date` | 0.0000 |
| `card.issued` | 0.0000 |
| `trans.trans_date` | 0.0000 |
| `trans.balance` | **0.0304** |
| `trans.operation` | 0.0000 |
| `trans.k_symbol` | 0.0000 |
| `order.k_symbol` | 0.0000 |

Six of seven are exactly identity. `trans.balance` is not, and the residual is
informative rather than a bug: rederiving the balance series from the amounts requires
an order for same-day transactions, which a day-granularity `trans_date` cannot supply.
0.0304 sd is the price of that ambiguity, and it bounds how good the ledger operator can
ever be.

## 2. Tier 1 results — Berka, `berka_split`, 3 seeds

Repairs run in dependency order (temporal -> ledger -> logical) because they interact:
rewriting `trans_date` changes the order the ledger accumulates in, and recomputing
`balance` changes which accounts are ever overdrawn, which is the antecedent of the
SANKC. UROK rule. The full check suite is re-run afterwards, so an operator breaking
another constraint would surface.

| check | metric | real | synthetic | repaired |
|---|---|---:|---:|---:|
| `loan_date ≥ account_date` | violation rate | 0.0% | 11.7% | **0.0%** |
| | median lag (days) | 401.0 | 405.4 | 451.3 |
| | p10 lag (days) | 166.0 | −37.7 | **145.8** |
| `card.issued ≥ account_date` | violation rate | 0.0% | 9.6% | **0.0%** |
| | median lag | 645.5 | 828.7 | 877.3 |
| | p10 lag | 278.0 | 26.5 | **272.3** |
| `trans_date ≥ account_date` | violation rate | 0.0% | 3.3% | **0.0%** |
| | median lag | 683.0 | 723.9 | 745.6 |
| | p10 lag | 188.0 | 130.1 | **175.6** |
| final balance = Σ signed amounts | consistency | 98.3% | 0.0% | **100.0%** |
| | median abs error | 0.2 | 98,388 | **0.0** |
| `balance[i] − balance[i−1] = amount[i]` | consistency | 68.7% | 0.02% | **100.0%** |
| VYBER KARTOU ⟹ holds a card | implication | 100.0% | 19.2% | **100.0%** |
| SANKC. UROK ⟹ ever overdrawn | implication | 94.7% | 70.0% | **94.5%** |
| order UVER ⟹ has a loan | implication | 94.1% | 14.5% | **94.2%** |

Every constraint is restored, and the logical rules land on the **real** rate rather than
a perfect 100%: 94.5% against a real 94.7%, 94.2% against a real 94.1%. That is
deliberate. Repairing to 100% a rule that the real database satisfies 94% of the time
produces a database distinguishable from real by the very check meant to validate it.

The mechanism matters for reading these numbers. A repaired row has its *code* removed,
so it leaves the firing set as well as the violating set — the surviving denominator is
`satisfied + allowed`, not the original match count. Apportioning against the original
count instead undershoots badly wherever the synthetic rate starts far from real: the
UVER rule lands at 0.70 rather than 0.94, because 86% of its firing rows are violations
to begin with.

**Rows rewritten, mean per seed:**

| operator | rows |
|---|---:|
| temporal — `loan_date` | 59 |
| temporal — `card.issued` | 63 |
| temporal — `trans_date` | 25,793 |
| ledger — derive | 791,918 |
| implication — VYBER KARTOU | 4,810 |
| implication — SANKC. UROK | 6 |
| implication — order UVER | 443 |

## 3. What the repairs cost

Distance of each rewritten column from the real marginal, before and after.

| column | kind | before | after | change |
|---|---|---:|---:|---:|
| `loan.loan_date` | continuous | 0.1920 | 0.2472 | +0.055 |
| `card.issued` | continuous | 0.0514 | 0.2636 | +0.212 |
| `trans.trans_date` | continuous | 0.0873 | 0.1383 | +0.051 |
| `trans.balance` | continuous | 0.0090 | **3.8795** | **+3.870** |
| `trans.operation` | discrete | 0.0012 | 0.0066 | +0.005 |
| `trans.k_symbol` | discrete | 0.0033 | 0.0033 | 0.000 |
| `order.k_symbol` | discrete | 0.0221 | 0.0991 | +0.077 |

**Constraint satisfaction is not free, and one repair is far from free.** The temporal
and logical operators cost a few hundredths of a standard deviation. The ledger operator
costs 3.87 — the derived balance column is nearly four real standard deviations from the
real marginal, because it inherits every error in the `amount` column it accumulates.
The ledger identity is only ever as good as the amounts underneath it; making the
invariant exact does not make the numbers right.

## 4. Which temporal strategy: the marginal and the dependency disagree

Three ways to remove a date violation, all reaching 0%:

| strategy | p10 lag: loan / card / trans | `trans_date` marginal distance |
|---|---|---:|
| real | 166 / 278 / 188 | — |
| synthetic | −38 / 26 / 130 | 0.0873 |
| `clamp` | 0.9 / 27.7 / 130.1 | **0.0936** |
| `resample` | **145.8 / 272.3 / 175.6** | 0.1383 |
| `shift` | 145.8 / 272.3 / 260.4 | 0.3872 |

`clamp` sets each violating date to its parent's, which piles the repaired rows onto a
single day: the p10 lag collapses to ~1 day against a real 166. `resample` draws a fresh
lag from the real lag distribution and restores the left tail almost exactly (145.8 vs
166, 272.3 vs 278). `shift` moves a whole account's transaction series forward and
overshoots, because it displaces non-violating rows too.

Note that **the two axes disagree about the winner**: `clamp` has the *lower* marginal
distance (0.0936 vs 0.1383) and the *worse* dependency statistic. Judging a repair by
marginal fidelity alone would select the strategy that destroys the dependency. This is
the same lesson as the fidelity/utility gap in `docs/downstream/FIXES.md`, appearing in a
third place.

`resample` is the default.

## 5. An endpoint check is gameable; state the invariant over the whole history

`ledger_reconciles` states the financial invariant at the end of each account's history.
Two operators satisfy it identically:

| | account-level check | step-level check | `balance` distortion | rows edited |
|---|---:|---:|---:|---:|
| real | 98.3% | 68.7% | — | — |
| synthetic | 0.0% | 0.02% | 0.0090 | — |
| `final_only` | **100.0%** | 0.02% | **0.0223** | 3,375 |
| `derive` | **100.0%** | **100.0%** | 3.8795 | 791,918 |

`final_only` rewrites one row per account — the last — and passes the account-level check
perfectly, at 1/235th the edits and 175x less distortion. It has repaired nothing: every
intermediate balance is still inconsistent, and the step-level check stays at 0.02%.

This is why `ledger_step_consistency` was added to `checks.py` after the repair work, not
before it: building the repair is what exposed that the original check could be satisfied
without the ledger accumulating. **Any constraint stated over the endpoint of a sequence
has this weakness.** For an AmEx-style financial-consistency gate the practical
consequence is direct — a gate defined on statement closing balances can be passed by a
generator that never produced a coherent transaction sequence.

The real-data rate for the step check is **68.7%**, not 100%, and the median error is
exactly 0. The real ledger is consistent; what the check cannot do is order transactions
within a single day from a date column with day granularity. So `derive`'s 100% is
*above* the real ceiling — too clean, in the same way a 100% implication rate would be.

## 6. Tier 3 — making the violation unrepresentable

Rather than repairing dates after the fact, store a child event's date as
`log1p(days since the parent account opened)` and decode as

```
child_date = parent_date + expm1(max(value, 0))
```

Every decodable value is a non-negative lag, so a temporal violation cannot be expressed.
This fits GRDM specifically because its relational structure is sampled *before* any
attribute is generated: at decode time each row already knows its account, and the
account's date is already decoded. Same argument that makes S10's degree features
legitimate rather than leakage.

Two properties, both verified without training anything:

| column | round-trip max error | noise → violations, current repr | noise → violations, reparameterized |
|---|---:|---:|---:|
| `loan.loan_date` | **0 days** | 35.2% | **0.0%** |
| `card.issued` | **0 days** | 15.6% | **0.0%** |
| `trans.trans_date` | **0 days** | 15.5% | **0.0%** |

Encode-then-decode returns the real dates exactly, so the transform costs no information.
And a generator that models the column *as badly as pure Gaussian noise* still produces
zero temporal violations under the reparameterization, where the same noise in the
current representation violates 15–35% of the time. The constraint is carried by the
representation instead of learned from data.

`python -m dep_fidelity.reparam --write-dataset <dir>` emits a reparameterized copy of
the training data. Column names, domain files and dtypes are unchanged, so it is drop-in
for `main.py`; the remaining step is a training run, which the depth and architecture
experiments currently hold the GPUs for.

## 7. Reading all of this

Constraint enforcement is cheap, and it works: every dependency in the suite can be
restored to its real-data rate on a finished sample, in seconds, without retraining.
What it does **not** do is establish that the synthetic database is now good. Three
findings here point the same way:

- the repaired database satisfies the checks while its `balance` column sits four
  standard deviations from real,
- the strategy with the better marginal has the worse dependency statistic,
- and an operator that repairs nothing passes the account-level financial check.

Satisfaction of a constraint, fidelity of a marginal, and utility for a downstream task
are three different axes, and a repair can move them in three different directions.
Section 8 measures the third one directly.


## 8. Does repairing the constraints restore downstream utility? No.

The repaired databases are written as a `_repaired` variant of the baseline run, so
`downstream.run` scores them through exactly the same path fix S1 used — same split,
same test set, same fixed XGBoost hyperparameters. Synthetic-only AUROC, 3 generator
seeds x 5 XGBoost seeds:

| variant | loan default | card ownership |
|---|---|---|
| baseline | 0.5917 ± 0.044 (23.6%) | 0.6021 ± 0.017 (27.3%) |
| S1 · categorical decoding | 0.5963 ± 0.024 (24.8%) | 0.6009 ± 0.018 (27.0%) |
| **R1 · full repair** | 0.5730 ± 0.057 (18.8%) | 0.6028 ± 0.024 (27.5%) |
| **R1 · without the balance rewrite** | 0.5887 ± 0.072 (22.8%) | 0.6191 ± 0.023 (31.8%) |

**Restoring every dependency in the suite moves card ownership by +0.0007 AUROC and
moves loan default down.** A database that satisfies all eight checks at their real
rates is worth no more to a downstream model than one that violates all of them. This
is the S1 result again, reached by a completely different route: S1 repaired the most
visible *fidelity* symptom and changed nothing downstream; R1 repairs every *dependency*
and changes nothing downstream.

The two rows also separate the ledger operator's damage from the rest. Dropping the
`derive` rewrite — keeping the temporal and logical repairs, and satisfying the
account-level financial check the cheap way — recovers loan default from 18.8% to 22.8%.
That confirms the 3.87-sd distortion in section 3 is what costs the loan task: its
signal lives in transaction aggregates, and `balance` is one of them.

Card ownership under that same variant reads 31.8% against a 27.3% baseline. **Treat
this as suggestive, not established.** Per generator seed the effect is +0.033 / −0.015
/ +0.033: two seeds improve, one moves the other way, and the spread across seeds is
larger than the mean shift. It would need more seeds to call, and it is the only
positive downstream signal anywhere in this file.

### What that implies for the enforcement ladder

Tier 1 buys constraint satisfaction and nothing else. That is still worth having — an
AmEx-style Gate 4 is a pass/fail gate, and a database that fails it is unusable
regardless of its AUROC — but it should be presented as compliance, not as a fix for the
utility gap. The utility gap has a different cause, diagnosed in
`docs/downstream/DIAGNOSIS.md` as the conditioning window, and the fixes that move it (S3,
S5) are training-time changes.

The case for Tier 3 is stronger for exactly this reason. Post-hoc repair cannot teach
the generator anything: it edits outputs and the model stays as it was. A
reparameterization changes what the model is trained on, so the capacity it currently
spends failing to learn "children come after parents" is freed for the rest of the
distribution — and, unlike repair, it costs nothing at sampling time and distorts no
marginal. Section 6 shows the transform is lossless and the guarantee is structural;
what remains is a training run to find out whether the freed capacity is worth anything
downstream. That is the one open question this file cannot answer without a GPU.
