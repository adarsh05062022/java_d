# DeepSets downstream probe (`downstream/relmlp/`)

A second downstream estimator that does **not** hand-build relational features. Where
`downstream/tasks.py` rolls the child tables up into ~60 aggregate columns and feeds them to
XGBoost, this probe feeds the **raw child rows** to a small permutation-invariant network and
lets it learn the pooling.

It exists to answer one question: *is the real ≫ synthetic utility gap an artifact of the
hand-picked aggregates, or does it survive a model that learns aggregation end-to-end?*

## What is kept from `downstream/tasks.py`

Only the leakage guards, never the summaries:

| guard | why |
|---|---|
| `trans_date < loan_date`, per loan | `trans_balance` alone separates the loan classes otherwise (single-feature AUROC 1.0000) |
| `operation` dropped from card-task transaction children | `VYBER KARTOU` (code 5) is P(card)=1 |
| district `A2` dropped, `A3` kept | 77-level identity column over a 511-row train set |

The join graph itself comes from `dataset_meta.json`. Everything else the XGBoost path
engineers — 13 numeric trans summaries, per-code share vectors, client demographics rolled to
the account — is replaced by the learned set encoder.

## Model

`features.py` turns each example into `(parent row, {child table: raw child rows})`.
`model.py`:

```
RowEncoder:  numeric (standardised) + categorical (embeddings) -> 2-layer MLP -> h
pool:        mean(child h) ++ max(child h)   per parent          <- learned aggregate
head:        [parent h | pooled trans | pooled order | pooled client] -> MLP -> 1 logit
```

Hyperparameters are fixed (`d_hidden=32`, `d_emb=8`, dropout 0.1, Adam lr 1e-3, wd 1e-4,
**300 epochs, full batch, no early stopping**) for the same reason the XGBoost params are
fixed: per-regime tuning breaks comparability, and an early-stopping split drawn from real data
would leak real rows into the synthetic-only regime. Optimisation variance is measured by
repeating over `--train-seeds` and also reported as a 5-seed probability ensemble.

Metrics and bootstrap CIs are imported from `downstream/evaluate.py` unchanged, so the numbers
line up 1:1 with the XGBoost run.

## Run

```bash
conda run --no-capture-output -n tabsyn python -m downstream.relmlp.run \
    --dataset berka_split --grdm-seeds 0 1 2 --train-seeds 0 1 2 3 4
```

`--regimes` accepts `real`, `synth`, `mixed` (default `real synth` — the load-bearing
comparison). `--device` defaults to `cpu` (the model is tiny; CPU avoids GPU contention).
Results: `downstream/results_relmlp/<dataset>__<task>.json`. Summarise with
`python -m downstream.relmlp.summarize`.

## Scope

Berka only (`berka_loan_default`, `berka_card_ownership`), both binclass. California would need
a regression head and a household/sibling builder; not implemented.
