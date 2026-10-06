"""Fixed-hyperparameter XGBoost fitting and metric computation.

Hyperparameters are identical across every regime and never tuned. Per-regime tuning would make
the regimes incomparable, and an early-stopping set would have to be drawn from real data, which
would leak real rows into the synthetic-only regime.
"""

import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

# XGBoost is the one CPU-bound stage here, so its thread count is what keeps a process inside
# the per-process CPU budget while several experiments share the node. Override with
# RDBDIFF_XGB_JOBS; keep OMP_NUM_THREADS in step with it.
N_JOBS = int(os.environ.get("RDBDIFF_XGB_JOBS", "2"))

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    n_jobs=N_JOBS,
)


def _expand_proba(proba, model_classes, n_classes):
    """Widen a probability matrix back to the full class set.

    A regime whose training data lost a class entirely -- the failure mode we expect on Berka,
    where synthetic `loan.status` class B collapses from 31 rows to 2 -- yields a narrower
    `predict_proba`. Zero-filling the missing columns keeps the metric well defined and lets the
    collapse show up as a score drop rather than a crash.
    """
    full = np.zeros((proba.shape[0], n_classes))
    for i, c in enumerate(model_classes):
        full[:, int(c)] = proba[:, i]
    return full


def fit_predict(kind, X_train, y_train, X_test, seed, n_classes=0):
    """Return (y_pred, y_score). `y_score` is None for regression."""
    if kind == "regression":
        model = XGBRegressor(**XGB_PARAMS, random_state=seed, eval_metric="rmse")
        model.fit(X_train, y_train)
        return model.predict(X_test), None

    classes = np.unique(y_train)
    if len(classes) < 2:
        # Degenerate training set: nothing to fit. Predict the single class everywhere.
        only = int(classes[0])
        pred = np.full(len(X_test), only)
        score = np.zeros((len(X_test), n_classes))
        score[:, only] = 1.0
        return pred, (score[:, 1] if kind == "binclass" else score)

    params = dict(XGB_PARAMS)
    if kind == "binclass":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        eval_metric = "logloss"
    else:
        eval_metric = "mlogloss"

    model = XGBClassifier(**params, random_state=seed, eval_metric=eval_metric)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    full = _expand_proba(proba, model.classes_, max(n_classes, len(classes)))
    pred = full.argmax(axis=1)
    return pred, (full[:, 1] if kind == "binclass" else full)


def compute_metrics(kind, y_true, y_pred, y_score, n_classes=0):
    if kind == "regression":
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": mean_absolute_error(y_true, y_pred),
        }
    if kind == "binclass":
        return {
            "roc_auc": roc_auc_score(y_true, y_score),
            "average_precision": average_precision_score(y_true, y_score),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }
    labels = list(range(n_classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "roc_auc_ovr": roc_auc_score(
            y_true, y_score, multi_class="ovr", average="macro", labels=labels
        ),
    }


def bootstrap_ci(kind, y_true, y_pred, y_score, metric, n_boot=1000, seed=0, n_classes=0):
    """Percentile CI for one metric, resampling the test set.

    Berka's test sets are small (171 loans, 19 of them defaults), so the spread that matters is
    test-set sampling noise, not fit noise. This measures it directly and applies unchanged to
    the interim and the retrained runs.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score) if y_score is not None else None
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if kind != "regression" and len(np.unique(y_true[idx])) < 2:
            continue
        try:
            m = compute_metrics(
                kind,
                y_true[idx],
                y_pred[idx],
                y_score[idx] if y_score is not None else None,
                n_classes,
            )
        except ValueError:
            continue
        values.append(m[metric])
    if not values:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
