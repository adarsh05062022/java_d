"""Fixed-budget training of one RelMLP and prediction on the real test set."""

import os

import numpy as np
import torch
import torch.nn as nn

from downstream.relmlp.model import RelMLP, RelSpec, Standardizer, to_tensors

# The training node is shared; keep intra-op parallelism modest so several runs coexist inside
# a per-process CPU budget. Mirrors RDBDIFF_XGB_JOBS on the XGBoost path.
torch.set_num_threads(int(os.environ.get("RELMLP_THREADS", "4")))

EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4


def prepare(train_rel, test_rel, device):
    """Standardise numerics on the training regime, move both splits to tensors."""
    std = Standardizer().fit(train_rel)
    tr = to_tensors(std.transform(train_rel), device)
    te = to_tensors(std.transform(test_rel), device)
    return tr, te, RelSpec.from_data(train_rel)


def fit_predict(train_batch, test_batch, spec, seed, epochs=EPOCHS, lr=LR,
                weight_decay=WEIGHT_DECAY, device="cpu"):
    """Return (y_pred int64 [n_test], y_score float64 [n_test] = P(class 1))."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    y = train_batch["y"]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    n_test = test_batch["parent_num"].shape[0]
    if n_pos == 0 or n_neg == 0:
        only = float(y[0].item()) if len(y) else 0.0
        score = np.full(n_test, only, dtype="float64")
        return score.astype("int64"), score

    model = RelMLP(spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(train_batch), y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(test_batch)).double().cpu().numpy()
    return (score >= 0.5).astype("int64"), score
