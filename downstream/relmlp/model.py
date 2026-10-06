"""DeepSets relational classifier: per-row encoders + permutation-invariant pooling.

The parent row and every child row go through a small MLP (numeric features standardised
upstream, categoricals via learned embeddings). Child-row vectors are pooled per parent with
mean+max -- this is the learned replacement for the hand aggregates in downstream/tasks.py. The
head sees [parent_h | pooled_child_1 | ... ] and emits one logit.

Hyperparameters are fixed and never tuned, for the same reason downstream/evaluate.py fixes the
XGBoost ones: per-regime tuning would make the regimes incomparable, and an early-stopping split
drawn from real data would leak real rows into the synthetic-only regime. So training is a fixed
epoch budget, full-batch, identical for every regime; variance is measured by repeating over
`--train-seeds`.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

D_HIDDEN = 32
D_EMB = 8
P_DROP = 0.1


# ------------------------------------------------------------------------------------------
# numeric standardisation (fit on the training regime, applied to the fixed real test set)
# ------------------------------------------------------------------------------------------


def _mean_std(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape[0] == 0:
        return np.zeros(a.shape[1], "float32"), np.ones(a.shape[1], "float32")
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0)
    sd = np.where(~np.isfinite(sd) | (sd == 0), 1.0, sd)
    return mu.astype("float32"), sd.astype("float32")


def _apply(a: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.nan_to_num((a - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


class Standardizer:
    def fit(self, rel):
        self.p = _mean_std(rel.parent_num)
        self.c = {name: _mean_std(cb.num) for name, cb in rel.children.items()}
        return self

    def transform(self, rel):
        children = {
            name: replace(cb, num=_apply(cb.num, *self.c[name]))
            for name, cb in rel.children.items()
        }
        return replace(rel, parent_num=_apply(rel.parent_num, *self.p), children=children)


# ------------------------------------------------------------------------------------------
# RelData -> tensors
# ------------------------------------------------------------------------------------------


def _fold_oov(cat: np.ndarray, cards: List[int]) -> np.ndarray:
    """Codes outside [0, card) -- including the -1 from an unresolved join, and any
    out-of-domain synthetic value -- go to index `card`, a dedicated <unk> embedding slot."""
    out = cat.copy()
    for j, c in enumerate(cards):
        col = out[:, j]
        col[(col < 0) | (col >= c)] = c
        out[:, j] = col
    return out


def to_tensors(rel, device) -> Dict:
    b = {
        "parent_num": torch.as_tensor(rel.parent_num, dtype=torch.float32, device=device),
        "parent_cat": torch.as_tensor(_fold_oov(rel.parent_cat, rel.parent_cat_cards),
                                      dtype=torch.long, device=device),
        "y": torch.as_tensor(rel.y, dtype=torch.float32, device=device),
        "children": {},
    }
    for name, cb in rel.children.items():
        b["children"][name] = {
            "num": torch.as_tensor(cb.num, dtype=torch.float32, device=device),
            "cat": torch.as_tensor(_fold_oov(cb.cat, cb.cat_cards), dtype=torch.long, device=device),
            "group": torch.as_tensor(cb.group, dtype=torch.long, device=device),
        }
    return b


# ------------------------------------------------------------------------------------------
# spec + modules
# ------------------------------------------------------------------------------------------


@dataclass
class RelSpec:
    parent_num: int
    parent_cards: List[int]
    children: Dict[str, Tuple[int, List[int]]]  # name -> (n_num, cards)

    @classmethod
    def from_data(cls, rel) -> "RelSpec":
        return cls(
            rel.parent_num.shape[1],
            list(rel.parent_cat_cards),
            {n: (cb.num.shape[1], list(cb.cat_cards)) for n, cb in rel.children.items()},
        )


def scatter_stats(x: torch.Tensor, group: torch.Tensor, B: int) -> torch.Tensor:
    """Pool rows of `x` [T, d] by `group` into [B, 2d+1] = [mean | std | log1p(count)].

    All three moments go through `index_add_`, whose backward is a plain `index_select` -- fast
    on CPU. `scatter_reduce(reduce="amax")` was the earlier choice and its CPU backward is
    pathologically slow (minutes per epoch here). std carries the spread that max was standing
    in for, and log1p(count) keeps child cardinality, which mean alone discards.

    Empty groups fall out clean: sum 0 over a clamped count of 1 gives mean 0 and std ~0, and
    log1p(0) = 0. Nothing here is an in-place write on a grad-tracked tensor.
    """
    d = x.shape[1]
    if x.shape[0] == 0:
        return x.new_zeros(B, 2 * d + 1)
    cnt = x.new_zeros(B).index_add_(0, group, x.new_ones(x.shape[0]))
    denom = cnt.clamp(min=1.0).unsqueeze(1)
    mean = x.new_zeros(B, d).index_add_(0, group, x) / denom
    meansq = x.new_zeros(B, d).index_add_(0, group, x * x) / denom
    std = (meansq - mean * mean).clamp(min=0.0).add(1e-6).sqrt()
    return torch.cat([mean, std, torch.log1p(cnt).unsqueeze(1)], dim=1)


class RowEncoder(nn.Module):
    def __init__(self, n_num: int, cards: List[int], d_hidden=D_HIDDEN, d_emb=D_EMB, p=P_DROP):
        super().__init__()
        self.n_num = n_num
        self.embs = nn.ModuleList([nn.Embedding(c + 1, d_emb) for c in cards])
        in_dim = n_num + d_emb * len(cards)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        parts = [num] if self.n_num else []
        for i, emb in enumerate(self.embs):
            parts.append(emb(cat[:, i]))
        return self.mlp(torch.cat(parts, dim=1))


class RelMLP(nn.Module):
    def __init__(self, spec: RelSpec, d_hidden=D_HIDDEN, d_emb=D_EMB, p=P_DROP):
        super().__init__()
        self.d_hidden = d_hidden
        self.parent_enc = RowEncoder(spec.parent_num, spec.parent_cards, d_hidden, d_emb, p)
        self.child_names = list(spec.children)
        self.child_encs = nn.ModuleDict({
            name: RowEncoder(n, cards, d_hidden, d_emb, p)
            for name, (n, cards) in spec.children.items()
        })
        # parent -> d_hidden; each child -> mean|std|log1p(count) = 2*d_hidden + 1
        head_in = d_hidden + (2 * d_hidden + 1) * len(self.child_names)
        self.head = nn.Sequential(
            nn.Linear(head_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        B = batch["parent_num"].shape[0]
        h = [self.parent_enc(batch["parent_num"], batch["parent_cat"])]
        for name in self.child_names:
            cb = batch["children"][name]
            rows = self.child_encs[name](cb["num"], cb["cat"]) if cb["num"].shape[0] else \
                cb["num"].new_zeros(0, self.d_hidden)
            h.append(scatter_stats(rows, cb["group"], B))
        return self.head(torch.cat(h, dim=1)).squeeze(1)
