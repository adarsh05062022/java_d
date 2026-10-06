"""Sanity check: does a single-table diffusion model, trained on the already-flattened matrix,
recover the child-aggregate signal that GRDM's per-table graph-conditional generation lost?

    conda run --no-capture-output -n tabsyn python -m downstream.flatten_baseline --gpu 0

Task 5 (`aggregate_validation.py`) showed GRDM keeps 11-13% of a cross-table aggregate's
univariate signal against the label and 41-99% of an own-row column's, because the GNN denoiser
sees a child neighbourhood only through one pooled sum. This script removes that bottleneck by
construction: it takes the exact flattened matrix `downstream.tasks` builds for evaluation --
one row per disposition, 90 columns already spanning the account/client/district joins and the
trans/order rollups, label appended as one more column -- and fits a *plain* single-table
diffusion model on it. No graph, no GNN, no pooling: every column lives in the same row, so if
the model learns anything at all about `trans_balance_mean`, it learns it jointly with the label
in the same forward pass.

The denoiser class is `GaussianMultinomialDiffusion` from `rdbdiff/third_party/tab_ddpm` --
the exact same diffusion process GRDM subclasses (`rdbdiff/model.py:GRDM`) and the same one
ClavaDDPM trains per table -- with the vanilla `MLPDiffusion` in place of GRDM's `GNNDenoiser`,
and the same architecture GRDM/ClavaDDPM use elsewhere on Berka (d_layers, num_timesteps=2000,
lr, scheduler). So the only thing that differs from GRDM is exactly the thing under test:
graph-conditional generation vs. flatten-then-generate.

This does not produce a revertible relational database -- there is no fan-out or cross-row
consistency step here, deliberately. It only asks: does removing the aggregation bottleneck,
in isolation, recover the signal? If not, flattening is not the fix. If so, the harder
reconstruction problem (Task 6-adjacent) is worth the engineering.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.aggregate_validation import SELECTION, auc_ci, strength
from downstream.feature_compare import signed_retention
from downstream.data import REPO_ROOT, coerce_discrete, load_db, load_domains, load_meta
from downstream.run import DATASETS
from downstream.tasks import TASKS
from rdbdiff.third_party.tab_ddpm import GaussianMultinomialDiffusion
from rdbdiff.third_party.tab_ddpm.modules import MLPDiffusion

RESULTS = os.path.join(REPO_ROOT, "downstream", "results")
FIG_DIR = os.path.join(REPO_ROOT, "downstream", "figures")

# Same denoiser width and diffusion schedule GRDM/ClavaDDPM use for Berka (configs/*.json,
# ClavaDDPM/configs/berka.json) -- the point is to hold everything but graph-conditioning fixed.
D_LAYERS = [512, 1024, 1024, 1024, 1024, 512]
DIM_T = 128
NUM_TIMESTEPS = 2000
SCHEDULER = "cosine"
LR = 6e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4096
STEPS = 30000
LABEL_COL = "_label"


class FlatTable:
    """Numeric-quantile / categorical-factorize encoding, mirroring the ClavaDDPM/TabDDPM
    preprocessing convention (`pipeline_utils.pipeline_process_data`) closely enough for a
    single-table sanity check, without going through its file-based per-column-domain format.
    """

    def __init__(self, X: pd.DataFrame, cat_spec: dict, y: pd.Series):
        self.num_cols = [c for c in X.columns if c not in cat_spec]
        base_cat_cols = list(cat_spec.keys())

        # Missingness is itself informative here (e.g. a loan with no prior transactions), so a
        # NaN numeric column gets imputed *and* an explicit indicator category rather than being
        # silently zero-filled -- a quiet imputation would let the model treat "no history" as
        # "average history", which is a different failure than the one under test.
        self.nan_indicator_cols = [c for c in self.num_cols if X[c].isna().any()]
        for c in self.nan_indicator_cols:
            X[f"{c}__isnan"] = X[c].isna().astype(int)
        self.cat_cols = base_cat_cols + [f"{c}__isnan" for c in self.nan_indicator_cols] + [LABEL_COL]

        assert not X[base_cat_cols].isna().any().any(), "unexpected NaN in a categorical column"

        medians = X[self.num_cols].median()
        X_num = X[self.num_cols].fillna(medians)
        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(X_num)),
            subsample=10**9,
            random_state=0,
        )
        self.X_num = self.qt.fit_transform(X_num.to_numpy(dtype=np.float64)).astype(np.float32)
        self.medians = medians

        full_cat = X[base_cat_cols].copy()
        for c in self.nan_indicator_cols:
            full_cat[f"{c}__isnan"] = X[f"{c}__isnan"]
        full_cat[LABEL_COL] = np.asarray(y).astype(int)

        self.cat_maps = {c: {v: i for i, v in enumerate(sorted(full_cat[c].unique()))}
                         for c in self.cat_cols}
        self.cat_inv = {c: {i: v for v, i in m.items()} for c, m in self.cat_maps.items()}
        self.X_cat = np.stack(
            [full_cat[c].map(self.cat_maps[c]).to_numpy() for c in self.cat_cols], axis=1
        ).astype(np.float32)
        self.num_classes = np.array([len(self.cat_maps[c]) for c in self.cat_cols])

    @property
    def n_num(self):
        return len(self.num_cols)

    def tensor(self):
        return torch.tensor(np.concatenate([self.X_num, self.X_cat], axis=1), dtype=torch.float32)

    def decode(self, sample: np.ndarray) -> pd.DataFrame:
        """Undo quantile transform and category factorization on a generated `sample` array."""
        x_num = self.qt.inverse_transform(sample[:, : self.n_num].astype(np.float64))
        x_cat = sample[:, self.n_num :]
        out = pd.DataFrame(x_num, columns=self.num_cols)
        for j, c in enumerate(self.cat_cols):
            codes = np.clip(x_cat[:, j].round().astype(int), 0, self.num_classes[j] - 1)
            out[c] = [self.cat_inv[c][v] for v in codes]
        return out


def train(diffusion, x, device, steps=STEPS, batch_size=BATCH_SIZE, log_every=2000):
    diffusion.to(device)
    diffusion.train()
    opt = torch.optim.AdamW(diffusion.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = x.shape[0]
    dummy_y = {"y": torch.zeros(batch_size, dtype=torch.long, device=device)}
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        xb = x[idx].to(device)
        opt.zero_grad()
        loss_multi, loss_gauss = diffusion.mixed_loss(xb, dummy_y)
        loss = loss_multi + loss_gauss
        loss.backward()
        for g in opt.param_groups:
            g["lr"] = LR * (1 - step / steps)
        opt.step()
        if (step + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  step {step + 1:>6d}/{steps}  gloss {loss_gauss.item():.4f}  "
                  f"mloss {loss_multi.item():.4f}  ({el:.0f}s, {(step + 1) / el:.1f} it/s)")
    diffusion.eval()


def run_task(task_name, grdm_seeds, steps, gpu, xgb_seeds=(0, 1, 2)):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    cfg = dict(DATASETS["berka_split"])
    meta = load_meta(cfg["real_train"])
    domains = load_domains(cfg["domains"], list(meta["tables"]))
    real_tables = coerce_discrete(load_db(cfg["real_train"]), domains)
    task = TASKS[task_name]
    X, y, cat_spec = task.build(real_tables, domains)
    print(f"\n{'=' * 78}\n=== {task_name}  ({X.shape[0]} rows x {X.shape[1]} cols, "
          f"{len(cat_spec)} categorical, {y.mean():.1%} positive)\n{'=' * 78}")

    ft = FlatTable(X.copy(), dict(cat_spec), y)
    x = ft.tensor()
    d_in = ft.n_num + int(ft.num_classes.sum())
    print(f"  encoded: {ft.n_num} numeric + {len(ft.cat_cols)} categorical "
          f"(incl. {len(ft.nan_indicator_cols)} missingness flags + label)  d_in={d_in}")

    per_seed_rows = []
    for seed in grdm_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        denoiser = MLPDiffusion(
            d_in=d_in, num_classes=0, is_y_cond=False,
            rtdl_params={"d_layers": D_LAYERS, "dropout": 0.0}, dim_t=DIM_T,
        )
        diffusion = GaussianMultinomialDiffusion(
            num_classes=ft.num_classes, num_numerical_features=ft.n_num, denoise_fn=denoiser,
            num_timesteps=NUM_TIMESTEPS, gaussian_loss_type="mse", scheduler=SCHEDULER,
            device=device,
        )
        print(f"\n[seed {seed}] training {steps} steps on {device}...")
        train(diffusion, x, device, steps=steps)

        with torch.no_grad():
            sample, _ = diffusion.sample_all(
                num_samples=len(X), batch_size=min(BATCH_SIZE, len(X)),
                y_dist=torch.ones(1, device=device),
            )
        synth = ft.decode(sample.numpy())
        synth_y = synth.pop(LABEL_COL).astype(int)
        drop_cols = [f"{c}__isnan" for c in ft.nan_indicator_cols if f"{c}__isnan" in synth.columns]
        synth = synth.drop(columns=drop_cols)
        print(f"  sampled {len(synth)} synthetic rows, {synth_y.mean():.1%} positive "
              f"(real {y.mean():.1%})")

        for family, feats in (("aggregate", SELECTION[task_name]["aggregate"]),
                              ("control", SELECTION[task_name]["control"])):
            for feat in feats:
                if feat not in X.columns:
                    continue
                a, lo, hi, n = auc_ci(synth[feat], synth_y, seed=seed)
                per_seed_rows.append(dict(task=task_name, feature=feat, family=family,
                                          grdm_seed=seed, synth_auc=a, synth_lo=lo, synth_hi=hi))
        del diffusion, denoiser

    # real-data reference, identical measurement to Task 5
    real_rows = {}
    for family, feats in (("aggregate", SELECTION[task_name]["aggregate"]),
                          ("control", SELECTION[task_name]["control"])):
        for feat in feats:
            if feat not in X.columns:
                continue
            ar, lo, hi, n = auc_ci(X[feat], y)
            real_rows[feat] = dict(real_auc=ar, real_lo=lo, real_hi=hi, family=family)

    summary = []
    for feat, rr in real_rows.items():
        seeds = [r for r in per_seed_rows if r["feature"] == feat]
        aucs = [r["synth_auc"] for r in seeds if r["synth_auc"] == r["synth_auc"]]
        real_signal = not (rr["real_lo"] <= 0.5 <= rr["real_hi"])
        mean_s = float(np.mean(aucs)) if aucs else float("nan")
        sd_s = float(np.std(aucs)) if aucs else float("nan")
        ret = signed_retention(rr["real_auc"], mean_s) if real_signal and aucs else float("nan")
        n_cover = sum(1 for r in seeds if r["synth_lo"] == r["synth_lo"] and r["synth_lo"] <= 0.5 <= r["synth_hi"])
        summary.append(dict(
            task=task_name, feature=feat, family=rr["family"], real_signal=real_signal,
            real_auc=rr["real_auc"], real_lo=rr["real_lo"], real_hi=rr["real_hi"],
            synth_auc_mean=mean_s, synth_auc_sd=sd_s, per_seed=[r["synth_auc"] for r in seeds],
            signal_retained=ret, seeds_covering_half=n_cover, n_seeds=len(seeds),
        ))
        tag = "SIG" if real_signal else "---"
        per_seed_str = ", ".join(f"{v:.3f}" for v in summary[-1]["per_seed"])
        print(f"  {rr['family']:9s} {feat:22s} {tag}  real {rr['real_auc']:.3f}  "
              f"flat-synth {mean_s:.3f}±{sd_s:.3f}  [{per_seed_str}]  "
              f"retained {'%.0f%%' % (100*ret) if ret == ret else 'n/a':>5s}  "
              f"CI covers .5 in {n_cover}/{len(seeds)}")
    return summary


def render(rows):
    lines = []
    for task_name in sorted({r["task"] for r in rows}, key=lambda t: t != "berka_card_ownership"):
        trs = [r for r in rows if r["task"] == task_name]
        lines += [f"### {task_name} -- flatten-and-generate baseline", "",
                  "| feature | family | real AUROC | flat-synth AUROC (per seed) | "
                  "retained | synth CI covers 0.5 |",
                  "|---|---|---|---|---|---|"]
        for r in trs:
            fam = "cross-table agg" if r["family"] == "aggregate" else "own/1-hop"
            per_seed = ", ".join(f"{v:.3f}" for v in r["per_seed"])
            ret = f"**{r['signal_retained']:.0%}**" if r["signal_retained"] == r["signal_retained"] else "_n/a_"
            lines.append(
                f"| `{r['feature']}` | {fam} | {r['real_auc']:.3f} | "
                f"{r['synth_auc_mean']:.3f}±{r['synth_auc_sd']:.3f}  ({per_seed}) | {ret} | "
                f"{r['seeds_covering_half']}/{r['n_seeds']} |"
            )
        agg = [r["signal_retained"] for r in trs if r["family"] == "aggregate" and r["real_signal"]]
        ctl = [r["signal_retained"] for r in trs if r["family"] == "control" and r["real_signal"]]
        lines += ["", f"- cross-table aggregates retain **{np.mean(agg):.1%}** "
                      f"({len(agg)} real-signal features) -- vs GRDM's 10.7% (card) / 13.2% (loan)",
                  f"- own-row / 1-hop controls retain **{np.mean(ctl):.1%}** ({len(ctl)} features)", ""]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="*", default=["berka_card_ownership", "berka_loan_default"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    all_rows = []
    for t in args.tasks:
        all_rows += run_task(t, args.seeds, args.steps, args.gpu)

    out = os.path.join(RESULTS, "berka_split__flatten_baseline.json")
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\n" + "\n".join(render(all_rows)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
