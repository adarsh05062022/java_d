import math
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import wandb
from rdbdiff.model import GRDM
from rdbdiff.third_party.tab_ddpm import update_ema


class TrainingBudgetExhausted(Exception):
    """Raised at an epoch boundary when the wall-clock budget runs out. The checkpoint is
    already written, so the relauncher just has to start the process again."""

    def __init__(self, step: int):
        super().__init__(f"wall-clock budget exhausted during training at step {step}")
        self.step = step


class GRDMTrainer:
    def __init__(
        self,
        diffusion_model: GRDM,
        train_dataloader,
        lr,
        weight_decay,
        steps,
        save_dir: str,
        device=torch.device("cuda"),
        train_batch_size: int = 4096,
        n_total_target_nodes: int = None,
        max_epochs: int = None,
        target_node_types: list[str] = None,
        seed_balance_alpha: float = 0.0,
        deadline: float = None,
    ):
        self.diffusion_model = diffusion_model
        # Exponential Moving Average (EMA) model is a copy of the denoising model
        # that is not trained but updated as an EMA of the trained model.
        self.ema_model = deepcopy(self.diffusion_model._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        # data iterator that yields batches indefinitely from the datalaoder
        self.train_iter = get_iter_from_dataloader(train_dataloader)

        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # number of steps per epoch is determined by the dataloader size
        self.steps_per_epoch = len(train_dataloader)
        # it should hold that steps_per_epoch * train_batch_size >= n_total_target_nodes
        assert self.steps_per_epoch == math.ceil(
            n_total_target_nodes / train_batch_size
        )
        if max_epochs is not None:
            # limit the total number of steps to steps_per_epoch * max_epochs
            self.steps = min(steps, self.steps_per_epoch * max_epochs)
        else:
            self.steps = steps

        self.save_dir = save_dir
        self.device = device
        self.train_batch_size = train_batch_size
        self.n_total_target_nodes = n_total_target_nodes
        self.target_node_types = target_node_types
        self.seed_balance_alpha = seed_balance_alpha
        # absolute time.time() after which we stop at the next epoch boundary and exit
        # cleanly, or None for no limit
        self.deadline = deadline

        self.loss_history = pd.DataFrame(columns=["step", "loss"])

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_training_step(self, batch):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        loss, node_type_to_loss = self.diffusion_model.relational_gaussian_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss, node_type_to_loss

    def run_training_loop(self, start_step: int = 0):
        training_start_time = time.time()
        self.diffusion_model.train()

        step = start_step
        epoch = start_step // self.steps_per_epoch

        node_type_to_curr_count = {node_type: 0 for node_type in self.target_node_types}
        node_type_to_curr_loss = {
            node_type: 0.0 for node_type in self.target_node_types
        }

        best_epoch = 0
        best_loss = float("inf")

        epoch_start_time = time.time()
        while step < self.steps:
            batch = next(self.train_iter)

            batch_loss, batch_node_type_to_loss = self._run_training_step(batch)
            self._anneal_lr(step)

            for node_type, loss_node_type in batch_node_type_to_loss.items():
                node_type_to_curr_count[node_type] += loss_node_type.size(0)
                node_type_to_curr_loss[node_type] += loss_node_type.sum().item()

            wandb.log({f"step": step})
            wandb.log({f"batch_loss": batch_loss.item()})

            if (step + 1) % self.steps_per_epoch == 0:
                # end of epoch
                epoch_end_time = time.time()
                epoch_total_time = epoch_end_time - epoch_start_time

                # With uniform seeding an epoch visits every target node exactly once. Balanced
                # seeding (S3) draws with replacement, so the visit count is a random variable
                # and only the total is fixed; check that instead, and guard the per-type
                # division since a rare type can go unvisited in an epoch.
                seen = sum(node_type_to_curr_count.values())
                if self.seed_balance_alpha > 0.0:
                    assert seen > 0
                else:
                    assert seen == self.n_total_target_nodes
                epoch_loss = 0.0
                for node_type in self.target_node_types:
                    if node_type_to_curr_count[node_type] == 0:
                        continue
                    epoch_loss += (
                        node_type_to_curr_loss[node_type]
                        / node_type_to_curr_count[node_type]
                    )
                epoch_loss = np.around(epoch_loss, 4)

                print(
                    f"Epoch: {epoch}, Step: {(step + 1)}/{self.steps}, Loss: {epoch_loss}"
                )
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    epoch_loss,
                ]
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss

                wandb.log({f"epoch": epoch})
                wandb.log({f"epoch_loss": epoch_loss})
                wandb.log({f"epoch_time": epoch_total_time})
                wandb.log(
                    {f"average_step_time": epoch_total_time / self.steps_per_epoch}
                )
                wandb.log({f"best_epoch": best_epoch})
                wandb.log({f"best_loss": best_loss})

                self.save_checkpoint(step + 1)

                # Stop at an epoch boundary once the wall-clock budget is spent. The
                # checkpoint above is the resume point, so this costs nothing but the
                # process restart; the alternative is being SIGKILLed mid-epoch by the
                # node's ~24h process limit and losing up to one epoch.
                if self.deadline is not None and time.time() >= self.deadline:
                    print(
                        f"Wall-clock budget exhausted at step {step + 1}; "
                        f"checkpointed and exiting for relaunch.",
                        flush=True,
                    )
                    raise TrainingBudgetExhausted(step + 1)

                node_type_to_curr_count = {
                    node_type: 0 for node_type in self.target_node_types
                }
                node_type_to_curr_loss = {
                    node_type: 0.0 for node_type in self.target_node_types
                }
                epoch += 1
                epoch_start_time = time.time()

            update_ema(
                self.ema_model.parameters(),
                self.diffusion_model._denoise_fn.parameters(),
            )
            step += 1

        # end of training
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        wandb.log({f"total_training_time": total_training_time})

    def save_checkpoint(self, step: int):
        """Save a resumable mid-training checkpoint (model, EMA, optimizer, step, loss
        history). Only called at epoch boundaries, so `load_checkpoint`'s returned step
        is always a multiple of `steps_per_epoch` and the per-epoch node-count assert in
        `run_training_loop` stays valid after resuming."""
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")
        tmp_path = checkpoint_path + ".tmp"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.diffusion_model._denoise_fn.state_dict(),
                "ema_state_dict": self.ema_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            tmp_path,
        )
        os.replace(tmp_path, checkpoint_path)  # atomic, avoids a truncated file if killed mid-write

    def has_checkpoint(self):
        return os.path.exists(os.path.join(self.save_dir, "checkpoint.pt"))

    def load_checkpoint(self) -> int:
        checkpoint = torch.load(
            os.path.join(self.save_dir, "checkpoint.pt"), weights_only=False
        )
        self.diffusion_model._denoise_fn.load_state_dict(checkpoint["model_state_dict"])
        self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint["loss_history"]
        return checkpoint["step"]

    def save_model_and_loss_history(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.loss_history.to_csv(os.path.join(self.save_dir, "loss.csv"), index=False)
        torch.save(
            self.diffusion_model._denoise_fn.state_dict(),
            os.path.join(self.save_dir, "model.pt"),
        )
        torch.save(
            self.ema_model.state_dict(), os.path.join(self.save_dir, "model_ema.pt")
        )

    def is_model_already_trained(self):
        return os.path.exists(os.path.join(self.save_dir, "model_ema.pt"))

    def load_model(self):
        self.diffusion_model._denoise_fn.load_state_dict(
            torch.load(os.path.join(self.save_dir, "model.pt"), weights_only=True)
        )
        self.ema_model.load_state_dict(
            torch.load(os.path.join(self.save_dir, "model_ema.pt"), weights_only=True)
        )


def get_iter_from_dataloader(dataloader):
    while True:
        yield from dataloader
