import gc
import os
import time
from typing import Dict

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

import wandb
from rdbdiff.dataloader import get_neighbor_dataloader
from rdbdiff.model import GRDM


class SamplingBudgetExhausted(Exception):
    """Raised when the wall-clock budget runs out mid-sampling. State is checkpointed
    before this is raised, so the relauncher just has to start the process again."""

    def __init__(self, t: int):
        super().__init__(f"wall-clock budget exhausted during sampling at timestep {t}")
        self.t = t


class GRDMSampler:
    def __init__(
        self,
        diffusion_model: GRDM,
        save_dir: str,
        num_timesteps: int,
        num_neighbors: list,
        sample_batch_size: int,
        device=torch.device("cuda"),
        dimension_tables: list = None,
        node_type_to_d_in: Dict[NodeType, int] = None,
        checkpoint_every: int = 25,
        deadline: float = None,
    ):
        diffusion_model._denoise_fn.load_state_dict(
            torch.load(
                os.path.join(save_dir, "model_ema.pt"),
                weights_only=True,
            )
        )
        diffusion_model.to(device)
        diffusion_model.eval()

        self.diffusion_model = diffusion_model
        self.num_timesteps = num_timesteps
        self.num_neighbors = num_neighbors
        self.sample_batch_size = sample_batch_size
        self.device = device
        self.dimension_tables = dimension_tables
        self.node_type_to_d_in = node_type_to_d_in
        self.checkpoint_every = checkpoint_every
        # absolute time.time() after which we stop and exit cleanly, or None for no limit
        self.deadline = deadline
        self.checkpoint_path = os.path.join(save_dir, "sampling_checkpoint.pt")

    @torch.no_grad()
    def sample_node_attributes(
        self,
        graph: HeteroData,
    ) -> HeteroData:
        """
        Generate synthetic rows for each table in the database.
        This method keeps the edges of the input data and generates new nodes.
        """
        sampling_start_time = time.time()
        self.diffusion_model.eval()

        # The whole state of this loop is (t, x per node type), so it can be checkpointed
        # and resumed -- 2000 timesteps do not fit in one process lifetime on a contended GPU.
        start_t = self.load_sampling_checkpoint(graph)
        if start_t is None:
            # base noise: z_T ~ N(0, I) for numerical features; uniform log-one-hot for the
            # categorical tail when the model uses multinomial categorical diffusion
            for node_type in graph.node_types:
                if (
                    self.dimension_tables is not None
                    and node_type in self.dimension_tables
                ):
                    continue
                graph[node_type].x = self.diffusion_model.relational_init_noise(
                    node_type, graph[node_type].x
                )
            start_t = self.num_timesteps - 1
        else:
            print(f"Resuming sampling from timestep {start_t}...", flush=True)

        # reverse diffusion process: at each timestep, we take one denoising step on all nodes of all types
        # z_t = denoise(z_{t+1}, t)
        for t in reversed(range(0, start_t + 1)):
            step_start_time = time.time()
            wandb.log({f"sampling/sample_timestep": t})

            # get dataloader
            dataloader = self.get_dataloader(graph)

            # run 1 step of denoising on the whole database and collect them in denoised_attributes
            denoised_attributes = {node_type: [] for node_type in graph.node_types}
            # process data in batches
            for batch in dataloader:
                batch = batch.to(self.device)

                x_in = batch.x_dict
                t_in = {
                    node_type: torch.full(
                        (x_in[node_type].size(0),),
                        t,
                        device=self.device,
                        dtype=torch.long,
                    )
                    for node_type in batch.node_types
                }
                if self.dimension_tables is not None:
                    for node_type in self.dimension_tables:
                        t_in[node_type] = torch.zeros_like(t_in[node_type])

                model_out = self.diffusion_model._denoise_fn(x_in, t_in, batch)

                # s = t + 1
                for node_type in model_out:
                    n_input_nodes = (
                        torch.isin(batch.input_id, batch[node_type].n_id).sum().item()
                    )
                    if n_input_nodes > 0:
                        assert model_out[node_type].size(0) == n_input_nodes
                        z_s = x_in[node_type][:n_input_nodes]
                        z_t = self.diffusion_model.relational_reverse_step(
                            node_type,
                            model_out[node_type],
                            z_s,
                            t_in[node_type][:n_input_nodes],
                        )
                        denoised_attributes[node_type].append(
                            z_t.detach().cpu().float()
                        )

            # update data with denoised rows
            for node_type in graph.node_types:
                if (
                    self.dimension_tables is not None
                    and node_type in self.dimension_tables
                ):
                    continue
                denoised_attributes[node_type] = torch.cat(
                    denoised_attributes[node_type], dim=0
                )
                graph[node_type].x = denoised_attributes[node_type]

            # TODO: check if this is needed
            # free up GPU memory
            torch.cuda.empty_cache()
            # free up CPU memory
            gc.collect()

            step_end_time = time.time()
            step_total_time = step_end_time - step_start_time
            wandb.log({f"sampling/step_time": step_total_time})
            print(
                f"Sampling timestep {t}/{self.num_timesteps} took {step_total_time:.2f}s",
                flush=True,
            )

            # Checkpoint on a fixed grid of t so the work lost to a kill is bounded by
            # checkpoint_every timesteps regardless of where the process died.
            if t % self.checkpoint_every == 0:
                self.save_sampling_checkpoint(graph, t)

            if self.deadline is not None and time.time() >= self.deadline:
                # Save at whatever t we reached, then hand back to the relauncher. The
                # checkpoint is the resume point, so no sampling work is repeated.
                self.save_sampling_checkpoint(graph, t)
                raise SamplingBudgetExhausted(t)

        sampling_end_time = time.time()
        total_sampling_time = sampling_end_time - sampling_start_time
        wandb.log({f"sampling/total_sampling_time": total_sampling_time})
        # sampling finished; the checkpoint is now dead weight (it holds a full copy of x)
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        return graph

    def save_sampling_checkpoint(self, graph: HeteroData, t: int):
        """Persist the reverse-diffusion state: the next timestep to run and every node
        type's current x. Written via .tmp + os.replace so a kill mid-write cannot leave a
        truncated checkpoint behind."""
        tmp_path = self.checkpoint_path + ".tmp"
        torch.save(
            {
                "t": t - 1,  # t itself is done; resume at the next one down
                "node_type_to_x": {
                    node_type: graph[node_type].x.detach().cpu()
                    for node_type in graph.node_types
                },
            },
            tmp_path,
        )
        os.replace(tmp_path, self.checkpoint_path)

    def load_sampling_checkpoint(self, graph: HeteroData):
        """Restore x for every node type and return the timestep to resume at, or None if
        there is no checkpoint (in which case the caller seeds fresh noise)."""
        if not os.path.exists(self.checkpoint_path):
            return None
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        for node_type, x in checkpoint["node_type_to_x"].items():
            graph[node_type].x = x
        t = checkpoint["t"]
        if t < 0:
            return None
        return t

    def get_dataloader(self, graph: HeteroData):
        dataloader, _ = get_neighbor_dataloader(
            data=graph,
            batch_size=self.sample_batch_size,
            shuffle=False,
            disjoint=False,
            num_neighbors=self.num_neighbors,
            dimension_tables=self.dimension_tables,
            node_type_to_d_in=self.node_type_to_d_in,
        )
        return dataloader
