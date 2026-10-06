from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from rdbdiff.gnn import build_gnn

from rdbdiff.third_party.tab_ddpm import (
    MLP,
    GaussianMultinomialDiffusion,
    timestep_embedding,
)


# The Graph-Conditional Relational Diffusion Model (GRDM)
class GRDM(GaussianMultinomialDiffusion):
    def __init__(
        self,
        denoise_fn,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        gaussian_parametrization="eps",
        multinomial_loss_type="vb_stochastic",
        parametrization="x0",
        scheduler="cosine",
        device=torch.device("cpu"),
        dimension_tables: list = None,
        self_cond_aux_weight: float = 1.0,
        categorical_diffusion: str = "gaussian",
        node_type_to_num_classes: Dict[NodeType, list] = None,
        node_type_to_n_num: Dict[NodeType, int] = None,
    ):
        super(GRDM, self).__init__(
            num_classes=np.array([0]),
            num_numerical_features=0,
            denoise_fn=denoise_fn,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            gaussian_parametrization=gaussian_parametrization,
            multinomial_loss_type=multinomial_loss_type,
            parametrization=parametrization,
            scheduler=scheduler,
            device=device,
        )
        self.device = device
        self.dimension_tables = dimension_tables
        # weight on the x0-reconstruction loss that supervises the first of the denoiser's
        # two passes (fix S11). Unused unless the denoiser was built with self-conditioning.
        self.self_cond_aux_weight = self_cond_aux_weight

        # Discrete / multinomial categorical diffusion. Under "gaussian" (default) nothing
        # below fires and the model is bit-identical to the published one. Under "multinomial"
        # the categorical block of each node type -- laid out as one-hot indicator columns,
        # exactly the S4 layout -- is denoised in log space with the parent class's tested
        # multinomial machinery (`q_sample` / `_multinomial_loss` / `p_sample`), never as a
        # Gaussian over the simplex. The parent hard-assumes a single flat `num_classes`; we
        # keep a per-node-type spec and swap it in around each call via `_cat_ctx`.
        self.categorical_diffusion = categorical_diffusion
        self._cat: Dict[NodeType, dict] = {}
        if categorical_diffusion == "multinomial":
            assert node_type_to_num_classes is not None and node_type_to_n_num is not None
            for nt, ks in node_type_to_num_classes.items():
                self._cat[nt] = self._build_cat_spec(
                    np.asarray(list(ks), dtype=np.int64), int(node_type_to_n_num[nt])
                )

    def _build_cat_spec(self, ks: np.ndarray, n_num: int) -> dict:
        """Per-node-type analogue of the parent's flat `num_classes` bookkeeping."""
        if len(ks):
            expanded = torch.from_numpy(
                np.concatenate([np.full(int(k), int(k), dtype=np.int64) for k in ks])
            ).to(self.device)
            cum = np.cumsum(ks)
            slices = [np.arange(ks[0])] + [
                np.arange(cum[i - 1], cum[i]) for i in range(1, len(cum))
            ]
            offsets = torch.from_numpy(np.append([0], cum)).to(self.device)
        else:
            expanded = torch.zeros(0, dtype=torch.long, device=self.device)
            slices = [np.arange(0)]
            offsets = torch.zeros(2, dtype=torch.long, device=self.device)
        return dict(
            num_classes=ks,
            num_classes_expanded=expanded,
            slices_for_classes=slices,
            offsets=offsets,
            n_num=int(n_num),
            K=int(ks.sum()),
        )

    @contextmanager
    def _cat_ctx(self, node_type: NodeType):
        """Temporarily point the parent's flat multinomial buffers at one node type's spec."""
        spec = self._cat[node_type]
        saved = (
            self.num_classes,
            self.num_classes_expanded,
            self.slices_for_classes,
            self.offsets,
        )
        self.num_classes = spec["num_classes"]
        self.num_classes_expanded = spec["num_classes_expanded"]
        self.slices_for_classes = spec["slices_for_classes"]
        self.offsets = spec["offsets"]
        try:
            yield spec
        finally:
            (
                self.num_classes,
                self.num_classes_expanded,
                self.slices_for_classes,
                self.offsets,
            ) = saved

    def _has_cat(self, node_type: NodeType) -> bool:
        return (
            self.categorical_diffusion == "multinomial"
            and node_type in self._cat
            and self._cat[node_type]["K"] > 0
        )

    def relational_gaussian_loss(
        self,
        batch: HeteroData,
    ):
        batch_size = (
            batch.batch_size
        )  # number of sampled seed nodes, forming disconnected subgraphs (in case of disjoint=True)

        # sample time for batch_size target nodes/subgraphs. pt is the per-sample probability
        # of drawing this t (uniform: 1/T); the multinomial loss needs it to reweight the KL.
        t, pt = self.sample_time(batch_size, self.device, "uniform")

        # clean features
        x_dict = batch.x_dict  # {node_type: Tensor(num_nodes, d_in)}

        # noisy features
        x_in = {}  # {node_type: Tensor(num_nodes, d_in)}
        # noise levels
        t_in = {}  # {node_type: Tensor(num_nodes,)}
        # target features (mixture of features at noise level 0 and t-1), only needed for the batch_size target nodes
        x_target = (
            {}
        )  # {node_type: {"x_num": Tensor(num_target_nodes_node_type, d_in), "x_num_t": ..., "noise": ...}}
        for node_type in x_dict:
            # subgraph_ids[i] is the id of the target node that defines the subgraph
            # we use it to make sure that all nodes in a subgraph are noised at the same noise level
            subgraph_ids = batch[node_type].batch

            t_broadcasted = t[
                subgraph_ids
            ]  # if 2 nodes are in the same subgraph, they will have the same t
            pt_broadcasted = pt[subgraph_ids]

            if self.dimension_tables is not None and node_type in self.dimension_tables:
                # dimension_tables are not diffused, so we keep them as-is
                x_in[node_type] = x_dict[node_type]
                t_in[node_type] = torch.zeros_like(t_broadcasted)
                # nodes of this type are never target nodes
                continue

            if self._has_cat(node_type):
                # multinomial path: the tail K columns are one-hot; noise the numerical head
                # with Gaussian diffusion and the categorical tail in log space.
                n_num = self._cat[node_type]["n_num"]
                full = x_dict[node_type]
                x_num = full[:, :n_num]
                x_cat_oh = full[:, n_num:]
                if n_num > 0:
                    noise = torch.randn_like(x_num)
                    x_num_t = self.gaussian_q_sample(x_num, t_broadcasted, noise=noise)
                else:
                    noise = torch.zeros_like(x_num)
                    x_num_t = x_num
                with self._cat_ctx(node_type):
                    log_x_cat = torch.log(x_cat_oh.float().clamp(min=1e-30))
                    log_x_cat_t = self.q_sample(
                        log_x_start=log_x_cat, t=t_broadcasted
                    )
                x_in[node_type] = torch.cat([x_num_t, log_x_cat_t], dim=1)
            else:
                # we view all features as numerical and apply Gaussian diffusion
                x_num = x_dict[node_type]
                noise = torch.randn_like(x_num)
                x_num_t = self.gaussian_q_sample(x_num, t_broadcasted, noise=noise)
                x_cat_oh = log_x_cat = log_x_cat_t = None
                x_in[node_type] = x_num_t
            t_in[node_type] = t_broadcasted

            # collect features needed for loss computation
            # we only need the features of the input nodes, which are the first nodes of each node type
            # n_id is the global node index for every sampled node in the graph
            # input_id is the global index of the input_nodes
            # we compute the number of nodes in input_id that are also in n_id of this specific type:
            n_input_nodes = (
                torch.isin(batch.input_id, batch[node_type].n_id).sum().item()
            )

            if (
                n_input_nodes > 0
            ):  # not necessarily all node types have been sampled in this batch as input nodes
                # check that the first n_input_nodes nodes are actually input nodes:
                assert torch.isin(
                    batch[node_type].n_id[:n_input_nodes], batch.input_id
                ).all()
                x_target_node_type = {}
                x_target_node_type["t"] = t_broadcasted[:n_input_nodes]
                x_target_node_type["pt"] = pt_broadcasted[:n_input_nodes]
                x_target_node_type["x_num"] = x_num[:n_input_nodes]
                x_target_node_type["x_num_t"] = x_num_t[:n_input_nodes]
                x_target_node_type["noise"] = noise[:n_input_nodes]
                x_target_node_type["has_cat"] = log_x_cat is not None
                if log_x_cat is not None:
                    x_target_node_type["log_x_cat"] = log_x_cat[:n_input_nodes]
                    x_target_node_type["log_x_cat_t"] = log_x_cat_t[:n_input_nodes]

                x_target[node_type] = x_target_node_type

        # make sure that the total number of input nodes matches the batch size
        assert (
            np.sum([x_target[node_type]["x_num"].size(0) for node_type in x_target])
            == batch_size
        )

        # run denoising model
        model_out = self._denoise_fn(
            x_in=x_in,
            t_in=t_in,
            batch=batch,
        )  # {node_type: Tensor(num_target_nodes_node_type, d_in_node_type)}

        # compute loss
        node_type_to_loss = {}
        for node_type in model_out:
            model_out_node_type = model_out[node_type]
            x_target_node_type = x_target[node_type]

            if x_target_node_type.get("has_cat"):
                # split the denoiser output: numerical head -> eps (MSE), categorical tail
                # -> per-class logits (multinomial VB loss). Both terms are per-seed-node
                # vectors, so node_type_to_loss keeps its "(num_seed_nodes,)" shape and the
                # trainer's per-table counts/averages still mean what they meant.
                spec = self._cat[node_type]
                n_num = spec["n_num"]
                model_out_num = model_out_node_type[:, :n_num]
                model_out_cat = model_out_node_type[:, n_num:]
                loss_gauss = (
                    self._gaussian_loss(
                        model_out_num,
                        x_target_node_type["x_num"],
                        x_target_node_type["x_num_t"],
                        x_target_node_type["t"],
                        x_target_node_type["noise"],
                    )
                    if n_num > 0
                    else 0.0
                )
                with self._cat_ctx(node_type):
                    loss_multi = self._multinomial_loss(
                        model_out_cat,
                        x_target_node_type["log_x_cat"],
                        x_target_node_type["log_x_cat_t"],
                        x_target_node_type["t"],
                        x_target_node_type["pt"],
                        {},
                    ) / max(len(spec["num_classes"]), 1)
                node_type_to_loss[node_type] = loss_gauss + loss_multi
            else:
                loss_gauss = self._gaussian_loss(
                    model_out_node_type,
                    x_target_node_type["x_num"],
                    x_target_node_type["x_num_t"],
                    x_target_node_type["t"],
                    x_target_node_type["noise"],
                )  # shape: (num_target_nodes_node_type,)

                node_type_to_loss[node_type] = loss_gauss

        # NOTE: using sum for now to avoid introducing a scaling constant that would depend
        # on the number of sampled node types in this specific batch.
        loss = torch.stack(
            [loss_node_type.mean() for loss_node_type in node_type_to_loss.values()]
        ).sum()

        # Self-conditioning (fix S11): the denoiser's first pass emits an x0 estimate for
        # every node, which the second pass consumes in place of the raw noisy neighbour
        # features. That estimate is fed forward detached, so this term is the only thing
        # training it -- without it the head would keep its initialisation forever.
        # It is kept out of node_type_to_loss so the trainer's per-table averages still
        # mean what they meant before.
        x0_hat_seed = getattr(self._denoise_fn, "last_x0_hat_seed", None)
        if x0_hat_seed and self.self_cond_aux_weight > 0:
            aux = torch.stack(
                [
                    F.mse_loss(
                        # under multinomial diffusion x0_hat spans num+cat while x_num is
                        # only the numerical head; the slice is a no-op on the Gaussian path
                        x0_hat_seed[node_type][
                            :, : x_target[node_type]["x_num"].shape[1]
                        ],
                        x_target[node_type]["x_num"],
                    )
                    for node_type in model_out
                    if node_type in x0_hat_seed
                ]
            ).sum()
            loss = loss + self.self_cond_aux_weight * aux

        return loss, node_type_to_loss

    @torch.no_grad()
    def relational_init_noise(
        self, node_type: NodeType, x_like: torch.Tensor
    ) -> torch.Tensor:
        """Base noise z_T for one node type's rows. Gaussian everywhere on the default path;
        under multinomial diffusion the categorical tail starts as a uniform log-one-hot draw
        rather than white noise."""
        if not self._has_cat(node_type):
            return torch.randn_like(x_like)
        n_num = self._cat[node_type]["n_num"]
        out = torch.randn_like(x_like)
        with self._cat_ctx(node_type):
            uniform_logits = torch.zeros(
                x_like.size(0), self._cat[node_type]["K"], device=x_like.device
            )
            out[:, n_num:] = self.log_sample_categorical(uniform_logits)
        return out

    @torch.no_grad()
    def relational_reverse_step(
        self,
        node_type: NodeType,
        model_out_nt: torch.Tensor,
        z_s_nt: torch.Tensor,
        t_nt: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse-diffusion step for one node type's seed rows -> z_{t-1}. Gaussian
        p_sample on the default path; split num (Gaussian) / cat (multinomial p_sample) under
        `--categorical_diffusion multinomial`."""
        if not self._has_cat(node_type):
            return self.gaussian_p_sample(
                model_out_nt, z_s_nt, t_nt, clip_denoised=False
            )["sample"]
        n_num = self._cat[node_type]["n_num"]
        out = torch.empty_like(z_s_nt)
        if n_num > 0:
            out[:, :n_num] = self.gaussian_p_sample(
                model_out_nt[:, :n_num], z_s_nt[:, :n_num], t_nt, clip_denoised=False
            )["sample"]
        with self._cat_ctx(node_type):
            out[:, n_num:] = self.p_sample(
                model_out_nt[:, n_num:], z_s_nt[:, n_num:], t_nt, {}
            )
        return out


class GNNDenoiser(nn.Module):
    def __init__(
        self,
        node_type_to_d_in: Dict[NodeType, int],
        gnn_params: Dict,
        rtdl_params: Dict,
        dim_t: int = 128,
        dimension_tables: list = None,
        node_type_to_rtdl_d_layers: Dict[NodeType, list] = None,
        gnn_residual: bool = False,
        degree_dim: int = 0,
        self_conditioning: bool = False,
    ):
        # TODO: consider ID awareness (see paper and relbench tutorial train_model.ipynb)
        super().__init__()
        self.dim_t = dim_t
        self.dimension_tables = dimension_tables or []
        self.self_conditioning = self_conditioning
        # populated by forward() when self-conditioning is on; the diffusion loss reads it
        self.last_x0_hat_seed = None

        # same embedding module is applied to the time step of every node in the graph
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

        # a different projection layer is applied to the nodes from each type
        self.projs = nn.ModuleDict()
        for node_type, d_in in node_type_to_d_in.items():
            self.projs[node_type] = nn.Linear(d_in, dim_t)

        # Exact per-relation cardinality (fix S10). Neighbour sampling caps how many children
        # a node's representation is built from (`--num_neighbors -1 16` sees 16 of an
        # account's ~235 transactions), so the aggregate the GNN forms is a small sample of
        # the real one and carries no information about how many children there actually are
        # -- which is itself a feature downstream models use. The structure is sampled before
        # any attribute is generated, so the true degrees are known at generation time too.
        self.deg_proj = nn.Linear(degree_dim, dim_t) if degree_dim > 0 else None

        # a single GNN is used to process the whole graph
        # specifically, during training, the GNN is used to get node representation of the input nodes.
        self.gnn = build_gnn(
            channels=dim_t,
            residual=gnn_residual,
            **gnn_params,
        )

        # Self-conditioning (fix S11). While a row is denoised, its neighbours are at the same
        # noise level, so at large t the conditioning window is effectively empty and the
        # measured window sits one hop behind the nominal depth (docs/downstream/DIAGNOSIS.md A2). Running the
        # GNN twice and letting the second pass see the first pass's x0 estimate of each
        # neighbour restores that hop without paying for another one.
        self.x0_heads = nn.ModuleDict()
        self.sc_projs = nn.ModuleDict()
        if self_conditioning:
            for node_type, d_in in node_type_to_d_in.items():
                # the second pass consumes an x0 estimate for every node type ...
                proj = nn.Linear(d_in, dim_t)
                # ... injected through a zero-initialised map, so an untrained model is
                # exactly the one-pass model and the estimate is only used once it is worth
                # something
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                self.sc_projs[node_type] = proj
                # ... but only diffused types need one predicted; dimension tables are never
                # noised, so their clean values are already available
                if node_type not in self.dimension_tables:
                    self.x0_heads[node_type] = nn.Linear(dim_t, d_in)

        # a different MLP is applied to the representations of the input nodes obtained by the GNN
        self.mlps = nn.ModuleDict()
        for node_type, d_in in node_type_to_d_in.items():
            if dimension_tables is not None and node_type in dimension_tables:
                continue
            if node_type_to_rtdl_d_layers is not None:
                rtdl_params["d_layers"] = node_type_to_rtdl_d_layers[node_type]
            rtdl_params["d_in"] = dim_t
            rtdl_params["d_out"] = d_in
            self.mlps[node_type] = MLP.make_baseline(**rtdl_params)

        # reset parameters
        self.gnn.reset_parameters()

    def _node_embeddings(
        self,
        x_in: Dict[NodeType, torch.Tensor],
        t_emb: Dict[NodeType, torch.Tensor],
        batch: HeteroData,
        x0_hat: Dict[NodeType, torch.Tensor] = None,
    ) -> Dict[NodeType, torch.Tensor]:
        """Project input features and add the time embedding, plus whatever conditioning
        signals are enabled."""
        h = {}
        for node_type, x in x_in.items():
            e = self.projs[node_type](x) + t_emb[node_type]
            if self.deg_proj is not None:
                e = e + self.deg_proj(batch[node_type].deg.to(e.dtype))
            if x0_hat is not None:
                e = e + self.sc_projs[node_type](x0_hat[node_type])
            h[node_type] = e
        return h

    @staticmethod
    def _n_input_nodes(batch: HeteroData, node_type: NodeType) -> int:
        # we only need the features of the input nodes, which are the first nodes of each node type
        # n_id is the global node index for every sampled node in the graph
        # input_id is the global index of the input_nodes
        # we compute the number of nodes in input_id that are also in n_id of this specific type:
        n_input_nodes = torch.isin(batch.input_id, batch[node_type].n_id).sum().item()
        if n_input_nodes > 0:
            # check that the first n_input_nodes nodes are actually input nodes:
            assert torch.isin(
                batch[node_type].n_id[:n_input_nodes], batch.input_id
            ).all()
        return n_input_nodes

    def forward(
        self,
        x_in: Dict[NodeType, torch.Tensor],  # shape: (num_nodes, d_in)
        t_in: Dict[NodeType, torch.Tensor],  # shape: (num_nodes,)
        batch: HeteroData,
    ):
        # embed time
        t_emb = {
            node_type: self.time_embed(
                timestep_embedding(t, self.dim_t)
            )  # shape: (num_nodes, dim_t)
            for node_type, t in t_in.items()
        }

        # project input features and add time embedding
        h = self._node_embeddings(x_in, t_emb, batch)

        self.last_x0_hat_seed = None
        if self.self_conditioning:
            # first pass: one round of message passing, read out as an x0 estimate for every
            # node in the batch (a single linear map, so this costs a GNN forward and not a
            # second set of MLP heads)
            z_first = self.gnn(h, batch.edge_index_dict)
            x0_hat = {}
            for node_type, x in x_in.items():
                if node_type in self.x0_heads and node_type in z_first:
                    x0_hat[node_type] = self.x0_heads[node_type](z_first[node_type])
                else:
                    x0_hat[node_type] = x
            # keep the seed-node slice for the auxiliary loss that supervises this head
            self.last_x0_hat_seed = {
                node_type: x0_hat[node_type][: self._n_input_nodes(batch, node_type)]
                for node_type in self.x0_heads
                if node_type in x0_hat and self._n_input_nodes(batch, node_type) > 0
            }
            # second pass sees the estimate rather than the raw noise. Detached, so the first
            # pass is trained by its own reconstruction loss instead of by whatever happens
            # to make the denoiser's job easier.
            h = self._node_embeddings(
                x_in,
                t_emb,
                batch,
                x0_hat={k: v.detach() for k, v in x0_hat.items()},
            )

        # run GNN
        x_out = self.gnn(
            h,
            batch.edge_index_dict,
        )  # {node_type: Tensor(num_nodes, dim_t)}

        # run MLPs
        x_final = {}
        for node_type in x_out:
            if node_type not in self.mlps:
                continue
            # extract features of input nodes
            n_input_nodes = self._n_input_nodes(batch, node_type)
            if n_input_nodes > 0:
                input_nodes_gnn_representations = x_out[node_type][:n_input_nodes]
                x_final[node_type] = self.mlps[node_type](
                    input_nodes_gnn_representations
                )

        # make sure that the total number of input nodes matches the batch size
        assert (
            np.sum([x_final[node_type].size(0) for node_type in x_final])
            == batch.batch_size
        )
        return x_final
