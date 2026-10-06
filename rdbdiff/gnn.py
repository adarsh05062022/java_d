"""GNN backbone for the relational denoiser.

Upstream uses relbench's ``HeteroGraphSAGE`` unchanged: a single neighbourhood statistic
(``gnn_aggr``, ``"sum"`` in every config here) and ``conv -> LayerNorm -> ReLU`` at every
layer including the last. Two properties of that stack are load-bearing for downstream
utility (see ``docs/downstream/DIAGNOSIS.md``):

* **One statistic per neighbourhood.** A downstream model's features over a child table are
  counts, sums, means, extremes and spreads. A denoiser that only ever sees the sum of its
  children's embeddings cannot condition a row on how *dispersed* its children are, so those
  correlations are not in the generated data either.
* **No residual path.** Each layer rewrites the representation from scratch, so depth
  replaces near-row information rather than adding to it. That is consistent with fix S2,
  where ``n_hops=2`` bought the cross-table task 18 points and cost the single-parent task 17.

Both are off by default: ``residual=False`` with a single ``aggr`` string reproduces
relbench's module exactly, and ``model.py`` only reaches for this class when a flag asks for
one of the changes.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType


class RelationalGraphSAGE(nn.Module):
    r"""Heterogeneous GraphSAGE with optional multi-statistic aggregation and residuals.

    Args:
        node_types: node types in the graph.
        edge_types: edge types in the graph.
        channels: hidden width, shared by every node type.
        aggr: neighbourhood aggregation. A string reproduces upstream behaviour; a list
            (e.g. ``["sum", "mean", "max", "std"]``) makes every :class:`SAGEConv` a
            :class:`MultiAggregation` that concatenates the statistics before the linear
            map, so the denoiser sees the shape of a neighbourhood rather than one moment
            of it.
        num_layers: number of message-passing layers, kept equal to ``n_hops``.
        residual: add the incoming representation back after each layer, so a hop refines
            the row's own state instead of overwriting it.
        final_activation: keep the ReLU on the last layer. Upstream does; with residuals the
            MLP heads are better served by an unrectified representation.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: Union[str, List[str]] = "sum",
        num_layers: int = 2,
        residual: bool = False,
        final_activation: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.final_activation = final_activation

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (channels, channels),
                            channels,
                            aggr=list(aggr) if isinstance(aggr, (list, tuple)) else aggr,
                        )
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )
            )
            norm_dict = nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        n_layers = len(self.convs)
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            out_dict = conv(x_dict, edge_index_dict)
            is_last = i == n_layers - 1

            next_dict = {}
            for node_type, h in out_dict.items():
                h = norm_dict[node_type](h)
                if not is_last or self.final_activation:
                    h = h.relu()
                if self.residual:
                    # HeteroConv only emits node types that are the destination of some
                    # sampled edge, and the loader keeps node counts fixed across layers,
                    # so the shapes line up whenever the key is present at all.
                    h = h + x_dict[node_type]
                next_dict[node_type] = h

            # A node type that received no messages this layer keeps its representation
            # rather than dropping out of the dict, which would leave the denoiser with no
            # prediction for its seed nodes.
            for node_type, h in x_dict.items():
                next_dict.setdefault(node_type, h)

            x_dict = next_dict

        return x_dict


def build_gnn(
    channels: int,
    node_types: List[NodeType],
    edge_types: List[EdgeType],
    aggr: Union[str, List[str]],
    num_layers: int,
    residual: bool,
) -> nn.Module:
    """Return the upstream module when nothing is asked for, ours otherwise.

    Keeping relbench's class on the default path means a baseline run is bit-identical to
    the published one rather than merely equivalent-looking.
    """
    if not residual and isinstance(aggr, str):
        from relbench.modeling.nn import HeteroGraphSAGE

        return HeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

    return RelationalGraphSAGE(
        node_types=node_types,
        edge_types=edge_types,
        channels=channels,
        aggr=aggr,
        num_layers=num_layers,
        residual=residual,
        final_activation=not residual,
    )
