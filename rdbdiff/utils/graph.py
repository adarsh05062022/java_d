"""
Code adapted from https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/graph.py
and https://github.com/pyg-team/pytorch-frame/blob/master/torch_frame/data/mapper.py
"""

from typing import Dict

import numpy as np
import pandas as pd
import torch
from relbench.base import Database
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index


def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, str]],
    col_to_num_classes: Dict[str, Dict[str, int]] = None,
    add_degree_features: bool = False,
) -> HeteroData:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": "numerical"}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        dtype = _get_default_numpy_dtype()
        # col_to_stype only contains columns that are not pkey or fkey at this point
        rows_np = df[col_to_stype.keys()].to_numpy().astype(dtype)

        num_numerical_features = len(
            [stype for stype in col_to_stype.values() if stype == "numerical"]
        )
        num_categorical_features = len(
            [stype for stype in col_to_stype.values() if stype == "categorical"]
        )

        # Fix S4: expand each categorical column into one indicator column per class, so the
        # model no longer sees class codes as points on a line. Ordinal codes give the diffusion
        # a false metric in which a rare class sitting between two dense ones (Berka
        # `loan.status` B, between A and C) is absorbed by its neighbours on rounding.
        # `sort_columns` guarantees numerical columns precede categorical ones, so the
        # categorical block is exactly the tail of `rows_np`.
        if col_to_num_classes is not None:
            cat_cols = [c for c, s in col_to_stype.items() if s == "categorical"]
            blocks = [rows_np[:, :num_numerical_features]]
            for i, col in enumerate(cat_cols):
                n_classes = col_to_num_classes[table_name][col]
                codes = rows_np[:, num_numerical_features + i].astype(np.int64)
                onehot = np.zeros((len(codes), n_classes), dtype=dtype)
                onehot[np.arange(len(codes)), np.clip(codes, 0, n_classes - 1)] = 1.0
                blocks.append(onehot)
            rows_np = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
            num_categorical_features = rows_np.shape[1] - num_numerical_features

        data[table_name].x = torch.from_numpy(rows_np)

        # Add number of numerical and categorical features
        data[table_name].num_numerical_features = torch.full(
            (data[table_name].num_nodes,), num_numerical_features, dtype=torch.long
        )
        data[table_name].num_categorical_features = torch.full(
            (data[table_name].num_nodes,), num_categorical_features, dtype=torch.long
        )

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    if add_degree_features:
        add_relation_degrees(data)

    data.validate()

    return data


def add_relation_degrees(data: HeteroData) -> None:
    r"""Attach each node's exact per-relation child count as a ``deg`` feature (fix S10).

    A downstream model's strongest features over a child table are counts and count-derived
    ratios -- how many transactions an account has, what share of them carry a given symbol.
    The denoiser cannot recover those from message passing: ``--num_neighbors -1 16`` hands it
    16 of an account's ~235 transactions, so every aggregate it forms is a fixed-size sample
    whose size says nothing about the real one. The graph structure is sampled in full before
    any attribute is generated, so the true degree is available at generation time exactly as
    it is at training time.

    Degrees are stored as one dense ``(num_nodes, num_edge_types)`` block per node type, with
    zeros in the columns of relations that do not point at that type. The uniform width is
    what lets the block survive the ``to_homogeneous`` / ``to_heterogeneous`` round trip the
    dataloader performs. Counts are ``log1p``-compressed because table fan-outs here span four
    orders of magnitude.
    """
    # sorted, not insertion-ordered, so the real graph and the sampled structure graph agree
    # on which column means which relation
    edge_types = sorted(data.edge_types)
    for node_type in data.node_types:
        data[node_type].deg = torch.zeros(
            (data[node_type].num_nodes, len(edge_types)),
            dtype=torch.get_default_dtype(),
        )
    for j, edge_type in enumerate(edge_types):
        dst = edge_type[-1]
        dst_index = data[edge_type].edge_index[1]
        counts = torch.bincount(dst_index, minlength=data[dst].num_nodes)
        data[dst].deg[:, j] = torch.log1p(counts.to(torch.get_default_dtype()))


# Taken from https://github.com/pyg-team/pytorch-frame/blob/master/torch_frame/data/mapper.py
def _get_default_numpy_dtype() -> np.dtype:
    r"""Returns the default numpy dtype."""
    # NOTE: We are converting the default PyTorch dtype into a string
    # representation that can be understood by numpy.
    # TODO: Think of a "less hacky" way to do this.
    dtype = str(torch.get_default_dtype()).split(".")[-1]
    return np.dtype(dtype)
