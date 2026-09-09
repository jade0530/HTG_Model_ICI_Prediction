"""Sample-level classifier over a cell-gene heterogeneous graph."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, GraphConv, HeteroConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

Pooling = Literal["mean", "attention"]
Readout = Literal["cell", "gene", "both"]
EncoderKind = Literal["placeholder", "sage", "gat"]
GENE_TO_CELL = ("gene", "expressed_by", "cell")
CELL_TO_GENE = ("cell", "expresses", "gene")


class CellAttentionPool(nn.Module):
    """Soft attention over cells in one sample graph.

    A small gate scores each cell; weights are softmax-normalised within the
    graph, then used for a weighted sum. Mean pooling is the uniform special
    case of the same readout.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.gate(x).squeeze(-1)
        weights = softmax(scores, batch)
        pooled = global_add_pool(x * weights.unsqueeze(-1), batch)
        return pooled, weights


class BipartiteHTGEncoder(nn.Module):
    """Typed message passing on cell-gene expression edges and their reverse.

    ``sage`` is weighted GraphSAGE-style mean aggregation (PyG GraphConv).
    ``gat`` is GATv2 with expression as edge attributes. Each layer updates
    both node types, then residual + LayerNorm.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_layers: int = 2,
        conv: Literal["sage", "gat"] = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if conv not in ("sage", "gat"):
            raise ValueError("conv must be 'sage' or 'gat'")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.conv_kind = conv
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.cell_norms = nn.ModuleList()
        self.gene_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_hetero_conv(hidden_dim, conv=conv, heads=heads))
            self.cell_norms.append(nn.LayerNorm(hidden_dim))
            self.gene_norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        edge_weight_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        for conv, cell_norm, gene_norm in zip(self.layers, self.cell_norms, self.gene_norms):
            residual = x_dict
            if self.conv_kind == "sage":
                out = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
            else:
                edge_attr_dict = {key: _edge_attr(weight) for key, weight in edge_weight_dict.items()}
                out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {
                "cell": F.dropout(
                    F.relu(cell_norm(out["cell"] + residual["cell"])),
                    p=self.dropout,
                    training=self.training,
                ),
                "gene": F.dropout(
                    F.relu(gene_norm(out["gene"] + residual["gene"])),
                    p=self.dropout,
                    training=self.training,
                ),
            }
        return x_dict


def _hetero_conv(hidden_dim: int, *, conv: str, heads: int) -> HeteroConv:
    dims = (hidden_dim, hidden_dim)
    if conv == "sage":
        make = lambda: GraphConv(dims, hidden_dim, aggr="mean")
    else:
        make = lambda: GATv2Conv(
            dims,
            hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=1,
            add_self_loops=False,
        )
    return HeteroConv(
        {
            GENE_TO_CELL: make(),
            CELL_TO_GENE: make(),
        },
        aggr="sum",
    )


def _edge_attr(weight: Tensor) -> Tensor:
    return weight.unsqueeze(-1) if weight.dim() == 1 else weight


class SampleGraphClassifier(nn.Module):
    """Cell-gene HTG: typed GNN layers, then pool to one logit.

    ``encoder='sage'`` / ``'gat'`` run bipartite HeteroConv. ``placeholder`` is
    the original one-shot weighted gene-to-cell sum. ``pooling`` is mean or
    attention. ``readout`` chooses cell states, gene states, or both concatenated.
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 64,
        pooling: Pooling = "mean",
        encoder: EncoderKind = "sage",
        num_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.1,
        readout: Readout = "cell",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError("pooling must be 'mean' or 'attention'")
        if encoder not in ("placeholder", "sage", "gat"):
            raise ValueError("encoder must be 'placeholder', 'sage', or 'gat'")
        if readout not in ("cell", "gene", "both"):
            raise ValueError("readout must be 'cell', 'gene', or 'both'")
        self.pooling = pooling
        self.encoder = encoder
        self.readout = readout
        self.gene_emb = nn.Embedding(num_genes, hidden_dim)
        self.cell_in = nn.Linear(1, hidden_dim)
        self.cell_lin = nn.Linear(hidden_dim, hidden_dim) if encoder == "placeholder" else None
        self.htg = (
            None
            if encoder == "placeholder"
            else BipartiteHTGEncoder(
                hidden_dim,
                num_layers=num_layers,
                conv=encoder,
                heads=gat_heads,
                dropout=dropout,
            )
        )
        self.cell_pool = CellAttentionPool(hidden_dim) if pooling == "attention" else None
        self.gene_pool = CellAttentionPool(hidden_dim) if pooling == "attention" else None
        head_in = hidden_dim * (2 if readout == "both" else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph: HeteroData) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gene_x = self.gene_emb(graph["gene"].global_id)
        if self.htg is None:
            cell_x = self._placeholder_cells(graph, gene_x)
        else:
            states = self.htg(
                {"cell": self._cell_start(graph, gene_x), "gene": gene_x},
                {
                    GENE_TO_CELL: graph[GENE_TO_CELL].edge_index,
                    CELL_TO_GENE: graph[CELL_TO_GENE].edge_index,
                },
                {
                    GENE_TO_CELL: graph[GENE_TO_CELL].edge_weight,
                    CELL_TO_GENE: graph[CELL_TO_GENE].edge_weight,
                },
            )
            cell_x, gene_x = states["cell"], states["gene"]
        return (
            cell_x,
            gene_x,
            _node_batch(graph, "cell", cell_x),
            _node_batch(graph, "gene", gene_x),
        )

    def encode_cells(self, graph: HeteroData) -> tuple[Tensor, Tensor]:
        cell_x, _, cell_batch, _ = self.encode(graph)
        return cell_x, cell_batch

    def _cell_start(self, graph: HeteroData, like: Tensor) -> Tensor:
        n_cells = int(graph["cell"].num_nodes)
        features = getattr(graph["cell"], "x", None)
        if features is None:
            features = like.new_ones(n_cells, 1)
        return self.cell_in(features)

    def _placeholder_cells(self, graph: HeteroData, gene_x: Tensor) -> Tensor:
        src, dst = graph[GENE_TO_CELL].edge_index
        weight = graph[GENE_TO_CELL].edge_weight.unsqueeze(-1)
        messages = gene_x[src] * weight
        cell_x = gene_x.new_zeros(int(graph["cell"].num_nodes), gene_x.size(-1))
        cell_x.index_add_(0, dst, messages)
        degree = gene_x.new_zeros(int(graph["cell"].num_nodes), 1)
        degree.index_add_(0, dst, weight)
        lin = self.cell_lin
        if lin is None:
            raise RuntimeError("placeholder encoder requires cell_lin")
        return F.relu(lin(cell_x / degree.clamp(min=1e-6)))

    def pool_cells(self, cell_x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor | None]:
        return _pool_nodes(cell_x, batch, self.cell_pool)

    def pool_genes(self, gene_x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor | None]:
        return _pool_nodes(gene_x, batch, self.gene_pool)

    def forward(self, graph: HeteroData) -> Tensor:
        cell_x, gene_x, cell_batch, gene_batch = self.encode(graph)
        parts: list[Tensor] = []
        if self.readout in ("cell", "both"):
            pooled_cells, _ = self.pool_cells(cell_x, cell_batch)
            parts.append(pooled_cells)
        if self.readout in ("gene", "both"):
            pooled_genes, _ = self.pool_genes(gene_x, gene_batch)
            parts.append(pooled_genes)
        return self.head(torch.cat(parts, dim=-1)).reshape(-1)


def _pool_nodes(
    x: Tensor,
    batch: Tensor,
    pool: CellAttentionPool | None,
) -> tuple[Tensor, Tensor | None]:
    if pool is None:
        return global_mean_pool(x, batch), None
    return pool(x, batch)


def _node_batch(graph: HeteroData, node_type: str, x: Tensor) -> Tensor:
    batch = getattr(graph[node_type], "batch", None)
    if batch is None:
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return batch
