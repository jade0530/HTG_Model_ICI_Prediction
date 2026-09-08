"""Minimal sample-level classifier over a cell-gene heterogeneous graph."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

Pooling = Literal["mean", "attention"]


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


class SampleGraphClassifier(nn.Module):
    """Placeholder HTG encoder: gene embeddings, weighted gene->cell, then pool.

    ``pooling='mean'`` averages cells. ``pooling='attention'`` learns which
    cells to emphasise. The input/output contract stays one logit per graph.
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 64,
        pooling: Pooling = "mean",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError("pooling must be 'mean' or 'attention'")
        self.pooling = pooling
        self.gene_emb = nn.Embedding(num_genes, hidden_dim)
        self.cell_lin = nn.Linear(hidden_dim, hidden_dim)
        self.cell_pool = CellAttentionPool(hidden_dim) if pooling == "attention" else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_cells(self, graph: HeteroData) -> tuple[Tensor, Tensor]:
        gene_x = self.gene_emb(graph["gene"].global_id)
        src, dst = graph["gene", "expressed_by", "cell"].edge_index
        weight = graph["gene", "expressed_by", "cell"].edge_weight.unsqueeze(-1)
        messages = gene_x[src] * weight
        cell_x = gene_x.new_zeros(int(graph["cell"].num_nodes), gene_x.size(-1))
        cell_x.index_add_(0, dst, messages)
        degree = gene_x.new_zeros(int(graph["cell"].num_nodes), 1)
        degree.index_add_(0, dst, weight)
        cell_x = F.relu(self.cell_lin(cell_x / degree.clamp(min=1e-6)))
        batch = getattr(graph["cell"], "batch", None)
        if batch is None:
            batch = torch.zeros(cell_x.size(0), dtype=torch.long, device=cell_x.device)
        return cell_x, batch

    def pool_cells(self, cell_x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.cell_pool is None:
            return global_mean_pool(cell_x, batch), None
        return self.cell_pool(cell_x, batch)

    def forward(self, graph: HeteroData) -> Tensor:
        cell_x, batch = self.encode_cells(graph)
        pooled, _ = self.pool_cells(cell_x, batch)
        return self.head(pooled).reshape(-1)
