"""Minimal sample-level classifier over a cell-gene heterogeneous graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.pool import global_mean_pool


class SampleGraphClassifier(nn.Module):
    """Placeholder HTG encoder: gene embeddings, weighted gene->cell, then pool.

    Replace the body of ``forward`` when a richer architecture is ready. The
    input/output contract stays the same: one logit per sample graph.
    """

    def __init__(self, num_genes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gene_emb = nn.Embedding(num_genes, hidden_dim)
        self.cell_lin = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: HeteroData) -> Tensor:
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
        pooled = global_mean_pool(cell_x, batch)
        return self.head(pooled).reshape(-1)
