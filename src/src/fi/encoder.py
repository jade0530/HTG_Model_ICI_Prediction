"""Edge-aware FI gene encoder and the two self-supervised decoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv

from src.fi.message_graph import CURATED_SEMANTICS, EDGE_ATTR_NAMES


class FIEncoder(nn.Module):
    """Trainable gene embeddings updated by GINEConv on FI edge attributes.

    Each layer is convolution, residual, LayerNorm, ReLU, then dropout.
    ``edge_attr`` is projected inside GINEConv and added to the neighbour message.
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        edge_dim: int = len(EDGE_ATTR_NAMES),
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.embedding = nn.Embedding(num_genes, hidden_dim)
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            update = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(update, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.embedding.weight
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            updated = conv(x, edge_index, edge_attr)
            x = F.dropout(F.relu(norm(updated + residual)), p=self.dropout, training=self.training)
        return x


class PairMLP(nn.Module):
    """Score or reconstruct an unordered gene pair from its two embeddings."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        left = embeddings[pairs[:, 0]]
        right = embeddings[pairs[:, 1]]
        features = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
        return self.mlp(features)


class FIPretrainer(nn.Module):
    """Encoder plus link and semantic decoders. ICI labels are not used."""

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = FIEncoder(num_genes, hidden_dim, num_layers, dropout)
        self.link_decoder = PairMLP(hidden_dim, 1)
        self.semantic_decoder = PairMLP(hidden_dim, len(CURATED_SEMANTICS))

    def encode(self, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.encoder(edge_index, edge_attr)

    def link_logits(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        return self.link_decoder(embeddings, pairs).squeeze(-1)

    def semantic_logits(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        return self.semantic_decoder(embeddings, pairs)
