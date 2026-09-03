"""Training / evaluation loop for sample-level graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from src.data.data_loader import SampleData
from src.data.sampler import CellSampler
from src.graph.build_local_graph import GeneStrategy, GeneUniverse
from src.train.dataset import (
    SampleRecord,
    SampleGraphDataset,
    assert_patient_disjoint,
    split_by_patient,
)
from src.train.metrics import classification_metrics, format_metrics
from src.train.model import SampleGraphClassifier


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    num_cells: int = 512
    gene_strategy: GeneStrategy = "hvg"
    batch_size: int = 2
    epochs: int = 5
    lr: float = 1e-3
    val_fraction: float = 0.25
    seed: int = 0
    device: str = "cpu"
    threshold: float = 0.5


@dataclass
class EpochResult:
    epoch: int
    train: dict[str, float] = field(default_factory=dict)
    val: dict[str, float] = field(default_factory=dict)


def run_epoch(
    model: SampleGraphClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        y = batch.y.reshape(-1).to(dtype=logit.dtype)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()) * int(y.numel()))
        logits.append(logit.detach().cpu())
        labels.append(y.detach().cpu())
    y_true = torch.cat(labels).numpy()
    y_prob = torch.sigmoid(torch.cat(logits)).numpy()
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(np.sum(losses) / max(len(y_true), 1))
    return metrics


def _loader(
    items: Sequence[SampleData | SampleRecord],
    *,
    sampler: CellSampler,
    gene_universe: GeneUniverse,
    config: TrainConfig,
    training: bool,
) -> DataLoader:
    dataset = SampleGraphDataset(
        items,
        sampler=sampler,
        gene_universe=gene_universe,
        gene_strategy=config.gene_strategy,
        training=training,
        seed=config.seed,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training)


def fit(
    samples: Sequence[SampleData],
    gene_universe: GeneUniverse,
    *,
    config: TrainConfig | None = None,
    log: bool = True,
) -> list[EpochResult]:
    """Train a sample-level graph classifier and report acc / AUPRC / F1."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    train_samples, val_samples = split_by_patient(
        samples, val_fraction=config.val_fraction, seed=config.seed
    )
    assert_patient_disjoint(train_samples, val_samples)
    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    train_loader = _loader(
        train_samples,
        sampler=sampler,
        gene_universe=gene_universe,
        config=config,
        training=True,
    )
    val_loader = _loader(
        val_samples,
        sampler=sampler,
        gene_universe=gene_universe,
        config=config,
        training=False,
    )
    model = SampleGraphClassifier(len(gene_universe), hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[EpochResult] = []
    for epoch in range(config.epochs):
        train_metrics = run_epoch(
            model, train_loader, device=device, optimizer=optimizer, threshold=config.threshold
        )
        val_metrics = run_epoch(
            model, val_loader, device=device, optimizer=None, threshold=config.threshold
        )
        result = EpochResult(epoch=epoch, train=train_metrics, val=val_metrics)
        history.append(result)
        if log:
            print(f"epoch {epoch} {format_metrics('train', train_metrics)}")
            print(f"epoch {epoch} {format_metrics('val', val_metrics)}")
    return history


def evaluate(
    model: SampleGraphClassifier,
    samples: Iterable[SampleData],
    gene_universe: GeneUniverse,
    *,
    config: TrainConfig | None = None,
) -> dict[str, float]:
    config = config or TrainConfig()
    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    loader = _loader(
        list(samples),
        sampler=sampler,
        gene_universe=gene_universe,
        config=config,
        training=False,
    )
    return run_epoch(
        model,
        loader,
        device=torch.device(config.device),
        optimizer=None,
        threshold=config.threshold,
    )
