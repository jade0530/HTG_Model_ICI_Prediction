"""Sample-level homogeneous GCN baseline for ICI R/NR prediction.

Unlike the HTG model, this GCN has one node type (cells) and k-NN cell-cell
edges. There are no gene nodes.

Train::

    python -m src.benchmark.gcn \\
        --dataset-root ../data/test \\
        --split-json ../outputs/train/split.json \\
        --output-dir ../../HTG_Model_ICI_Results/benchmark/gcn

Re-score a split from a saved run::

    python -m src.benchmark.gcn eval --split val --checkpoint ... --dataset-root ...
    python -m src.benchmark.gcn eval --split test --checkpoint ... --dataset-root ...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from src.benchmark.gcn.graph import build_cell_knn_graph, load_cell_expression
from src.benchmark.mlp.features import binary_labels
from src.benchmark.mlp.mlp import (
    apply_split_json,
    _split_payload,
    _write_json,
    _write_split_outputs,
)
from src.data.data_loader import select_train_hvgs
from src.data.sampler import CellSampler
from src.train.dataset import (
    DEFAULT_MANIFEST,
    REPO_ROOT,
    SampleRecord,
    assert_patient_disjoint,
    collect_records,
    split_by_patient,
)
from src.train.loop import auprc_improved, seed_everything
from src.train.metrics import ThresholdStrategy, classification_metrics, format_metrics, select_threshold
from src.train.report import plot_metrics_by_split, write_results_tables

DEFAULT_OUTPUT_DIR = REPO_ROOT.parent / "HTG_Model_ICI_Results" / "benchmark" / "gcn"
MODEL_NAME = "model.pt"
GENES_NAME = "genes.txt"


@dataclass
class GCNConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    k_neighbors: int = 10
    num_cells: int = 256
    n_hvg: int = 500
    epochs: int = 50
    patience: int = 8
    min_delta: float = 0.005
    lr: float = 1e-3
    batch_size: int = 2
    val_fraction: float = 0.25
    test_fraction: float = 0.0
    seed: int = 0
    threshold: float = 0.5
    threshold_strategy: ThresholdStrategy = "max_f1"
    use_pos_weight: bool = True
    tissue: str = "Tumor"
    ici_phase: str = "pre"


class SampleGCN(nn.Module):
    """GCN on a homogeneous cell graph, then mean-pool to one sample logit."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("in_dim, hidden_dim, and num_layers must be positive")
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(int(in_dim), int(hidden_dim)))
        for _ in range(int(num_layers) - 1):
            self.convs.append(GCNConv(int(hidden_dim), int(hidden_dim)))
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        x = graph.x
        edge_index = graph.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        pooled = global_mean_pool(x, batch)
        return self.head(pooled).reshape(-1)


def _standardize_fit(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(list(matrices), axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _pos_weight(y_train: np.ndarray, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    n_pos = float(np.sum(y_train >= 0.5))
    n_neg = float(len(y_train) - n_pos)
    return torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)


def _cell_indices(record: SampleRecord, sampler: CellSampler, *, view_index: int) -> np.ndarray:
    import anndata as ad

    adata = ad.read_h5ad(record.h5ad_path, backed="r")
    try:
        n_cells = int(adata.n_obs)
    finally:
        if adata.isbacked:
            adata.file.close()
    return sampler.sample(n_cells, training=False, view_index=view_index)


def _build_graphs(
    records: Sequence[SampleRecord],
    genes: Sequence[str],
    *,
    config: GCNConfig,
    sampler: CellSampler,
    gene_mean: np.ndarray,
    gene_std: np.ndarray,
) -> list[Data]:
    graphs = []
    for index, record in enumerate(records):
        graphs.append(
            build_cell_knn_graph(
                record,
                genes,
                sampler=sampler,
                k_neighbors=config.k_neighbors,
                training=False,
                view_index=index,
                gene_mean=gene_mean,
                gene_std=gene_std,
            )
        )
    return graphs


def _predict_loader(model: SampleGCN, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch).detach().cpu())
            labels.append(batch.y.reshape(-1).detach().cpu())
    y_true = torch.cat(labels).numpy()
    y_prob = torch.sigmoid(torch.cat(logits)).numpy()
    return y_true, y_prob


def _run_epoch(
    model: SampleGCN,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
    threshold: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        logit = model(batch)
        y = batch.y.reshape(-1).to(dtype=logit.dtype)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
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


def train_gcn(
    records: Sequence[SampleRecord],
    *,
    config: GCNConfig | None = None,
    split_json: str | Path | None = None,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> dict[str, object]:
    """Fit HVGs on train only, train a cell-kNN GCN, tune threshold on val."""
    config = config or GCNConfig()
    seed_everything(config.seed)
    if split_json is not None:
        train_items, val_items, test_items = apply_split_json(records, split_json)
    elif config.test_fraction > 0:
        train_items, val_items, test_items = split_by_patient(
            records,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
    else:
        train_items, val_items = split_by_patient(
            records, val_fraction=config.val_fraction, seed=config.seed
        )
        test_items = []
    assert_patient_disjoint(train_items, val_items, test_items)

    genes = tuple(str(name) for name in select_train_hvgs(train_items, n_hvg=config.n_hvg))
    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    train_cells = [
        load_cell_expression(record.h5ad_path, genes, _cell_indices(record, sampler, view_index=index))
        for index, record in enumerate(train_items)
    ]
    gene_mean, gene_std = _standardize_fit(train_cells)
    train_graphs = _build_graphs(
        train_items, genes, config=config, sampler=sampler, gene_mean=gene_mean, gene_std=gene_std
    )
    val_graphs = _build_graphs(
        val_items, genes, config=config, sampler=sampler, gene_mean=gene_mean, gene_std=gene_std
    )
    test_graphs = (
        _build_graphs(test_items, genes, config=config, sampler=sampler, gene_mean=gene_mean, gene_std=gene_std)
        if test_items
        else []
    )

    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False)
    model = SampleGCN(len(genes), config.hidden_dim, config.num_layers, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    y_train = binary_labels(train_items)
    pos_weight = _pos_weight(y_train, config.use_pos_weight)
    best_val_auprc = float("-inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    patience_count = 0
    if config.epochs < 1 or config.patience < 1 or config.min_delta < 0:
        raise ValueError("epochs and patience must be >= 1; min_delta must be non-negative")

    for epoch in range(config.epochs):
        train_metrics = _run_epoch(
            model, train_loader, optimizer=optimizer, pos_weight=pos_weight, threshold=config.threshold
        )
        val_metrics = _run_epoch(
            model, val_loader, optimizer=None, pos_weight=None, threshold=config.threshold
        )
        val_auprc = val_metrics.get("auprc", float("nan"))
        try:
            val_auprc = float(val_auprc)
        except (TypeError, ValueError):
            val_auprc = float("nan")
        if auprc_improved(val_auprc, best_val_auprc, config.min_delta):
            best_val_auprc = val_auprc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if log:
            print(f"epoch {epoch} {format_metrics('train', train_metrics)}")
            print(f"epoch {epoch} {format_metrics('val', val_metrics)}")
        if patience_count >= config.patience:
            if log:
                print(f"Early stopping at epoch {epoch}; best val AUPRC={best_val_auprc}")
            break

    model.load_state_dict(best_state)
    _, train_prob = _predict_loader(model, DataLoader(train_graphs, batch_size=config.batch_size, shuffle=False))
    y_val, val_prob = _predict_loader(model, val_loader)
    threshold = select_threshold(y_val, val_prob, strategy=config.threshold_strategy, fixed=config.threshold)
    y_test = None
    test_prob = None
    if test_graphs:
        y_test, test_prob = _predict_loader(model, DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False))

    split_metrics: dict[str, dict[str, float]] = {
        "train": classification_metrics(y_train, train_prob, threshold=threshold),
        "val": classification_metrics(y_val, val_prob, threshold=threshold),
    }
    if y_test is not None and test_prob is not None:
        split_metrics["test"] = classification_metrics(y_test, test_prob, threshold=threshold)

    if log:
        print(f"samples={len(records)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")
        print(f"genes={len(genes)} k={config.k_neighbors} threshold={threshold:.4f}")
        for name, metrics in split_metrics.items():
            print(format_metrics(name, metrics))

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {item.name: getattr(config, item.name) for item in fields(config)},
                "genes": list(genes),
                "scaler_mean": gene_mean,
                "scaler_std": gene_std,
                "threshold": threshold,
                "threshold_strategy": config.threshold_strategy,
                "in_dim": len(genes),
            },
            out / MODEL_NAME,
        )
        (out / GENES_NAME).write_text("\n".join(genes) + "\n")
        _write_json(out / "config.json", {item.name: getattr(config, item.name) for item in fields(config)})
        _write_json(
            out / "split.json",
            {
                "train": _split_payload(train_items),
                "val": _split_payload(val_items),
                "test": _split_payload(test_items),
            },
        )
        _write_json(
            out / "threshold.json",
            {"strategy": config.threshold_strategy, "threshold": threshold, "tuned_on": "val"},
        )
        write_results_tables(out, split_metrics)
        plot_metrics_by_split(split_metrics, out)
        checkpoint = out / MODEL_NAME
        _write_split_outputs(
            out, "train", train_items, y_train, train_prob,
            threshold=threshold, threshold_strategy=config.threshold_strategy, checkpoint=checkpoint,
        )
        _write_split_outputs(
            out, "val", val_items, y_val, val_prob,
            threshold=threshold, threshold_strategy=config.threshold_strategy, checkpoint=checkpoint,
        )
        if test_items and y_test is not None and test_prob is not None:
            _write_split_outputs(
                out, "test", test_items, y_test, test_prob,
                threshold=threshold, threshold_strategy=config.threshold_strategy, checkpoint=checkpoint,
            )
        if log:
            print(f"output_dir={out}")

    return {
        "model": model,
        "genes": genes,
        "threshold": threshold,
        "metrics": split_metrics,
        "output_dir": out,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
    }


def _resolve_checkpoint(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        return resolved
    if resolved.name == MODEL_NAME:
        return resolved.parent
    raise FileNotFoundError(f"checkpoint directory not found: {resolved}")


def _config_from_payload(payload: dict) -> GCNConfig:
    allowed = {item.name for item in fields(GCNConfig)}
    values = {key: value for key, value in (payload or {}).items() if key in allowed}
    return GCNConfig(**values)


def evaluate_gcn(
    records: Sequence[SampleRecord],
    *,
    checkpoint: str | Path,
    split: str,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> dict[str, object]:
    """Score train/val/test using a saved GCN run and its frozen gene list / split."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")
    run_dir = _resolve_checkpoint(checkpoint)
    model_path = run_dir / MODEL_NAME
    split_path = run_dir / "split.json"
    if not model_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"incomplete GCN run in {run_dir}")
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    config = _config_from_payload(payload.get("config") or {})
    genes = tuple(str(name) for name in payload["genes"])
    gene_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    gene_std = np.asarray(payload["scaler_std"], dtype=np.float32)
    threshold = float(payload.get("threshold", 0.5))
    strategy = str(payload.get("threshold_strategy", "max_f1"))
    model = SampleGCN(int(payload["in_dim"]), config.hidden_dim, config.num_layers, config.dropout)
    model.load_state_dict(payload["model"])
    train_items, val_items, test_items = apply_split_json(records, split_path)
    chosen = {"train": train_items, "val": val_items, "test": test_items}[split]
    if not chosen:
        raise ValueError(f"saved split has no {split} samples")
    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    graphs = _build_graphs(
        chosen, genes, config=config, sampler=sampler, gene_mean=gene_mean, gene_std=gene_std
    )
    y_true, y_prob = _predict_loader(model, DataLoader(graphs, batch_size=config.batch_size, shuffle=False))
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    if log:
        print(format_metrics(split, metrics))
    out = Path(output_dir) if output_dir is not None else run_dir
    _write_split_outputs(
        out,
        split,
        chosen,
        y_true,
        y_prob,
        threshold=threshold,
        threshold_strategy=strategy,
        checkpoint=model_path,
    )
    return {"metrics": metrics, "threshold": threshold, "output_dir": out, "split": split}


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sample-level homogeneous GCN baseline")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data" / "test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--num-cells", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-strategy", choices=("max_f1", "youden", "fixed"), default="max_f1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args(argv)


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GCN run")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    return parser.parse_args(argv)


def _collect(args: argparse.Namespace, *, require_label: bool = True) -> list[SampleRecord]:
    return collect_records(
        args.dataset_root,
        args.manifest,
        tissue=None if str(args.tissue).lower() == "all" else args.tissue,
        ici_phase=None if str(args.ici_phase).lower() == "all" else args.ici_phase,
        require_label=require_label,
    )


def main_train(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    records = _collect(args)
    result = train_gcn(
        records,
        config=GCNConfig(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_gnn_layers,
            dropout=args.dropout,
            k_neighbors=args.k_neighbors,
            num_cells=args.num_cells,
            n_hvg=args.n_hvg,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            lr=args.lr,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            threshold=args.threshold,
            threshold_strategy=args.threshold_strategy,
            use_pos_weight=not args.no_pos_weight,
            tissue=args.tissue,
            ici_phase=args.ici_phase,
        ),
        split_json=args.split_json,
        output_dir=args.output_dir,
    )
    val = result["metrics"]["val"]
    auprc = val.get("auprc", float("nan"))
    print(
        f"val_auprc={'nan' if auprc != auprc else f'{auprc:.4f}'} "
        f"threshold={result['threshold']:.4f} output_dir={result['output_dir']}"
    )


def main_eval(argv: list[str] | None = None) -> None:
    args = parse_eval_args(argv)
    records = _collect(args)
    result = evaluate_gcn(
        records,
        checkpoint=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
    )
    print(f"wrote {result['output_dir']}/{result['split']}/predictions.csv")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        main_eval(argv[1:])
        return
    main_train(argv)


if __name__ == "__main__":
    main()
