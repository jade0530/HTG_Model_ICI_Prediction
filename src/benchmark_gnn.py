"""Homogeneous cell-cell GNN baseline on raw h5ad expression.

Each h5ad is one sample. Cells are nodes; KNN on the aligned gene vectors
builds cell-cell edges. A SAGE GNN pools cells to one R/NR score.
Pass the HTG ``split.json`` (or sample-id lists) so the split matches the
other baselines. Hyperparameters are scored on the given val fold.

``python benchmark_gnn.py predict --checkpoint RUN_DIR --dataset-root NEW_H5ADS --output-dir OUT``
scores unseen sample h5ads with the saved model. Gene names are aligned by
name; genes missing from a new file are filled with 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import src  # noqa: F401  # pins CUDA_VISIBLE_DEVICES before torch

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphConv, global_mean_pool
from torch_geometric.utils import to_undirected

from src.train.benchmark_data import (
    LABEL_TO_Y,
    add_data_args,
    add_predict_args,
    assign_split,
    load_cell_features,
    load_records,
    resolve_genes,
    score_metrics,
    write_history_csv,
    write_predict_report,
    write_search_csv,
)
from src.train.loop import EpochResult, resolve_device, seed_everything
from src.train.metrics import format_metrics, select_threshold
from src.train.model import CellAttentionPool
from src.train.report import write_run_report

K_VALUES = (5, 15)
HIDDEN_DIMS = (32, 64)
NUM_LAYERS = (2, 3)
DROPOUTS = (0.1, 0.3)


class CellKNNClassifier(nn.Module):
    """SAGE or GAT over a cell-cell KNN graph, then one sample logit."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        encoder: str = "sage",
        pooling: str = "mean",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if encoder == "gat":
                conv = GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, add_self_loops=True)
            else:
                conv = GraphConv(hidden_dim, hidden_dim, aggr="mean")
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.attn = CellAttentionPool(hidden_dim) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tensor:
        x = F.relu(self.input(data.x))
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, data.edge_index) + x)), p=self.dropout, training=self.training)
        if self.attn is not None:
            pooled, _ = self.attn(x, data.batch)
        else:
            pooled = global_mean_pool(x, data.batch)
        return self.head(pooled).reshape(-1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell-cell KNN GNN baseline on raw sample h5ads")
    add_data_args(parser)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-cells", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def knn_edge_index(cells: np.ndarray, k: int) -> Tensor:
    n = int(cells.shape[0])
    k_eff = min(int(k), n - 1)
    if k_eff < 1:
        return torch.empty((2, 0), dtype=torch.long)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(cells)
    idx = nbrs.kneighbors(cells, return_distance=False)
    src: list[int] = []
    dst: list[int] = []
    for i, row in enumerate(idx):
        for j in row:
            if int(j) == i:
                continue
            src.append(i)
            dst.append(int(j))
    return to_undirected(torch.tensor([src, dst], dtype=torch.long))


def make_knn_graph(cells: np.ndarray, label: int, k: int) -> Data:
    x = torch.tensor(cells, dtype=torch.float32)
    return Data(x=x, edge_index=knn_edge_index(cells, k), y=torch.tensor([label], dtype=torch.float32))


def scale_cells(cell_list: list[np.ndarray], scaler: StandardScaler) -> list[np.ndarray]:
    return [scaler.transform(cells).astype(np.float32) for cells in cell_list]


def load_fold_cells(records, gene_names, *, num_cells: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    cells = [
        load_cell_features(record, gene_names, num_cells=num_cells, seed=seed, view_index=i)
        for i, record in enumerate(records)
    ]
    y = np.array([LABEL_TO_Y[record.label] for record in records], dtype=np.int64)
    return cells, y


def make_loader(cell_list, labels, k: int, batch_size: int, shuffle: bool) -> DataLoader:
    graphs = [make_knn_graph(cells, int(label), k) for cells, label in zip(cell_list, labels)]
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def make_model(in_dim: int, params: dict) -> CellKNNClassifier:
    return CellKNNClassifier(
        in_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        encoder=params.get("encoder", "sage"),
        pooling=params.get("pooling", "mean"),
    )


def pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def run_epoch(model, loader, device, *, optimizer=None, weight=None, threshold: float = 0.5) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        y = batch.y.reshape(-1).to(dtype=logit.dtype)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=weight)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()) * int(y.numel()))
        logits.append(logit.detach().cpu())
        labels.append(y.detach().cpu())
    y_true = torch.cat(labels).numpy()
    y_prob = torch.sigmoid(torch.cat(logits)).numpy()
    return score_metrics(y_true, y_prob, threshold=threshold, loss=float(np.sum(losses) / len(y_true)))


@torch.no_grad()
def predict_scores(model, loader, device) -> np.ndarray:
    model.eval()
    logits = []
    for batch in loader:
        logits.append(model(batch.to(device)).detach().cpu())
    return torch.sigmoid(torch.cat(logits)).numpy()


def run_epochs(model, train_loader, val_loader, *, device, epochs, lr, weight) -> list[EpochResult]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[EpochResult] = []
    for epoch in range(epochs):
        history.append(
            EpochResult(
                epoch=epoch,
                train=run_epoch(model, train_loader, device, optimizer=optimizer, weight=weight),
                val=run_epoch(model, val_loader, device, weight=weight),
            )
        )
    return history


def tune(train_cells, y_train, val_cells, y_val, *, in_dim, epochs, batch_size, device, seed) -> tuple[dict, list[dict]]:
    weight = pos_weight(y_train).to(device)
    rows = []
    best_params = None
    best_auprc = -1.0
    for k, hidden_dim, num_layers, dropout in product(K_VALUES, HIDDEN_DIMS, NUM_LAYERS, DROPOUTS):
        params = {
            "k": k,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "encoder": "sage",
            "pooling": "mean",
            "lr": 1e-3,
        }
        seed_everything(seed)
        train_loader = make_loader(train_cells, y_train, k, batch_size, shuffle=True)
        val_loader = make_loader(val_cells, y_val, k, batch_size, shuffle=False)
        model = make_model(in_dim, params).to(device)
        history = run_epochs(
            model, train_loader, val_loader, device=device, epochs=epochs, lr=params["lr"], weight=weight
        )
        metrics = history[-1].val
        row = {**params, **{f"val_{key}": value for key, value in metrics.items()}}
        rows.append(row)
        print(
            f"gnn k={k} hidden={hidden_dim} layers={num_layers} "
            f"dropout={dropout} val_auprc={metrics['auprc']:.4f}"
        )
        if metrics["auprc"] > best_auprc:
            best_auprc = metrics["auprc"]
            best_params = params
    return best_params, rows


def main_train(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.split_json is None and (args.train_ids is None or args.val_ids is None):
        raise SystemExit("pass --split-json, or both --train-ids and --val-ids")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")

    records = load_records(args.dataset_root, args.manifest, args.tissue, args.ici_phase)
    folds = assign_split(
        records,
        split_json=args.split_json,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        test_ids=args.test_ids,
    )
    print(f"samples train={len(folds['train'])} val={len(folds['val'])} test={len(folds['test'])}")

    genes = resolve_genes(folds["train"], n_hvg=args.n_hvg, gene_universe=args.gene_universe)
    train_cells, y_train = load_fold_cells(folds["train"], genes, num_cells=args.num_cells, seed=args.seed)
    val_cells, y_val = load_fold_cells(folds["val"], genes, num_cells=args.num_cells, seed=args.seed)
    test_cells, y_test = (
        load_fold_cells(folds["test"], genes, num_cells=args.num_cells, seed=args.seed)
        if folds["test"]
        else (None, None)
    )

    scaler = StandardScaler()
    scaler.fit(np.vstack(train_cells))
    train_cells = scale_cells(train_cells, scaler)
    val_cells = scale_cells(val_cells, scaler)
    if test_cells is not None:
        test_cells = scale_cells(test_cells, scaler)
    print(f"features={len(genes)} cells={args.num_cells}")

    best_params, search_rows = tune(
        train_cells,
        y_train,
        val_cells,
        y_val,
        in_dim=len(genes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    print(f"best_params={best_params}")
    seed_everything(args.seed)
    train_loader = make_loader(train_cells, y_train, best_params["k"], args.batch_size, shuffle=True)
    val_loader = make_loader(val_cells, y_val, best_params["k"], args.batch_size, shuffle=False)
    model = make_model(len(genes), best_params).to(device)
    history = run_epochs(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=best_params["lr"],
        weight=pos_weight(y_train).to(device),
    )
    for row in history:
        print(f"epoch {row.epoch} {format_metrics('train', row.train)}")
        print(f"epoch {row.epoch} {format_metrics('val', row.val)}")

    train_scores = predict_scores(
        model, make_loader(train_cells, y_train, best_params["k"], args.batch_size, False), device
    )
    val_scores = predict_scores(
        model, make_loader(val_cells, y_val, best_params["k"], args.batch_size, False), device
    )
    threshold = select_threshold(y_val, val_scores, strategy=args.threshold_strategy, fixed=args.threshold)
    predictions = {"train": (y_train, train_scores), "val": (y_val, val_scores)}
    if test_cells is not None:
        predictions["test"] = (
            y_test,
            predict_scores(model, make_loader(test_cells, y_test, best_params["k"], args.batch_size, False), device),
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "model": "cell_knn_gnn",
                "best_params": best_params,
                "epochs": args.epochs,
                "num_cells": args.num_cells,
                "n_hvg": args.n_hvg,
                "n_features": len(genes),
                "seed": args.seed,
                "split_json": str(args.split_json) if args.split_json else None,
                "gene_universe": str(args.gene_universe) if args.gene_universe else None,
                "threshold_strategy": args.threshold_strategy,
            },
            indent=2,
        )
        + "\n"
    )
    (out / "split.json").write_text(
        json.dumps(
            {
                name: [
                    {"sample_id": record.sample_id, "patient_key": record.patient_key, "label": record.label}
                    for record in folds[name]
                ]
                for name in ("train", "val", "test")
            },
            indent=2,
        )
        + "\n"
    )
    (out / "gene_universe.txt").write_text("\n".join(genes) + "\n")
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
    write_search_csv(out / "hyperparam_search.csv", search_rows)
    write_history_csv(out / "history.csv", history)
    torch.save(model.state_dict(), out / "model.pt")
    dump(
        {
            "genes": genes,
            "scaler": scaler,
            "threshold": threshold,
            "threshold_strategy": args.threshold_strategy,
            "best_params": best_params,
            "num_cells": args.num_cells,
            "in_dim": len(genes),
        },
        out / "model.joblib",
    )
    write_run_report(
        out,
        history=history,
        predictions=predictions,
        threshold=threshold,
        threshold_strategy=args.threshold_strategy,
    )
    print(f"threshold={threshold:.4f} output_dir={out}")


def parse_predict_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score unseen h5ads with a saved cell-KNN GNN run")
    add_predict_args(parser)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args(argv)


def predict(
    checkpoint: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    manifest: Path | None = None,
    tissue: str = "all",
    ici_phase: str = "all",
    device: str = "auto",
    batch_size: int = 2,
) -> dict:
    from src.train.dataset import DEFAULT_MANIFEST

    path = Path(checkpoint)
    run_dir = path if path.is_dir() else path.parent
    bundle = load(run_dir / "model.joblib")
    resolved = resolve_device(device)
    model = make_model(bundle["in_dim"], bundle["best_params"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=resolved))
    model.to(resolved)
    records = load_records(
        dataset_root,
        manifest or DEFAULT_MANIFEST,
        tissue,
        ici_phase,
        require_label=False,
    )
    cells = [
        load_cell_features(
            record,
            bundle["genes"],
            num_cells=bundle["num_cells"],
            seed=0,
            view_index=i,
        )
        for i, record in enumerate(records)
    ]
    cells = scale_cells(cells, bundle["scaler"])
    labels = np.zeros(len(records), dtype=np.int64)
    loader = make_loader(cells, labels, bundle["best_params"]["k"], batch_size, shuffle=False)
    scores = predict_scores(model, loader, resolved)
    print(f"samples={len(records)} threshold={bundle['threshold']:.4f} checkpoint={run_dir / 'model.pt'}")
    metrics = write_predict_report(
        output_dir,
        records,
        scores,
        threshold=float(bundle["threshold"]),
        checkpoint=str(run_dir / "model.pt"),
        threshold_strategy=bundle.get("threshold_strategy", "max_f1"),
    )
    return {"output_dir": Path(output_dir), "metrics": metrics, "scores": scores}


def main_predict(argv: list[str] | None = None) -> None:
    args = parse_predict_args(argv)
    result = predict(
        args.checkpoint,
        args.dataset_root,
        args.output_dir,
        manifest=args.manifest,
        tissue=args.tissue,
        ici_phase=args.ici_phase,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"wrote {result['output_dir']}/predictions.csv")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "predict":
        main_predict(argv[1:])
        return
    main_train(argv)


if __name__ == "__main__":
    main()
