"""Sample-level MLP baseline for ICI R/NR prediction.

Uses the existing SampleRecord loader, patient-disjoint split, and mean-expression
HVG features.

Train::

    python -m src.benchmark.mlp \\
        --dataset-root ../data/test \\
        --split-json ../outputs/train/split.json \\
        --output-dir ../../HTG_Model_ICI_Results/benchmark/mlp

Re-score a split from a saved run::

    python -m src.benchmark.mlp eval --split val --checkpoint ... --dataset-root ...
    python -m src.benchmark.mlp eval --split test --checkpoint ... --dataset-root ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.benchmark.mlp.features import binary_labels, feature_matrix
from src.data.data_loader import select_train_hvgs
from src.train.dataset import (
    DEFAULT_MANIFEST,
    REPO_ROOT,
    SampleRecord,
    assert_patient_disjoint,
    collect_records,
    patient_key,
    split_by_patient,
)
from src.train.loop import auprc_improved, seed_everything
from src.train.metrics import ThresholdStrategy, classification_metrics, format_metrics, select_threshold
from src.train.report import (
    plot_metrics_by_split,
    plot_roc_pr,
    write_confusion_matrix,
    write_predictions,
    write_results_tables,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT.parent / "HTG_Model_ICI_Results" / "benchmark" / "mlp"
MODEL_NAME = "model.pt"
GENES_NAME = "genes.txt"


def _split_payload(items: Sequence[SampleRecord]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": str(item.sample_id),
            "patient_key": patient_key(item),
            "label": str(item.label),
        }
        for item in items
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _index_records(records: Sequence[SampleRecord]) -> dict[str, SampleRecord]:
    index: dict[str, SampleRecord] = {}
    for record in records:
        key = str(record.sample_id)
        if key in index:
            raise ValueError(f"duplicate sample_id in dataset: {key}")
        index[key] = record
    return index


def apply_split_json(
    records: Sequence[SampleRecord],
    split_json: str | Path,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    """Reuse a frozen GNN/benchmark split: match rows by ``sample_id``."""
    payload = json.loads(Path(split_json).read_text())
    index = _index_records(records)

    def resolve(name: str) -> list[SampleRecord]:
        rows = payload.get(name) or []
        missing = [str(row["sample_id"]) for row in rows if str(row["sample_id"]) not in index]
        if missing:
            raise ValueError(f"{split_json} {name} samples not found: {missing[:8]}")
        return [index[str(row["sample_id"])] for row in rows]

    train_items = resolve("train")
    val_items = resolve("val")
    test_items = resolve("test")
    if not train_items or not val_items:
        raise ValueError(f"{split_json} must contain non-empty train and val lists")
    assert_patient_disjoint(train_items, val_items, test_items)
    return train_items, val_items, test_items


def _prediction_rows(
    records: Sequence[SampleRecord],
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for record, prob in zip(records, y_prob, strict=True):
        rows.append(
            {
                "sample_id": str(record.sample_id),
                "patient_key": patient_key(record),
                "cell_type": "",
                "label": str(record.label),
                "prob_R": float(prob),
                "pred": "R" if float(prob) >= threshold else "NR",
            }
        )
    return rows


def _write_split_outputs(
    output_dir: Path,
    split: str,
    records: Sequence[SampleRecord],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    threshold_strategy: str,
    checkpoint: Path,
) -> dict[str, float]:
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    write_confusion_matrix(output_dir, split, y_true, y_prob, threshold=threshold)
    plot_roc_pr(output_dir, split, y_true, y_prob, threshold=threshold)
    write_predictions(
        output_dir / split,
        rows=_prediction_rows(records, y_prob, threshold=threshold),
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        checkpoint=str(checkpoint),
        metrics=metrics,
    )
    return metrics


@dataclass
class MLPConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    n_hvg: int = 500
    epochs: int = 50
    patience: int = 8
    min_delta: float = 0.005
    lr: float = 1e-3
    val_fraction: float = 0.25
    test_fraction: float = 0.0
    seed: int = 0
    threshold: float = 0.5
    threshold_strategy: ThresholdStrategy = "max_f1"
    use_pos_weight: bool = True
    tissue: str = "Tumor"
    ici_phase: str = "pre"


class SampleMLP(nn.Module):
    """Fully connected network over a sample-level gene vector -> one logit."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for hidden in hidden_dims:
            width = int(hidden)
            if width < 1:
                raise ValueError("hidden layer sizes must be positive")
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def _standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32, copy=False)


def _pos_weight(y_train: np.ndarray, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    n_pos = float(np.sum(y_train >= 0.5))
    n_neg = float(len(y_train) - n_pos)
    return torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)


def _predict_proba(model: SampleMLP, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(features, dtype=np.float32)))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def _epoch_pass(
    model: SampleMLP,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    training = optimizer is not None
    model.train(training)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    logit = model(x)
    loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_prob = torch.sigmoid(logit).cpu().numpy()
    metrics = classification_metrics(labels, y_prob, threshold=threshold)
    metrics["loss"] = float(loss.detach().item())
    return metrics, y_prob


def train_mlp(
    records: Sequence[SampleRecord],
    *,
    config: MLPConfig | None = None,
    split_json: str | Path | None = None,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> dict[str, object]:
    """Fit HVGs on train only, train an MLP, tune threshold on val, score test if present."""
    config = config or MLPConfig()
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
    x_train_raw = feature_matrix(train_items, genes)
    x_val_raw = feature_matrix(val_items, genes)
    mean, std = _standardize_fit(x_train_raw)
    x_train = _standardize(x_train_raw, mean, std)
    x_val = _standardize(x_val_raw, mean, std)
    y_train = binary_labels(train_items)
    y_val = binary_labels(val_items)

    model = SampleMLP(len(genes), config.hidden_dims, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos_weight = _pos_weight(y_train, config.use_pos_weight)
    best_val_auprc = float("-inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    patience_count = 0
    if config.epochs < 1 or config.patience < 1 or config.min_delta < 0:
        raise ValueError("epochs and patience must be >= 1; min_delta must be non-negative")

    for epoch in range(config.epochs):
        train_metrics, _ = _epoch_pass(
            model,
            x_train,
            y_train,
            optimizer=optimizer,
            pos_weight=pos_weight,
            threshold=config.threshold,
        )
        val_metrics, _ = _epoch_pass(
            model,
            x_val,
            y_val,
            optimizer=None,
            pos_weight=None,
            threshold=config.threshold,
        )
        val_auprc = val_metrics.get("auprc", float("nan"))
        try:
            val_auprc = float(val_auprc)
        except (TypeError, ValueError):
            val_auprc = float("nan")
        improved = auprc_improved(val_auprc, best_val_auprc, config.min_delta)
        if improved:
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
    train_prob = _predict_proba(model, x_train)
    val_prob = _predict_proba(model, x_val)
    threshold = select_threshold(
        y_val, val_prob, strategy=config.threshold_strategy, fixed=config.threshold
    )
    test_prob = None
    y_test = None
    if test_items:
        y_test = binary_labels(test_items)
        test_prob = _predict_proba(
            model, _standardize(feature_matrix(test_items, genes), mean, std)
        )

    split_metrics: dict[str, dict[str, float]] = {
        "train": classification_metrics(y_train, train_prob, threshold=threshold),
        "val": classification_metrics(y_val, val_prob, threshold=threshold),
    }
    if test_items and y_test is not None and test_prob is not None:
        split_metrics["test"] = classification_metrics(y_test, test_prob, threshold=threshold)

    if log:
        print(f"samples={len(records)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")
        print(f"genes={len(genes)} threshold={threshold:.4f} strategy={config.threshold_strategy}")
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
                "scaler_mean": mean,
                "scaler_std": std,
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


def _hidden_dims_from_config(payload: dict) -> tuple[int, ...]:
    raw = payload.get("hidden_dims", (128, 64))
    return tuple(int(value) for value in raw)


def evaluate_mlp(
    records: Sequence[SampleRecord],
    *,
    checkpoint: str | Path,
    split: str,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> dict[str, object]:
    """Score train/val/test using a saved MLP run and its frozen gene list / split."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")
    run_dir = _resolve_checkpoint(checkpoint)
    model_path = run_dir / MODEL_NAME
    split_path = run_dir / "split.json"
    if not model_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"incomplete MLP run in {run_dir}")
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    genes = tuple(str(name) for name in payload["genes"])
    mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    std = np.asarray(payload["scaler_std"], dtype=np.float32)
    threshold = float(payload.get("threshold", 0.5))
    strategy = str(payload.get("threshold_strategy", "max_f1"))
    hidden = _hidden_dims_from_config(payload.get("config") or {})
    dropout = float((payload.get("config") or {}).get("dropout", 0.2))
    model = SampleMLP(int(payload["in_dim"]), hidden, dropout)
    model.load_state_dict(payload["model"])
    train_items, val_items, test_items = apply_split_json(records, split_path)
    chosen = {"train": train_items, "val": val_items, "test": test_items}[split]
    if not chosen:
        raise ValueError(f"saved split has no {split} samples")
    y_true = binary_labels(chosen)
    y_prob = _predict_proba(model, _standardize(feature_matrix(chosen, genes), mean, std))
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


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if not parts:
        raise ValueError("hidden-dims must be a comma-separated list of integers")
    return tuple(int(item) for item in parts)


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sample-level MLP baseline")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data" / "test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Frozen split.json (GNN or previous RF/MLP run). Overrides val/test fractions",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
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
    parser = argparse.ArgumentParser(description="Evaluate a saved MLP run")
    parser.add_argument("--checkpoint", type=Path, required=True, help="MLP output directory (contains model.pt)")
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
    result = train_mlp(
        records,
        config=MLPConfig(
            hidden_dims=_parse_hidden_dims(args.hidden_dims),
            dropout=args.dropout,
            n_hvg=args.n_hvg,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            lr=args.lr,
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
    result = evaluate_mlp(
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
