"""Training / evaluation loop for sample-level graphs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from src.data.data_loader import DEFAULT_N_HVG, SampleData
from src.data.sampler import CellSampler
from src.graph.build_local_graph import GeneStrategy, GeneUniverse
from src.train.dataset import (
    CellTypeLevel,
    SamplingMode,
    SampleRecord,
    SampleGraphDataset,
    annotation_keys_for,
    build_gene_universe,
    patient_key,
    split_by_patient,
)
from src.train.metrics import ThresholdStrategy, classification_metrics, format_metrics, select_threshold
from src.train.model import EncoderKind, Pooling, Readout, SampleGraphClassifier
from src.train.report import write_predictions, write_run_report


@dataclass
class TrainConfig:
    """Stage 1 training knobs. ``encoder`` is ``sage``, ``gat``, or ``placeholder``."""

    hidden_dim: int = 64
    num_cells: int = 512
    pooling: Pooling = "mean"
    readout: Readout = "cell"
    encoder: EncoderKind = "sage"
    num_gnn_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.1
    gene_strategy: GeneStrategy = "hvg"
    sampling_mode: SamplingMode = "random"
    cell_type_level: CellTypeLevel = "fine"
    cell_types: tuple[str, ...] = ()
    batch_size: int = 2
    epochs: int = 5
    lr: float = 1e-3
    val_fraction: float = 0.25
    test_fraction: float = 0.0
    n_hvg: int = DEFAULT_N_HVG
    cache_samples: bool = True
    use_pos_weight: bool = True
    seed: int = 0
    device: str = "auto"
    threshold: float = 0.5
    threshold_strategy: ThresholdStrategy = "max_f1"


@dataclass
class EpochResult:
    epoch: int
    train: dict[str, float] = field(default_factory=dict)
    val: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainResult:
    history: list[EpochResult]
    model: SampleGraphClassifier
    gene_universe: GeneUniverse
    hvg_by_sample: dict[str, tuple[str, ...]]
    best_epoch: int
    threshold: float = 0.5
    test: dict[str, float] | None = None
    output_dir: Path | None = None


@dataclass
class SavedRun:
    """Frozen training artefacts needed to score new samples."""

    model: SampleGraphClassifier
    config: TrainConfig
    gene_universe: GeneUniverse
    threshold: float
    epoch: int
    path: Path


def resolve_device(device: str) -> torch.device:
    choice = device.strip().lower()
    if choice in {"cpu", "cpu:0"}:
        return torch.device("cpu")
    if choice in {"auto", "cuda", "gpu", "gpu1", "cuda:0", "cuda:1"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if choice != "auto":
            raise RuntimeError(f"CUDA was requested ({device}) but is not available")
        return torch.device("cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_score(metrics: dict[str, float]) -> float:
    auprc = metrics.get("auprc", float("nan"))
    if auprc == auprc:
        return float(auprc)
    return float(metrics.get("f1", float("-inf")))


def run_epoch(
    model: SampleGraphClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    pos_weight: torch.Tensor | None = None,
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
        weight = None if pos_weight is None else pos_weight.to(device=device, dtype=logit.dtype)
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
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(np.sum(losses) / len(y_true))
    return metrics


@torch.no_grad()
def collect_predictions(
    model: SampleGraphClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_prob)`` for the best-checkpoint readout."""
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        logits.append(model(batch).detach().cpu())
        labels.append(batch.y.reshape(-1).detach().cpu())
    y_true = torch.cat(labels).numpy()
    y_prob = torch.sigmoid(torch.cat(logits)).numpy()
    return y_true, y_prob


def config_from_dict(payload: dict) -> TrainConfig:
    allowed = {item.name for item in dataclass_fields(TrainConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    if "cell_types" in values and values["cell_types"] is not None:
        values["cell_types"] = tuple(values["cell_types"])
    return TrainConfig(**values)


def build_classifier(num_genes: int, config: TrainConfig) -> SampleGraphClassifier:
    return SampleGraphClassifier(
        num_genes,
        hidden_dim=config.hidden_dim,
        pooling=config.pooling,
        readout=config.readout,
        encoder=config.encoder,
        num_layers=config.num_gnn_layers,
        gat_heads=config.gat_heads,
        dropout=config.dropout,
    )


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
        sampling_mode=config.sampling_mode,
        annotation_keys=annotation_keys_for(config.cell_type_level),
        cell_types=config.cell_types or None,
        training=training,
        seed=config.seed,
        n_hvg=config.n_hvg,
        cache_samples=config.cache_samples,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training)


def _item_label(item: SampleData | SampleRecord) -> str:
    return str(item.label)


def _pos_weight(train_items: Sequence[SampleData | SampleRecord], enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    n_pos = sum(1 for item in train_items if _item_label(item) == "R")
    n_neg = len(train_items) - n_pos
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def _split_payload(items: Sequence[SampleData | SampleRecord]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": str(item.sample_id),
            "patient_key": patient_key(item),
            "label": _item_label(item),
        }
        for item in items
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_history(path: Path, result: EpochResult) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["epoch", "split", "acc", "auroc", "auprc", "f1", "loss"])
        for split, metrics in (("train", result.train), ("val", result.val)):
            writer.writerow(
                [
                    result.epoch,
                    split,
                    metrics["acc"],
                    metrics.get("auroc", float("nan")),
                    metrics["auprc"],
                    metrics["f1"],
                    metrics["loss"],
                ]
            )


def _save_checkpoint(
    path: Path,
    *,
    model: SampleGraphClassifier,
    config: TrainConfig,
    gene_universe: GeneUniverse,
    epoch: int,
    metrics: dict[str, float],
    threshold: float | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "config": asdict(config),
        "gene_universe": list(gene_universe.names),
        "metrics": metrics,
    }
    if threshold is not None:
        payload["threshold"] = float(threshold)
        payload["threshold_strategy"] = config.threshold_strategy
    torch.save(payload, path)


def train(
    items: Sequence[SampleData | SampleRecord],
    *,
    config: TrainConfig | None = None,
    gene_universe: GeneUniverse | None = None,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> TrainResult:
    """Patient-disjoint train/val(/test), then fit the sample-graph classifier.

    Each sample keeps its own HVG list (scanpy on that sample, then cached).
    The gene universe is only a shared ID table. The R/NR cutoff is chosen on
    val after the best checkpoint; epoch logs still use ``threshold``.
    """
    config = config or TrainConfig()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if log:
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        print(f"device={device} ({gpu})")
    if config.test_fraction > 0:
        train_items, val_items, test_items = split_by_patient(
            items,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
    else:
        train_items, val_items = split_by_patient(
            items, val_fraction=config.val_fraction, seed=config.seed
        )
        test_items = []

    if gene_universe is None:
        gene_universe = build_gene_universe(list(train_items) + list(val_items) + list(test_items))

    out: Path | None = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "config.json", asdict(config))
        _write_json(
            out / "split.json",
            {
                "train": _split_payload(train_items),
                "val": _split_payload(val_items),
                "test": _split_payload(test_items),
            },
        )
        (out / "gene_universe.txt").write_text("\n".join(gene_universe.names) + "\n")

    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    train_loader = _loader(
        train_items, sampler=sampler, gene_universe=gene_universe, config=config, training=True
    )
    val_loader = _loader(
        val_items, sampler=sampler, gene_universe=gene_universe, config=config, training=False
    )
    model = build_classifier(len(gene_universe), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos_weight = _pos_weight(train_items, config.use_pos_weight)
    history: list[EpochResult] = []
    best_epoch = 0
    best_score = float("-inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_metrics: dict[str, float] = {}

    for epoch in range(config.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            threshold=config.threshold,
            pos_weight=pos_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            threshold=config.threshold,
        )
        result = EpochResult(epoch=epoch, train=train_metrics, val=val_metrics)
        history.append(result)
        if log:
            print(f"epoch {epoch} {format_metrics('train', train_metrics)}")
            print(f"epoch {epoch} {format_metrics('val', val_metrics)}")
        if out is not None:
            _append_history(out / "history.csv", result)
            _save_checkpoint(
                out / "last.pt",
                model=model,
                config=config,
                gene_universe=gene_universe,
                epoch=epoch,
                metrics=val_metrics,
            )
        score = _checkpoint_score(val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    hvg_by_sample = _collect_hvgs(train_loader, val_loader)

    def _split_predictions(split_items: Sequence[SampleData | SampleRecord]):
        if not split_items:
            return None
        loader = _loader(
            split_items, sampler=sampler, gene_universe=gene_universe, config=config, training=False
        )
        hvg_by_sample.update(loader.dataset.frozen_hvgs())
        return collect_predictions(model, loader, device=device)

    val_pred = _split_predictions(val_items)
    if val_pred is not None:
        chosen_threshold = select_threshold(
            val_pred[0], val_pred[1], strategy=config.threshold_strategy, fixed=config.threshold
        )
    else:
        chosen_threshold = float(config.threshold)
    if log:
        print(f"threshold={chosen_threshold:.4f} strategy={config.threshold_strategy}")

    test_pred = _split_predictions(test_items)
    test_metrics = None
    if test_pred is not None:
        test_metrics = classification_metrics(test_pred[0], test_pred[1], threshold=chosen_threshold)
        if log:
            print(format_metrics("test", test_metrics))

    if out is not None:
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if val_pred is not None:
            predictions["val"] = val_pred
        train_pred = _split_predictions(train_items)
        if train_pred is not None:
            predictions["train"] = train_pred
        if test_pred is not None:
            predictions["test"] = test_pred
        _write_json(out / "hvgs.json", {key: list(names) for key, names in sorted(hvg_by_sample.items())})
        _save_checkpoint(
            out / "best.pt",
            model=model,
            config=config,
            gene_universe=gene_universe,
            epoch=best_epoch,
            metrics=best_val_metrics,
            threshold=chosen_threshold,
        )
        write_run_report(
            out,
            history=history,
            predictions=predictions,
            threshold=chosen_threshold,
            threshold_strategy=config.threshold_strategy,
        )
    return TrainResult(
        history=history,
        model=model,
        gene_universe=gene_universe,
        hvg_by_sample=hvg_by_sample,
        best_epoch=best_epoch,
        threshold=chosen_threshold,
        test=test_metrics,
        output_dir=out,
    )


def _collect_hvgs(*loaders: DataLoader) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for loader in loaders:
        mapping.update(loader.dataset.frozen_hvgs())
    return mapping


def fit(
    samples: Sequence[SampleData],
    gene_universe: GeneUniverse,
    *,
    config: TrainConfig | None = None,
    log: bool = True,
) -> list[EpochResult]:
    """In-memory training used by tests. Writes no files."""
    return train(samples, config=config, gene_universe=gene_universe, log=log).history


def _resolve_checkpoint_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "best.pt"
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def _load_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"unrecognised checkpoint: {path}")
    return payload


def load_checkpoint(path: str | Path, *, device: str | None = None) -> SavedRun:
    """Rebuild the classifier, gene universe, and R/NR cutoff from a run."""
    ckpt_path = _resolve_checkpoint_path(path)
    payload = _load_payload(ckpt_path)
    config = config_from_dict(payload.get("config") or {})
    gene_universe = GeneUniverse(payload["gene_universe"])
    threshold = float(payload["threshold"]) if "threshold" in payload else float(config.threshold)
    config.threshold = threshold
    resolved = resolve_device(device or config.device)
    model = build_classifier(len(gene_universe), config)
    model.load_state_dict(payload["model"])
    model.to(resolved)
    model.eval()
    return SavedRun(
        model=model,
        config=config,
        gene_universe=gene_universe,
        threshold=threshold,
        epoch=int(payload.get("epoch", 0)),
        path=ckpt_path,
    )


def predict(
    items: Sequence[SampleData | SampleRecord],
    run: SavedRun | str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    log: bool = True,
) -> dict[str, object]:
    """Score new samples with a saved run. Graphs are rebuilt, weights are not.

    Each sample uses its own HVGs, mapped onto the frozen training gene universe.
    The val-tuned cutoff decides R vs NR.
    """
    saved = run if isinstance(run, SavedRun) else load_checkpoint(run, device=device)
    items = list(items)
    if not items:
        raise ValueError("predict requires at least one sample")
    config = saved.config
    resolved = resolve_device(device or config.device)
    saved.model.to(resolved)
    loader = _loader(
        items,
        sampler=CellSampler(num_cells=config.num_cells, seed=config.seed),
        gene_universe=saved.gene_universe,
        config=config,
        training=False,
    )
    y_true, y_prob = collect_predictions(saved.model, loader, device=resolved)
    y_pred = (y_prob >= saved.threshold).astype(np.int64)
    dataset = loader.dataset
    rows = []
    labelled_true: list[float] = []
    labelled_prob: list[float] = []
    for view, truth, prob, pred in zip(dataset.views, y_true, y_prob, y_pred, strict=True):
        item = dataset.items[view.item_index]
        label = _item_label(item)
        rows.append(
            {
                "sample_id": str(item.sample_id),
                "patient_key": patient_key(item),
                "cell_type": view.cell_type or "",
                "label": label,
                "prob_R": float(prob),
                "pred": "R" if int(pred) == 1 else "NR",
            }
        )
        if label in ("R", "NR"):
            labelled_true.append(float(truth))
            labelled_prob.append(float(prob))
    metrics = None
    if labelled_true:
        metrics = classification_metrics(
            np.asarray(labelled_true),
            np.asarray(labelled_prob),
            threshold=saved.threshold,
        )
    out: Path | None = None
    if output_dir is not None:
        out = Path(output_dir)
        write_predictions(
            out,
            rows=rows,
            threshold=saved.threshold,
            threshold_strategy=config.threshold_strategy,
            checkpoint=str(saved.path),
            metrics=metrics,
        )
    if log:
        print(f"samples={len(rows)} threshold={saved.threshold:.4f} checkpoint={saved.path}")
        if metrics is not None:
            print(format_metrics("predict", metrics))
    return {
        "rows": rows,
        "metrics": metrics,
        "threshold": saved.threshold,
        "output_dir": out,
        "run": saved,
    }
