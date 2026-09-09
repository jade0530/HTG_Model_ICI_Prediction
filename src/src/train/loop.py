"""Training / evaluation loop for sample-level graphs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from src.data.data_loader import DEFAULT_N_HVG, SampleData
from src.data.sampler import CellSampler
from src.graph.build_local_graph import GeneStrategy, GeneUniverse
from src.train.dataset import (
    SamplingMode,
    SampleRecord,
    SampleGraphDataset,
    build_gene_universe,
    patient_key,
    select_train_hvgs,
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
    batch_size: int = 2
    epochs: int = 5
    lr: float = 1e-3
    val_fraction: float = 0.25
    test_fraction: float = 0.0
    n_hvg: int = DEFAULT_N_HVG
    hvg_cells_per_sample: int = 256
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
    hvg_names: tuple[str, ...]
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
    hvg_names: tuple[str, ...]
    threshold: float
    epoch: int
    path: Path


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return TrainConfig(**{key: value for key, value in payload.items() if key in allowed})


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
    hvg_names: Sequence[str] | None = None,
) -> DataLoader:
    dataset = SampleGraphDataset(
        items,
        sampler=sampler,
        gene_universe=gene_universe,
        gene_strategy=config.gene_strategy,
        sampling_mode=config.sampling_mode,
        training=training,
        seed=config.seed,
        hvg_names=hvg_names,
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
    hvg_names: Sequence[str],
    epoch: int,
    metrics: dict[str, float],
    threshold: float | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "config": asdict(config),
        "gene_universe": list(gene_universe.names),
        "hvg_names": list(hvg_names),
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
    hvg_names: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> TrainResult:
    """Patient-disjoint train/val(/test), then fit the sample-graph classifier.

    ``gene_universe`` defaults to the union of train+val+test gene names.
    HVGs are estimated on the train fold only unless ``hvg_names`` is passed.
    The R/NR cutoff is chosen on val (``threshold_strategy``) after the best
    checkpoint; epoch logs still use ``threshold`` (default 0.5). AUROC/AUPRC
    ignore the cutoff.
    """
    config = config or TrainConfig()
    seed_everything(config.seed)
    device = resolve_device(config.device)
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
    if hvg_names is None and config.gene_strategy == "hvg":
        hvg_names = select_train_hvgs(
            train_items,
            n_hvg=config.n_hvg,
            cells_per_sample=config.hvg_cells_per_sample,
            seed=config.seed,
        )
    hvg_tuple = tuple(hvg_names) if hvg_names is not None else ()

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
        if hvg_tuple:
            (out / "hvg_names.txt").write_text("\n".join(hvg_tuple) + "\n")

    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    train_loader = _loader(
        train_items,
        sampler=sampler,
        gene_universe=gene_universe,
        config=config,
        training=True,
        hvg_names=hvg_tuple or None,
    )
    val_loader = _loader(
        val_items,
        sampler=sampler,
        gene_universe=gene_universe,
        config=config,
        training=False,
        hvg_names=hvg_tuple or None,
    )
    model = build_classifier(len(gene_universe), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos_weight = _pos_weight(train_items, config.use_pos_weight)
    history: list[EpochResult] = []
    best_epoch = 0
    best_score = float("-inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

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
                hvg_names=hvg_tuple,
                epoch=epoch,
                metrics=val_metrics,
            )
        score = _checkpoint_score(val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if out is not None:
                _save_checkpoint(
                    out / "best.pt",
                    model=model,
                    config=config,
                    gene_universe=gene_universe,
                    hvg_names=hvg_tuple,
                    epoch=epoch,
                    metrics=val_metrics,
                )

    model.load_state_dict(best_state)

    def _split_predictions(split_items: Sequence[SampleData | SampleRecord]) -> tuple[np.ndarray, np.ndarray]:
        loader = _loader(
            split_items,
            sampler=sampler,
            gene_universe=gene_universe,
            config=config,
            training=False,
            hvg_names=hvg_tuple or None,
        )
        return collect_predictions(model, loader, device=device)

    if val_items:
        val_true, val_prob = _split_predictions(val_items)
        chosen_threshold = select_threshold(
            val_true,
            val_prob,
            strategy=config.threshold_strategy,
            fixed=config.threshold,
        )
    else:
        val_true = val_prob = None
        chosen_threshold = float(config.threshold)
    if log:
        print(f"threshold={chosen_threshold:.4f} strategy={config.threshold_strategy}")

    test_metrics = None
    if test_items:
        test_loader = _loader(
            test_items,
            sampler=sampler,
            gene_universe=gene_universe,
            config=config,
            training=False,
            hvg_names=hvg_tuple or None,
        )
        test_metrics = run_epoch(
            model,
            test_loader,
            device=device,
            optimizer=None,
            threshold=chosen_threshold,
        )
        if log:
            print(format_metrics("test", test_metrics))
        if out is not None:
            _write_json(out / "test_metrics.json", test_metrics)
    if out is not None:
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if val_true is not None:
            predictions["val"] = (val_true, val_prob)
        for name, split_items in (("train", train_items), ("test", test_items)):
            if split_items:
                predictions[name] = _split_predictions(split_items)
        _write_json(
            out / "best.json",
            {
                "epoch": best_epoch,
                "score": best_score,
                "threshold": chosen_threshold,
                "threshold_strategy": config.threshold_strategy,
            },
        )
        _save_checkpoint(
            out / "best.pt",
            model=model,
            config=config,
            gene_universe=gene_universe,
            hvg_names=hvg_tuple,
            epoch=best_epoch,
            metrics=history[best_epoch].val if history else {},
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
        hvg_names=hvg_tuple,
        best_epoch=best_epoch,
        threshold=chosen_threshold,
        test=test_metrics,
        output_dir=out,
    )


def fit(
    samples: Sequence[SampleData],
    gene_universe: GeneUniverse,
    *,
    config: TrainConfig | None = None,
    log: bool = True,
) -> list[EpochResult]:
    """In-memory training used by tests. Writes no files."""
    return train(samples, config=config, gene_universe=gene_universe, log=log).history


def evaluate(
    model: SampleGraphClassifier,
    samples: Iterable[SampleData],
    gene_universe: GeneUniverse,
    *,
    config: TrainConfig | None = None,
    hvg_names: Sequence[str] | None = None,
) -> dict[str, float]:
    config = config or TrainConfig()
    sampler = CellSampler(num_cells=config.num_cells, seed=config.seed)
    loader = _loader(
        list(samples),
        sampler=sampler,
        gene_universe=gene_universe,
        config=config,
        training=False,
        hvg_names=hvg_names,
    )
    return run_epoch(
        model,
        loader,
        device=resolve_device(config.device),
        optimizer=None,
        threshold=config.threshold,
    )


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


def _sidecar_threshold(ckpt_path: Path) -> float | None:
    for name in ("threshold.json", "best.json"):
        side = ckpt_path.parent / name
        if not side.is_file():
            continue
        data = json.loads(side.read_text())
        if "threshold" in data:
            return float(data["threshold"])
    return None


def load_checkpoint(path: str | Path, *, device: str | None = None) -> SavedRun:
    """Rebuild the classifier, gene universe, HVGs, and R/NR cutoff from a run."""
    ckpt_path = _resolve_checkpoint_path(path)
    payload = _load_payload(ckpt_path)
    config = config_from_dict(payload.get("config") or {})
    gene_universe = GeneUniverse(payload["gene_universe"])
    hvg_names = tuple(str(name) for name in payload.get("hvg_names") or ())
    if "threshold" in payload:
        threshold = float(payload["threshold"])
    else:
        threshold = _sidecar_threshold(ckpt_path)
        if threshold is None:
            threshold = float(config.threshold)
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
        hvg_names=hvg_names,
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

    Each sample is mapped onto the frozen training gene universe and HVG list,
    then encoded with the saved GNN. The val-tuned cutoff decides R vs NR.
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
        hvg_names=saved.hvg_names or None,
    )
    y_true, y_prob = collect_predictions(saved.model, loader, device=resolved)
    y_pred = (y_prob >= saved.threshold).astype(np.int64)
    rows = []
    labelled_true: list[float] = []
    labelled_prob: list[float] = []
    for item, truth, prob, pred in zip(items, y_true, y_prob, y_pred, strict=True):
        label = _item_label(item)
        rows.append(
            {
                "sample_id": str(item.sample_id),
                "patient_key": patient_key(item),
                "label": label,
                "prob_R": float(prob),
                "pred": "R" if int(pred) == 1 else "NR",
            }
        )
        if label in ("R", "NR"):
            labelled_true.append(float(truth))
            labelled_prob.append(float(prob))
    metrics = None
    if labelled_true and len(set(labelled_true)) >= 1:
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
        print(
            f"samples={len(rows)} threshold={saved.threshold:.4f} "
            f"checkpoint={saved.path}"
        )
        if metrics is not None:
            print(format_metrics("predict", metrics))
    return {
        "rows": rows,
        "metrics": metrics,
        "threshold": saved.threshold,
        "output_dir": out,
        "run": saved,
    }
