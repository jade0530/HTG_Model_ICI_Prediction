"""Shared loaders for the sample-level sklearn baselines.

Each sample is one h5ad. Features are the mean gene vector over cells (raw
``adata.X``). The train/val/test assignment comes from an HTG ``split.json``
or from sample-id lists, so the baselines reuse the same patients.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from src.data.data_loader import select_train_hvgs
from src.train.dataset import SampleRecord, collect_records
from src.train.loop import EpochResult
from src.train.metrics import classification_metrics

LABEL_TO_Y = {"NR": 0, "R": 1}


@dataclass
class FoldData:
    records: list[SampleRecord]
    X: np.ndarray
    y: np.ndarray
    sample_ids: list[str]


def load_records(
    dataset_root: Path,
    manifest: Path,
    tissue: str,
    ici_phase: str,
    *,
    require_label: bool = True,
) -> list[SampleRecord]:
    return collect_records(
        dataset_root,
        manifest,
        tissue=None if tissue.lower() == "all" else tissue,
        ici_phase=None if ici_phase.lower() == "all" else ici_phase,
        require_label=require_label,
    )


def read_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def assign_split(
    records: Sequence[SampleRecord],
    *,
    split_json: Path | None = None,
    train_ids: Path | None = None,
    val_ids: Path | None = None,
    test_ids: Path | None = None,
) -> dict[str, list[SampleRecord]]:
    """Map records into folds using sample_id. ``split.json`` wins if given."""
    by_sample = {record.sample_id: record for record in records}
    if split_json is not None:
        payload = json.loads(Path(split_json).read_text())
        return {
            name: [by_sample[row["sample_id"]] for row in payload.get(name, [])]
            for name in ("train", "val", "test")
        }
    train = [by_sample[sample_id] for sample_id in read_id_list(train_ids)]
    val = [by_sample[sample_id] for sample_id in read_id_list(val_ids)]
    test = [by_sample[sample_id] for sample_id in read_id_list(test_ids)] if test_ids is not None else []
    return {"train": train, "val": val, "test": test}


def resolve_genes(
    train_records: Sequence[SampleRecord],
    *,
    n_hvg: int,
    gene_universe: Path | None,
) -> tuple[str, ...]:
    if gene_universe is not None:
        return tuple(line.strip() for line in Path(gene_universe).read_text().splitlines() if line.strip())
    return select_train_hvgs(train_records, n_hvg=n_hvg)


def align_mean_vector(mean: np.ndarray, var_names: Sequence[str], gene_names: Sequence[str]) -> np.ndarray:
    """Map a sample's gene means onto the training gene order. Missing genes stay 0."""
    index = {str(name): i for i, name in enumerate(var_names)}
    vec = np.zeros(len(gene_names), dtype=np.float64)
    for j, name in enumerate(gene_names):
        i = index.get(name)
        if i is not None:
            vec[j] = mean[i]
    return vec


def mean_expression(record: SampleRecord, gene_names: Sequence[str]) -> np.ndarray:
    """One sample = mean of raw ``X`` over cells, aligned to ``gene_names``."""
    import anndata as ad

    adata = ad.read_h5ad(record.h5ad_path)
    if sparse.issparse(adata.X):
        mean = np.asarray(adata.X.mean(axis=0)).ravel()
    else:
        mean = np.asarray(adata.X, dtype=np.float64).mean(axis=0)
    return align_mean_vector(mean, [str(name) for name in adata.var_names], gene_names)


def feature_matrix(records: Sequence[SampleRecord], gene_names: Sequence[str]) -> np.ndarray:
    return np.stack([mean_expression(record, gene_names) for record in records], axis=0)


def align_cell_matrix(X, var_names: Sequence[str], gene_names: Sequence[str]) -> np.ndarray:
    """Map cells x genes onto the training gene order. Missing genes stay 0."""
    index = {str(name): i for i, name in enumerate(var_names)}
    cols: list[int] = []
    keep: list[int] = []
    for j, name in enumerate(gene_names):
        i = index.get(name)
        if i is not None:
            cols.append(i)
            keep.append(j)
    out = np.zeros((int(X.shape[0]), len(gene_names)), dtype=np.float32)
    if not cols:
        return out
    if sparse.issparse(X):
        sl = X[:, cols].toarray()
    else:
        sl = np.asarray(X[:, cols], dtype=np.float32)
    out[:, keep] = sl
    return out


def load_cell_features(
    record: SampleRecord,
    gene_names: Sequence[str],
    *,
    num_cells: int,
    seed: int,
    view_index: int,
) -> np.ndarray:
    """Aligned cell x gene matrix, then a fixed subsample of ``num_cells``."""
    import anndata as ad

    from src.data.sampler import CellSampler

    adata = ad.read_h5ad(record.h5ad_path)
    cells = align_cell_matrix(adata.X, [str(name) for name in adata.var_names], gene_names)
    idx = CellSampler(num_cells=num_cells, seed=seed).sample(
        cells.shape[0], training=False, view_index=view_index
    )
    return cells[idx]


def build_fold(records: Sequence[SampleRecord], gene_names: Sequence[str]) -> FoldData:
    X = feature_matrix(records, gene_names)
    y = np.array([LABEL_TO_Y[record.label] for record in records], dtype=np.int64)
    sample_ids = [record.sample_id for record in records]
    return FoldData(records=list(records), X=X, y=y, sample_ids=sample_ids)


def score_metrics(y_true: np.ndarray, scores: np.ndarray, *, threshold: float, loss: float) -> dict[str, float]:
    metrics = classification_metrics(y_true, scores, threshold=threshold)
    metrics["loss"] = float(loss)
    return metrics


def write_history_csv(path: Path, history: Sequence[EpochResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "split", "acc", "auroc", "auprc", "f1", "loss"])
        for row in history:
            for split, metrics in (("train", row.train), ("val", row.val)):
                writer.writerow(
                    [
                        row.epoch,
                        split,
                        metrics["acc"],
                        metrics.get("auroc", float("nan")),
                        metrics["auprc"],
                        metrics["f1"],
                        metrics["loss"],
                    ]
                )


def write_search_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_data_args(parser) -> None:
    from src.train.dataset import DEFAULT_MANIFEST

    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=None, help="HTG split.json with train/val/test sample_ids")
    parser.add_argument("--train-ids", type=Path, default=None, help="Text file, one sample_id per line")
    parser.add_argument("--val-ids", type=Path, default=None, help="Text file, one sample_id per line")
    parser.add_argument("--test-ids", type=Path, default=None, help="Optional text file, one sample_id per line")
    parser.add_argument("--gene-universe", type=Path, default=None, help="Optional HTG gene_universe.txt")
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-strategy", choices=("max_f1", "youden", "fixed"), default="max_f1")


def add_predict_args(parser) -> None:
    from src.train.dataset import DEFAULT_MANIFEST

    parser.add_argument("--checkpoint", type=Path, required=True, help="Run directory or model.joblib")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Folder of unseen sample h5ads")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tissue", default="all")
    parser.add_argument("--ici-phase", default="all")


def load_saved_run(checkpoint: Path) -> dict:
    from joblib import load

    path = Path(checkpoint)
    joblib_path = path / "model.joblib" if path.is_dir() else path
    bundle = load(joblib_path)
    bundle.setdefault("threshold_strategy", "max_f1")
    bundle["path"] = str(joblib_path)
    return bundle


def write_predict_report(
    output_dir: Path,
    records: Sequence[SampleRecord],
    scores: np.ndarray,
    *,
    threshold: float,
    checkpoint: str,
    threshold_strategy: str = "max_f1",
) -> dict[str, float] | None:
    from src.train.metrics import METRIC_KEYS, format_metrics
    from src.train.report import plot_roc_pr, write_confusion_matrix, write_predictions

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    rows = []
    labelled_true: list[int] = []
    labelled_scores: list[float] = []
    for record, score in zip(records, scores, strict=True):
        pred = "R" if score >= threshold else "NR"
        rows.append(
            {
                "sample_id": record.sample_id,
                "patient_key": record.patient_key,
                "cell_type": "",
                "label": record.label,
                "prob_R": float(score),
                "pred": pred,
            }
        )
        if record.label in LABEL_TO_Y:
            labelled_true.append(LABEL_TO_Y[record.label])
            labelled_scores.append(float(score))
    metrics = None
    if labelled_true:
        y_true = np.asarray(labelled_true, dtype=np.int64)
        y_prob = np.asarray(labelled_scores, dtype=np.float64)
        metrics = classification_metrics(y_true, y_prob, threshold=threshold)
        with (output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            keys = [key for key in METRIC_KEYS if key in metrics]
            writer.writerow(["split", *keys])
            writer.writerow(
                ["predict"]
                + [f"{metrics[key]:.6f}" if metrics[key] == metrics[key] else "nan" for key in keys]
            )
        write_confusion_matrix(output_dir, "predict", y_true, y_prob, threshold=threshold)
        plot_roc_pr(output_dir, "predict", y_true, y_prob, threshold=threshold)
        print(format_metrics("predict", metrics))
    write_predictions(
        output_dir,
        rows=rows,
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        checkpoint=checkpoint,
        metrics=metrics,
    )
    return metrics
