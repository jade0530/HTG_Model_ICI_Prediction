"""Tables and figures for one training run."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from src.train.metrics import METRIC_KEYS, classification_metrics, confusion_counts

SPLIT_ORDER = ("train", "val", "test")
LABELS = ("NR", "R")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _fmt(value: object) -> str:
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_results_tables(
    output_dir: Path,
    split_metrics: Mapping[str, Mapping[str, float]],
    *,
    reference: str = "val",
) -> None:
    """Write absolute metrics and deltas relative to the validation split."""
    splits = [name for name in SPLIT_ORDER if name in split_metrics]
    metric_keys = [key for key in METRIC_KEYS if any(key in split_metrics[name] for name in splits)]
    abs_rows = [
        [name] + [_fmt(split_metrics[name].get(key, float("nan"))) for key in metric_keys]
        for name in splits
    ]
    _write_csv(output_dir / "results.csv", ["split", *metric_keys], abs_rows)

    ref = split_metrics.get(reference, {})
    rel_rows = []
    for name in splits:
        row: list[object] = [name]
        for key in metric_keys:
            value = float(split_metrics[name].get(key, float("nan")))
            baseline = float(ref.get(key, float("nan")))
            delta = value - baseline
            row.extend([_fmt(value), _fmt(baseline), _fmt(delta)])
        rel_rows.append(row)
    columns = ["split"]
    for key in metric_keys:
        columns.extend([key, f"{key}_{reference}", f"{key}_delta"])
    _write_csv(output_dir / "results_relative.csv", columns, rel_rows)


def write_confusion_matrix(
    output_dir: Path,
    split: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    matrix = confusion_counts(y_true, y_prob, threshold=threshold)
    _write_csv(
        output_dir / f"confusion_matrix_{split}.csv",
        ["true\\pred", f"pred_{LABELS[0]}", f"pred_{LABELS[1]}"],
        [
            [f"true_{LABELS[0]}", int(matrix[0, 0]), int(matrix[0, 1])],
            [f"true_{LABELS[1]}", int(matrix[1, 0]), int(matrix[1, 1])],
        ],
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({split})")
    vmax = max(int(matrix.max()), 1)
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > vmax / 2 else "black"
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{split}.png", dpi=150)
    plt.close(fig)
    return matrix


def plot_history(history: Sequence[object], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(getattr(row, "epoch")) for row in history]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, key in zip(axes, ("loss", "acc", "auroc", "auprc")):
        ax.plot(epochs, [getattr(row, "train").get(key, float("nan")) for row in history], label="train")
        ax.plot(epochs, [getattr(row, "val").get(key, float("nan")) for row in history], label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "history_curves.png", dpi=150)
    plt.close(fig)


def plot_metrics_by_split(split_metrics: Mapping[str, Mapping[str, float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    splits = [name for name in SPLIT_ORDER if name in split_metrics]
    keys = [key for key in ("acc", "auroc", "auprc", "f1") if any(key in split_metrics[name] for name in splits)]
    x = np.arange(len(keys))
    width = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for index, split in enumerate(splits):
        values = [float(split_metrics[split].get(key, float("nan"))) for key in keys]
        ax.bar(x + (index - (len(splits) - 1) / 2) * width, values, width=width, label=split)
    ax.set_xticks(x, keys)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Best-checkpoint metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_split.png", dpi=150)
    plt.close(fig)


def plot_roc_pr(
    output_dir: Path,
    split: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    y_int = np.asarray(y_true).reshape(-1).astype(np.int64)
    if len(np.unique(y_int)) < 2:
        return
    from sklearn.metrics import precision_recall_curve, roc_curve

    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_int, y_prob)
    precision, recall, _ = precision_recall_curve(y_int, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title(f"ROC ({split})")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[1].plot(recall, precision)
    axes[1].set_title(f"PR ({split})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_pr_{split}.png", dpi=150)
    plt.close(fig)


def write_run_report(
    output_dir: Path,
    *,
    history: Sequence[object],
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Write results tables, confusion matrices, and summary figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {
        split: classification_metrics(y_true, y_prob, threshold=threshold)
        for split, (y_true, y_prob) in predictions.items()
    }
    write_results_tables(output_dir, split_metrics)
    plot_history(history, output_dir)
    plot_metrics_by_split(split_metrics, output_dir)
    for split, (y_true, y_prob) in predictions.items():
        write_confusion_matrix(output_dir, split, y_true, y_prob, threshold=threshold)
        plot_roc_pr(output_dir, split, y_true, y_prob)
    return split_metrics
