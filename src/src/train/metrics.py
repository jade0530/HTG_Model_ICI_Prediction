"""Binary classification metrics for sample-level ICI response."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

METRIC_KEYS = ("acc", "auroc", "auprc", "f1", "loss")


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return accuracy, AUROC, AUPRC, and F1 for binary labels ``{0, 1}``."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_int = y_true.astype(np.int64)
    metrics: dict[str, float] = {
        "acc": float(accuracy_score(y_int, y_pred)),
        "f1": float(f1_score(y_int, y_pred, zero_division=0)),
    }
    if len(np.unique(y_int)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_int, y_prob))
        metrics["auprc"] = float(average_precision_score(y_int, y_prob))
    return metrics


def confusion_counts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """2x2 matrix with rows true {NR, R} and columns predicted {NR, R}."""
    y_int = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(np.int64)
    return confusion_matrix(y_int, y_pred, labels=[0, 1])


def format_metrics(split: str, metrics: Mapping[str, float]) -> str:
    def _fmt(key: str) -> str:
        value = metrics.get(key, float("nan"))
        return "nan" if value != value else f"{float(value):.4f}"

    return (
        f"{split}: acc={_fmt('acc')} auroc={_fmt('auroc')} auprc={_fmt('auprc')} "
        f"f1={_fmt('f1')} loss={_fmt('loss')}"
    )
