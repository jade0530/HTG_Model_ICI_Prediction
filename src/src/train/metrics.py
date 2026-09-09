"""Binary classification metrics for sample-level ICI response."""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

METRIC_KEYS = ("acc", "auroc", "auprc", "f1", "loss")
ThresholdStrategy = Literal["max_f1", "youden", "fixed"]


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


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    strategy: ThresholdStrategy = "max_f1",
    fixed: float = 0.5,
) -> float:
    """Choose an R/NR cutoff from validation scores.

    ``max_f1`` maximises F1. ``youden`` maximises TPR - FPR. ``fixed`` keeps
    ``fixed`` (default 0.5). AUROC/AUPRC do not use this cutoff.
    """
    if strategy == "fixed":
        return float(fixed)
    y_int = np.asarray(y_true).reshape(-1).astype(np.int64)
    scores = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_int.size == 0 or len(np.unique(y_int)) < 2:
        return float(fixed)
    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_int, scores)
        idx = int(np.nanargmax(tpr - fpr))
        chosen = thresholds[idx]
        if not np.isfinite(chosen):
            return float(fixed)
        return float(np.clip(chosen, 0.0, 1.0))
    if strategy != "max_f1":
        raise ValueError("strategy must be 'max_f1', 'youden', or 'fixed'")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_score = -1.0
    best_thr = float(fixed)
    for thr in candidates:
        f1 = f1_score(y_int, (scores >= thr).astype(np.int64), zero_division=0)
        if f1 > best_score:
            best_score = float(f1)
            best_thr = float(thr)
    return best_thr


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
