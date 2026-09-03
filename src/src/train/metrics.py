"""Binary classification metrics for sample-level ICI response."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return accuracy, AUPRC, and F1 for binary labels ``{0, 1}``."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty prediction set")
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_int = y_true.astype(np.int64)
    metrics: dict[str, float] = {
        "acc": float(accuracy_score(y_int, y_pred)),
        "f1": float(f1_score(y_int, y_pred, zero_division=0)),
    }
    if len(np.unique(y_int)) < 2:
        metrics["auprc"] = float("nan")
    else:
        metrics["auprc"] = float(average_precision_score(y_int, y_prob))
    return metrics


def format_metrics(split: str, metrics: Mapping[str, float]) -> str:
    auprc = metrics["auprc"]
    auprc_text = "nan" if np.isnan(auprc) else f"{auprc:.4f}"
    return (
        f"{split}: acc={metrics['acc']:.4f} auprc={auprc_text} "
        f"f1={metrics['f1']:.4f} loss={metrics.get('loss', float('nan')):.4f}"
    )
