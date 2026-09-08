"""Cell sampling for stochastic train views and reproducible validation views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data_loader import CELL_ANNOTATION_KEYS


@dataclass(frozen=True)
class CellSampler:
    """Sample cells for a local graph.

    Random and proportional draws use ``num_cells``. Per-type graphs
    (``sample_by_cell_type`` / ``sample_all_cell_types``) keep every cell of
    that type. Pass ``annotation_keys=MAIN_CELL_TYPE_KEYS`` to group by
    Immune_All_High labels.
    """

    num_cells: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")

    def sample(
        self,
        available_cells: int,
        *,
        training: bool,
        view_index: int = 0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Random subsample, ignoring cell type. Small samples are returned whole."""
        take = min(self.num_cells, available_cells)
        generator = self._rng(training=training, view_index=view_index, rng=rng)
        if take == available_cells:
            return np.arange(available_cells, dtype=np.int64)
        return np.sort(generator.choice(available_cells, size=take, replace=False)).astype(
            np.int64, copy=False
        )

    def sample_by_cell_type(
        self,
        available_cells: int,
        *,
        cell_type: str,
        training: bool,
        cell_annotation: Mapping[str, Sequence[object]],
        view_index: int = 0,
        rng: np.random.Generator | None = None,
        annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
    ) -> np.ndarray:
        """Return every cell of one label. ``num_cells`` is not applied.

        ``training`` / ``view_index`` / ``rng`` match ``sample()`` so callers
        can share a call pattern; they are unused because this path is exhaustive.
        """
        labels, _ = _annotation_columns(
            cell_annotation, available_cells, annotation_keys=annotation_keys
        )
        members = np.flatnonzero(labels.astype(str) == str(cell_type))
        if members.size == 0:
            raise ValueError(f"no cells with {annotation_keys[0]}={cell_type!r}")
        return members.astype(np.int64, copy=False)

    def sample_all_cell_types(
        self,
        available_cells: int,
        *,
        training: bool,
        cell_annotation: Mapping[str, Sequence[object]],
        view_index: int = 0,
        rng: np.random.Generator | None = None,
        annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
    ) -> dict[str, np.ndarray]:
        """Return every cell of each label, one index array per type.

        ``training`` / ``view_index`` / ``rng`` match ``sample()`` and are unused.
        """
        labels, _ = _annotation_columns(
            cell_annotation, available_cells, annotation_keys=annotation_keys
        )
        labels = labels.astype(str)
        return {
            cell_type: np.flatnonzero(labels == cell_type).astype(np.int64, copy=False)
            for cell_type in np.unique(labels)
        }

    def cell_type_proportional_sample(
        self,
        available_cells: int,
        *,
        training: bool,
        cell_annotation: Mapping[str, Sequence[object]],
        view_index: int = 0,
        rng: np.random.Generator | None = None,
        annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
    ) -> np.ndarray:
        """Mixed subsample that preserves label frequencies.

        Allocates ``num_cells`` across types, then draws within each type
        using confidence scores as weights.
        """
        take = min(self.num_cells, available_cells)
        generator = self._rng(training=training, view_index=view_index, rng=rng)
        labels, scores = _annotation_columns(
            cell_annotation, available_cells, annotation_keys=annotation_keys
        )
        if take == available_cells:
            return np.arange(available_cells, dtype=np.int64)
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        quotas = _proportional_quotas(counts, take)
        chosen = []
        for type_index, quota in enumerate(quotas):
            if quota == 0:
                continue
            members = np.flatnonzero(inverse == type_index)
            chosen.append(_sample_group(members, scores[members], int(quota), generator))
        return np.sort(np.concatenate(chosen)).astype(np.int64, copy=False)

    def _rng(
        self,
        *,
        training: bool,
        view_index: int,
        rng: np.random.Generator | None,
    ) -> np.random.Generator:
        if training:
            return rng if rng is not None else np.random.default_rng()
        return np.random.default_rng(self.seed + view_index)


def _annotation_columns(
    cell_annotation: Mapping[str, Sequence[object]],
    available_cells: int,
    *,
    annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    label_key, score_key = annotation_keys
    missing = [key for key in annotation_keys if key not in cell_annotation]
    if missing:
        raise ValueError(
            "cell-type sampling requires "
            f"{label_key!r} and {score_key!r}; missing {missing}"
        )
    labels = np.asarray(cell_annotation[label_key])
    scores = np.asarray(cell_annotation[score_key], dtype=np.float64)
    if labels.shape[0] != available_cells or scores.shape[0] != available_cells:
        raise ValueError("cell annotation length must match available_cells")
    return labels, scores


def _proportional_quotas(counts: np.ndarray, take: int) -> np.ndarray:
    raw = take * counts / counts.sum()
    quotas = np.minimum(counts, np.floor(raw)).astype(np.int64)
    leftover = take - int(quotas.sum())
    remainder = raw - np.floor(raw)
    for index in np.argsort(-remainder, kind="stable"):
        if leftover <= 0:
            break
        if quotas[index] < counts[index]:
            quotas[index] += 1
            leftover -= 1
    return quotas


def _sample_group(
    members: np.ndarray,
    scores: np.ndarray,
    quota: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if quota >= len(members):
        return members.astype(np.int64, copy=False)
    weights = np.clip(np.nan_to_num(scores, nan=0.0), 0.0, None)
    total = float(weights.sum())
    probabilities = None if total <= 0 else weights / total
    return generator.choice(members, size=quota, replace=False, p=probabilities).astype(
        np.int64, copy=False
    )
