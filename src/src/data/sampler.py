"""Cell sampling for stochastic train views and reproducible validation views."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data_loader import CELL_ANNOTATION_KEYS


@dataclass(frozen=True)
class CellSampler:
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
        """Return a random subsample of cells, ignoring cell type.

        Training uses the supplied stateful RNG (or a fresh RNG). Validation uses a
        seed derived from ``seed`` and ``view_index``, so each view is repeatable.
        Samples smaller than the requested size are returned whole.
        """
        take, generator = self._prepare(
            available_cells, training=training, view_index=view_index, rng=rng
        )
        return _sample_random(available_cells, take, generator)

    def sample_by_cell_type(
        self,
        available_cells: int,
        *,
        cell_type: str,
        training: bool,
        cell_annotation: Mapping[str, Sequence[object]],
        view_index: int = 0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return cells of one predicted label for a cell-type-specific local graph.

        Draws without replacement from cells whose ``predicted_labels`` match
        ``cell_type``, weighted by ``conf_score``. Types smaller than ``num_cells``
        are returned whole.
        """
        labels, scores = _annotation_columns(cell_annotation, available_cells)
        members = np.flatnonzero(labels.astype(str) == str(cell_type))
        if members.size == 0:
            raise ValueError(f"no cells with predicted_labels={cell_type!r}")
        take, generator = self._prepare(
            int(members.size),
            training=training,
            view_index=view_index,
            rng=rng,
            cell_type=str(cell_type),
        )
        return np.sort(_sample_group(members, scores[members], take, generator))

    def sample_all_cell_types(
        self,
        available_cells: int,
        *,
        training: bool,
        cell_annotation: Mapping[str, Sequence[object]],
        view_index: int = 0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Return one subsample per predicted label, each for its own local graph."""
        labels, _ = _annotation_columns(cell_annotation, available_cells)
        types = [str(name) for name in np.unique(labels.astype(str))]
        return {
            cell_type: self.sample_by_cell_type(
                available_cells,
                cell_type=cell_type,
                training=training,
                cell_annotation=cell_annotation,
                view_index=view_index,
                rng=rng,
            )
            for cell_type in types
        }

    def cell_type_proportional_sample(
        self,
        available_cells: int,
        *,
        training: bool,
        cell_annotation: Mapping[str, Sequence[object]],
        view_index: int = 0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return a mixed subsample that preserves cell-type frequencies.

        Allocates ``num_cells`` across ``predicted_labels`` in proportion to type
        counts, then draws within each type using ``conf_score`` as weights.
        """
        take, generator = self._prepare(
            available_cells, training=training, view_index=view_index, rng=rng
        )
        labels, scores = _annotation_columns(cell_annotation, available_cells)
        if take == available_cells:
            return np.arange(available_cells, dtype=np.int64)
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        quotas = _proportional_quotas(counts, take)
        chosen: list[np.ndarray] = []
        for type_index, quota in enumerate(quotas):
            if quota == 0:
                continue
            members = np.flatnonzero(inverse == type_index)
            chosen.append(_sample_group(members, scores[members], int(quota), generator))
        if not chosen:
            raise ValueError("cell-type sampling produced no cells")
        return np.sort(np.concatenate(chosen)).astype(np.int64, copy=False)

    def _prepare(
        self,
        available_cells: int,
        *,
        training: bool,
        view_index: int,
        rng: np.random.Generator | None,
        cell_type: str | None = None,
    ) -> tuple[int, np.random.Generator]:
        if available_cells < 0:
            raise ValueError("available_cells cannot be negative")
        take = min(self.num_cells, available_cells)
        return take, self._generator(
            training=training, view_index=view_index, rng=rng, cell_type=cell_type
        )

    def _generator(
        self,
        *,
        training: bool,
        view_index: int,
        rng: np.random.Generator | None,
        cell_type: str | None = None,
    ) -> np.random.Generator:
        if training and rng is not None:
            return rng
        if training:
            return np.random.default_rng()
        if cell_type is None:
            return np.random.default_rng(self.seed + view_index)
        return np.random.default_rng(
            [self.seed, view_index, zlib.adler32(cell_type.encode("utf-8"))]
        )


def _annotation_columns(
    cell_annotation: Mapping[str, Sequence[object]] | None,
    available_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_annotation is None:
        raise ValueError("cell-type sampling requires cell_annotation")
    missing = [key for key in CELL_ANNOTATION_KEYS if key not in cell_annotation]
    if missing:
        raise ValueError(
            "cell-type sampling requires 'predicted_labels' and 'conf_score'; "
            f"missing {missing}"
        )
    labels = np.asarray(cell_annotation["predicted_labels"])
    scores = np.asarray(cell_annotation["conf_score"], dtype=np.float64)
    if labels.shape[0] != available_cells or scores.shape[0] != available_cells:
        raise ValueError("cell annotation length must match available_cells")
    return labels, scores


def _sample_random(
    available_cells: int,
    take: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if take == 0:
        return np.arange(0, dtype=np.int64)
    if take == available_cells:
        return np.arange(available_cells, dtype=np.int64)
    return np.sort(generator.choice(available_cells, size=take, replace=False)).astype(
        np.int64, copy=False
    )


def _proportional_quotas(counts: np.ndarray, take: int) -> np.ndarray:
    total = int(counts.sum())
    take = min(int(take), total)
    if take == total:
        return counts.astype(np.int64, copy=True)
    raw = take * counts / total
    quotas = np.minimum(counts, np.floor(raw)).astype(np.int64)
    leftover = take - int(quotas.sum())
    remainder = raw - np.floor(raw)
    for index in np.argsort(-remainder, kind="stable"):
        if leftover <= 0:
            break
        if quotas[index] < counts[index]:
            quotas[index] += 1
            leftover -= 1
    if leftover > 0:
        spare = np.flatnonzero(counts - quotas)
        for index in spare:
            extra = min(leftover, int(counts[index] - quotas[index]))
            quotas[index] += extra
            leftover -= extra
            if leftover == 0:
                break
    return quotas


def _sample_group(
    members: np.ndarray,
    scores: np.ndarray,
    quota: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if quota >= len(members):
        return members.astype(np.int64, copy=False)
    weights = np.where(np.isfinite(scores), np.clip(scores, 0.0, None), 0.0)
    total = float(weights.sum())
    probabilities = None if total <= 0 else weights / total
    return generator.choice(members, size=quota, replace=False, p=probabilities).astype(
        np.int64, copy=False
    )
