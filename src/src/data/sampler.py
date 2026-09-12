"""Cell sampling for stochastic train views and reproducible validation views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data_loader import CELL_ANNOTATION_KEYS

SAMPLING_MODES = ("random", "proportional", "by_cell_type", "whole_sample")


@dataclass(frozen=True)
class CellSampler:
    """Sample cells for a local graph.

    Random and proportional draws use ``num_cells``. ``whole_sample`` and
    per-type graphs (``sample_by_cell_type`` / ``sample_all_cell_types``) keep
    every requested cell; ``num_cells`` is not applied. Pass
    ``annotation_keys=MAIN_CELL_TYPE_KEYS`` to group by Immune_All_High labels.
    """

    num_cells: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")

    def sample_all(self, available_cells: int) -> np.ndarray:
        """Return every cell index. ``num_cells`` is not applied."""
        if available_cells <= 0:
            raise ValueError("available_cells must be positive")
        return np.arange(available_cells, dtype=np.int64)

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

    def select_cells(
        self,
        available_cells: int,
        *,
        mode: str,
        training: bool,
        view_index: int = 0,
        rng: np.random.Generator | None = None,
        cell_annotation: Mapping[str, Sequence[object]] | None = None,
        cell_type: str | None = None,
        annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
    ) -> np.ndarray:
        """Choose cell indices for one local graph. Graph construction is separate."""
        if mode == "whole_sample":
            return self.sample_all(available_cells)
        kwargs = {
            "training": training,
            "view_index": view_index,
            "rng": rng,
        }
        if mode == "random":
            return self.sample(available_cells, **kwargs)
        if mode == "proportional":
            if cell_annotation is None:
                raise ValueError("proportional sampling requires cell_annotation")
            return self.cell_type_proportional_sample(
                available_cells,
                cell_annotation=cell_annotation,
                annotation_keys=annotation_keys,
                **kwargs,
            )
        if mode == "by_cell_type":
            if cell_type is None:
                raise ValueError("by_cell_type requires a cell_type")
            if cell_annotation is None:
                raise ValueError("by_cell_type requires cell_annotation")
            return self.sample_by_cell_type(
                available_cells,
                cell_type=cell_type,
                cell_annotation=cell_annotation,
                annotation_keys=annotation_keys,
                **kwargs,
            )
        raise ValueError(
            "sampling mode must be 'random', 'proportional', 'by_cell_type', or 'whole_sample'"
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
        labels = _label_column(
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
        labels = _label_column(
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

        Allocates ``num_cells`` across types in proportion to how often each
        type appears, then draws uniformly within type.
        """
        take = min(self.num_cells, available_cells)
        generator = self._rng(training=training, view_index=view_index, rng=rng)
        labels = _label_column(
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
            chosen.append(_sample_group(members, int(quota), generator))
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


def select_cells(
    sample: object,
    *,
    mode: str,
    sampler: CellSampler,
    training: bool,
    view_index: int = 0,
    rng: np.random.Generator | None = None,
    cell_type: str | None = None,
    annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
) -> np.ndarray:
    """Return ``selected_cells`` for ``mode``. Callers then pass them to ``build_local_graph``."""
    return sampler.select_cells(
        int(sample.num_cells),
        mode=mode,
        training=training,
        view_index=view_index,
        rng=rng,
        cell_annotation=getattr(sample, "cell_annotation", None),
        cell_type=cell_type,
        annotation_keys=annotation_keys,
    )


def _label_column(
    cell_annotation: Mapping[str, Sequence[object]],
    available_cells: int,
    *,
    annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
) -> np.ndarray:
    label_key = annotation_keys[0]
    if label_key not in cell_annotation:
        raise ValueError(f"cell-type sampling requires {label_key!r}")
    labels = np.asarray(cell_annotation[label_key])
    if labels.shape[0] != available_cells:
        raise ValueError("cell annotation length must match available_cells")
    return labels


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
    quota: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if quota >= len(members):
        return members.astype(np.int64, copy=False)
    return generator.choice(members, size=quota, replace=False).astype(np.int64, copy=False)
