"""Cell sampling for stochastic train views and reproducible validation views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        """Return sorted row indices without replacement.

        Training uses the supplied stateful RNG (or a fresh RNG). Validation uses a
        seed derived from ``seed`` and ``view_index``, so each view is repeatable.
        Samples smaller than the requested size are returned whole.
        """
        if available_cells < 0:
            raise ValueError("available_cells cannot be negative")
        take = min(self.num_cells, available_cells)
        if take == available_cells:
            return np.arange(available_cells, dtype=np.int64)
        generator = rng if training and rng is not None else (
            np.random.default_rng() if training else np.random.default_rng(self.seed + view_index)
        )
        return np.sort(generator.choice(available_cells, size=take, replace=False)).astype(
            np.int64, copy=False
        )
