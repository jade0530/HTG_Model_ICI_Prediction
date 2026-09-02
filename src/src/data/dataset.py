"""Sample-level data contract for Stage 1.

Splitting belongs outside this module: callers must create patient-disjoint sample
collections before wrapping them in a PyG dataset or loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import sparse


ExpressionMatrix = np.ndarray | sparse.spmatrix


@runtime_checkable
class SampleData(Protocol):
    sample_id: str
    patient_id: str
    X: ExpressionMatrix
    gene_names: Sequence[str]
    cell_metadata: Mapping[str, Sequence[object]]
    label: int


@dataclass(frozen=True)
class AnnDataSample:
    """A validated, sample-local view used by graph construction."""

    sample_id: str
    patient_id: str
    X: ExpressionMatrix
    gene_names: tuple[str, ...]
    cell_metadata: Mapping[str, Sequence[object]]
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be binary: 0 (NR) or 1 (R)")
        if self.X.ndim != 2:
            raise ValueError("X must be a two-dimensional cells x genes matrix")
        n_cells, n_genes = self.X.shape
        if n_genes != len(self.gene_names):
            raise ValueError("gene_names length must match the number of X columns")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene_names must be unique within a sample")
        for key, values in self.cell_metadata.items():
            if len(values) != n_cells:
                raise ValueError(f"cell metadata {key!r} does not match X rows")

    @property
    def num_cells(self) -> int:
        return int(self.X.shape[0])


def sample_from_anndata(
    adata: object,
    *,
    sample_id: str,
    patient_id: str,
    label: int,
    cell_metadata_keys: Sequence[str] = (),
    layer: str | None = None,
) -> AnnDataSample:
    """Create the Stage 1 contract from a sample-filtered AnnData object.

    ``adata`` must contain exactly one biological sample. Expression is taken from
    ``adata.X`` or from ``adata.layers[layer]`` and is not normalised here.
    """
    matrix = adata.X if layer is None else adata.layers[layer]  # type: ignore[attr-defined]
    gene_names = tuple(str(name) for name in adata.var_names)  # type: ignore[attr-defined]
    metadata = {
        key: np.asarray(adata.obs[key]).copy()  # type: ignore[attr-defined]
        for key in cell_metadata_keys
    }
    return AnnDataSample(
        sample_id=str(sample_id),
        patient_id=str(patient_id),
        X=matrix,
        gene_names=gene_names,
        cell_metadata=metadata,
        label=int(label),
    )
