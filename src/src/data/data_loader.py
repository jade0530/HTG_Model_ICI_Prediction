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
CELL_ANNOTATION_KEYS = ("predicted_labels", "conf_score")
DEFAULT_N_HVG = 3000


@runtime_checkable
class SampleData(Protocol):
    sample_id: str
    patient_id: str
    X: ExpressionMatrix
    gene_names: Sequence[str]
    hvg_names: Sequence[str]
    cell_annotation: Mapping[str, Sequence[object]]
    clinical_metadata: Mapping[str, Sequence[object]]
    label: str


@dataclass(frozen=True)
class AnnDataSample:
    """A validated, sample-local view used by graph construction."""

    sample_id: str
    patient_id: str
    X: ExpressionMatrix
    gene_names: tuple[str, ...]
    hvg_names: tuple[str, ...]
    cell_annotation: Mapping[str, Sequence[object]]
    clinical_metadata: Mapping[str, Sequence[object]]
    label: str
    n_hvg: int = DEFAULT_N_HVG

    def __post_init__(self) -> None:
        if self.label not in ("NR", "R"):
            raise ValueError("label must be binary: 'NR' or 'R'")
        if self.n_hvg <= 0:
            raise ValueError("n_hvg must be positive")
        if self.X.ndim != 2:
            raise ValueError("X must be a two-dimensional cells x genes matrix")
        n_cells, n_genes = self.X.shape
        if n_genes != len(self.gene_names):
            raise ValueError("gene_names length must match the number of X columns")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene_names must be unique within a sample")
        self._validate_column_map("cell_annotation", self.cell_annotation, n_cells)
        self._validate_column_map("clinical_metadata", self.clinical_metadata, n_cells)
        if self.cell_annotation:
            missing = [key for key in CELL_ANNOTATION_KEYS if key not in self.cell_annotation]
            if missing:
                raise ValueError(
                    "cell_annotation must contain 'predicted_labels' and 'conf_score'; "
                    f"missing {missing}"
                )
        gene_set = set(self.gene_names)
        if len(set(self.hvg_names)) != len(self.hvg_names):
            raise ValueError("hvg_names must be unique within a sample")
        unknown = [name for name in self.hvg_names if name not in gene_set]
        if unknown:
            raise ValueError(f"hvg_names contains genes absent from gene_names: {unknown[:5]}")
        if len(self.hvg_names) > self.n_hvg:
            raise ValueError("hvg_names cannot be longer than n_hvg")

    @staticmethod
    def _validate_column_map(
        name: str,
        columns: Mapping[str, Sequence[object]],
        n_cells: int,
    ) -> None:
        for key, values in columns.items():
            if len(values) != n_cells:
                raise ValueError(f"{name} {key!r} does not match X rows")

    @property
    def num_cells(self) -> int:
        return int(self.X.shape[0])


def _copy_obs_column(adata: object, key: str) -> np.ndarray:
    if key not in adata.obs.columns:
        raise ValueError(f"AnnData obs is missing required column {key!r}")
    return np.asarray(adata.obs[key]).copy()


def _hvgs_from_var(adata: object, n_hvg: int) -> tuple[str, ...]:
    if "highly_variable" not in adata.var.columns:
        raise ValueError("adata.var is missing 'highly_variable'")
    mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    names = np.asarray(adata.var_names).astype(str)
    selected = names[mask]
    if "highly_variable_rank" in adata.var.columns:
        ranks = np.asarray(adata.var["highly_variable_rank"], dtype=np.float64)[mask]
        selected = selected[np.argsort(ranks, kind="stable")]
    if selected.size > n_hvg:
        selected = selected[:n_hvg]
    return tuple(selected.tolist())


def select_sample_hvgs(adata: object, *, n_hvg: int = DEFAULT_N_HVG) -> tuple[str, ...]:
    """Select HVGs with scanpy on log-normalised ``adata.X``.

    Run this on the integrated AnnData *before* splitting into samples. It writes
    ``adata.var['highly_variable']``, so sample subsets reuse the same gene set.
    """
    import scanpy as sc

    if n_hvg <= 0:
        raise ValueError("n_hvg must be positive")
    sc.pp.highly_variable_genes(adata, n_top_genes=int(n_hvg), flavor="seurat")
    return _hvgs_from_var(adata, n_hvg)


def _resolve_hvg_names(
    adata: object,
    gene_names: Sequence[str],
    *,
    n_hvg: int,
    hvg_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if hvg_names is not None:
        selected = tuple(str(name) for name in hvg_names)
    elif "highly_variable" in getattr(adata.var, "columns", []):
        selected = _hvgs_from_var(adata, n_hvg)
    else:
        selected = select_sample_hvgs(adata, n_hvg=n_hvg)
    gene_set = set(gene_names)
    return tuple(name for name in selected if name in gene_set)


def sample_from_anndata(
    adata: object,
    *,
    sample_id: str,
    patient_id: str,
    label: str,
    load_normalised_expression: bool = False,
    load_cell_annotation: bool = False,
    load_clinical_metadata: bool = False,
    n_hvg: int = DEFAULT_N_HVG,
    hvg_names: Sequence[str] | None = None,
) -> AnnDataSample:
    """Create the Stage 1 contract from a sample-filtered AnnData object.

    ``adata`` must contain exactly one biological sample. Expression is taken from
    ``adata.X`` or from ``adata.layers['counts']`` and is not normalised here.

    ``predicted_labels`` and ``conf_score`` are stored on ``cell_annotation``.
    Every other ``adata.obs`` column is stored on ``clinical_metadata``.

    HVGs should be computed once on the integrated AnnData with
    ``select_sample_hvgs`` before splitting. This function reuses
    ``adata.var['highly_variable']`` when present; otherwise it accepts
    ``hvg_names`` or runs scanpy on the object it is given.
    """
    matrix = adata.X if load_normalised_expression else adata.layers["counts"]
    gene_names = tuple(str(name) for name in adata.var_names)
    cell_annotation: dict[str, np.ndarray] = {}
    clinical_metadata: dict[str, np.ndarray] = {}
    if load_cell_annotation:
        cell_annotation = {key: _copy_obs_column(adata, key) for key in CELL_ANNOTATION_KEYS}
    if load_clinical_metadata:
        clinical_metadata = {
            key: np.asarray(adata.obs[key]).copy()
            for key in adata.obs.columns
            if key not in CELL_ANNOTATION_KEYS
        }
    hvg_names = _resolve_hvg_names(adata, gene_names, n_hvg=n_hvg, hvg_names=hvg_names)
    return AnnDataSample(
        sample_id=str(sample_id),
        patient_id=str(patient_id),
        X=matrix,
        gene_names=gene_names,
        hvg_names=hvg_names,
        cell_annotation=cell_annotation,
        clinical_metadata=clinical_metadata,
        label=str(label),
        n_hvg=int(n_hvg),
    )
