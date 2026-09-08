"""Sample-level data contract for Stage 1.

Patient-disjoint splits belong outside this module. The contract is one
biological sample: expression, gene names, optional cell annotation, and a
binary ICI label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import sparse

from .gene_harmonise_celltypist import harmonise_and_annotate


ExpressionMatrix = np.ndarray | sparse.spmatrix
CELL_ANNOTATION_KEYS = ("predicted_labels", "conf_score")
MAIN_CELL_TYPE_KEYS = ("predicted_labels_mainCellType", "conf_scores_mainCellType")
DEFAULT_CELLTYPIST_MODEL = "Immune_All_High.pkl"
DEFAULT_N_HVG = 3000
DEFAULT_HG38_GFF = Path(
    "/Users/z5155527/Desktop/Benchmark-2025-Sep/external_validation_datasets/code/Homo_sapiens.GRCh38.115.gff3"
)
DEFAULT_HG19_GFF = Path(
    "/Users/z5155527/Desktop/ICI_Foundation_2026/gencode.v49lift37.annotation.gff3"
)
DEFAULT_CONVERTED_CELLTYPIST_MODEL = (
    Path.home() / ".celltypist" / "data" / "models" / "Immune_All_High_Ensembl.pkl"
)


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
    """Sample-local view used by graph construction."""

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
        if self.X.ndim != 2:
            raise ValueError("X must be a two-dimensional cells x genes matrix")
        n_cells, n_genes = self.X.shape
        if n_genes != len(self.gene_names) or len(set(self.gene_names)) != n_genes:
            raise ValueError("gene_names must be unique and match the number of X columns")
        self._check_columns("cell_annotation", self.cell_annotation, n_cells)
        self._check_columns("clinical_metadata", self.clinical_metadata, n_cells)
        if self.cell_annotation and any(key not in self.cell_annotation for key in CELL_ANNOTATION_KEYS):
            raise ValueError("cell_annotation must contain 'predicted_labels' and 'conf_score'")
        unknown = [name for name in self.hvg_names if name not in set(self.gene_names)]
        if unknown:
            raise ValueError(f"hvg_names contains genes absent from gene_names: {unknown[:5]}")

    @staticmethod
    def _check_columns(name: str, columns: Mapping[str, Sequence[object]], n_cells: int) -> None:
        for key, values in columns.items():
            if len(values) != n_cells:
                raise ValueError(f"{name} {key!r} does not match X rows")

    @property
    def num_cells(self) -> int:
        return int(self.X.shape[0])


def _obs_has_fine_labels(adata: object) -> bool:
    columns = adata.obs.columns
    return "predicted_labels" in columns and "conf_score" in columns


def _copy_obs(adata: object, keys: Sequence[str]) -> dict[str, np.ndarray]:
    columns = adata.obs.columns
    return {key: np.asarray(adata.obs[key]).copy() for key in keys if key in columns}


def _annotation_path(path: Path, kind: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} annotation file not found: {path}")
    return path


def _is_hg38_assembly(source_assembly: str) -> bool:
    text = str(source_assembly).strip().lower().replace("_", "").replace("-", "")
    return text in {"hg38", "grch38", "38"}


def annotate_cell_types(
    adata: object,
    *,
    model: str = DEFAULT_CELLTYPIST_MODEL,
    majority_voting: bool = False,
    source_assembly: str = "hg38",
    hg19_gtf: str | Path | None = None,
    target_hg38_gff: str | Path | None = None,
    target_v49_lift37_gtf: str | Path | None = None,
    converted_model_path: str | Path | None = None,
    counts_layer: str | None = "auto",
) -> object:
    """Run CellTypist Immune_All_High after mapping query genes to Ensembl IDs.

    Labels are copied back onto ``adata``; expression is not replaced. If
    ``predicted_labels`` / ``conf_score`` already exist, High-level labels are
    stored as ``predicted_labels_mainCellType`` / ``conf_scores_mainCellType``.
    Default assembly is hg38, which skips the lift37 GFF.
    """
    hg38_path = _annotation_path(
        Path(DEFAULT_HG38_GFF if target_hg38_gff is None else target_hg38_gff),
        "hg38",
    )
    if hg19_gtf is not None:
        hg19_path = _annotation_path(Path(hg19_gtf), "hg19/lift37")
    elif _is_hg38_assembly(source_assembly):
        hg19_path = hg38_path
    else:
        hg19_path = _annotation_path(DEFAULT_HG19_GFF, "hg19/lift37")
    if target_v49_lift37_gtf is not None:
        lift37_path = _annotation_path(Path(target_v49_lift37_gtf), "v49lift37")
    elif _is_hg38_assembly(source_assembly):
        lift37_path = None
    else:
        lift37_path = DEFAULT_HG19_GFF if DEFAULT_HG19_GFF.is_file() else None

    already_labelled = _obs_has_fine_labels(adata)
    annotated = harmonise_and_annotate(
        adata,
        source_assembly=source_assembly,
        hg19_gtf=hg19_path,
        target_hg38_gff=hg38_path,
        celltypist_model=model,
        target_v49_lift37_gtf=lift37_path,
        converted_model_path=converted_model_path or DEFAULT_CONVERTED_CELLTYPIST_MODEL,
        counts_layer=counts_layer,
        majority_voting=majority_voting,
    )
    labels = annotated.obs["predicted_labels"].astype(str).reindex(adata.obs_names)
    scores = annotated.obs["conf_score"].reindex(adata.obs_names)
    label_key, score_key = MAIN_CELL_TYPE_KEYS if already_labelled else CELL_ANNOTATION_KEYS
    adata.obs[label_key] = np.asarray(labels).astype(str)
    adata.obs[score_key] = np.asarray(scores, dtype=np.float64)
    return adata


def _hvgs_from_var(adata: object, n_hvg: int) -> tuple[str, ...]:
    mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    names = np.asarray(adata.var_names).astype(str)[mask]
    if "highly_variable_rank" in adata.var.columns:
        ranks = np.asarray(adata.var["highly_variable_rank"], dtype=np.float64)[mask]
        names = names[np.argsort(ranks, kind="stable")]
    return tuple(names[:n_hvg].tolist())


def select_sample_hvgs(adata: object, *, n_hvg: int = DEFAULT_N_HVG) -> tuple[str, ...]:
    """Select HVGs with scanpy on log-normalised ``adata.X``.

    Run on the integrated AnnData before splitting so every sample shares the
    same gene set via ``adata.var['highly_variable']``.
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
    elif "highly_variable" in adata.var.columns:
        selected = _hvgs_from_var(adata, n_hvg)
    else:
        selected = select_sample_hvgs(adata, n_hvg=n_hvg)
    gene_set = set(gene_names)
    return tuple(name for name in selected if name in gene_set)[:n_hvg]


def sample_from_anndata(
    adata: object,
    *,
    sample_id: str,
    patient_id: str,
    label: str,
    load_normalised_expression: bool = False,
    load_cell_annotation: bool = False,
    load_clinical_metadata: bool = False,
    cell_type_annotation: bool = False,
    celltypist_model: str = DEFAULT_CELLTYPIST_MODEL,
    source_assembly: str = "hg38",
    hg19_gtf: str | Path | None = None,
    target_hg38_gff: str | Path | None = None,
    n_hvg: int = DEFAULT_N_HVG,
    hvg_names: Sequence[str] | None = None,
) -> AnnDataSample:
    """Build the sample contract from a sample-filtered AnnData object.

    Expression comes from ``adata.X`` or ``adata.layers['counts']`` and is not
    normalised here. ``cell_type_annotation=True`` runs Immune_All_High after
    Ensembl mapping. Fine labels stay on ``predicted_labels`` / ``conf_score``;
    High-level labels use ``MAIN_CELL_TYPE_KEYS``.
    """
    if cell_type_annotation:
        annotate_cell_types(
            adata,
            model=celltypist_model,
            source_assembly=source_assembly,
            hg19_gtf=hg19_gtf,
            target_hg38_gff=target_hg38_gff,
        )
    gene_names = tuple(str(name) for name in adata.var_names)
    annotation_keys = CELL_ANNOTATION_KEYS + MAIN_CELL_TYPE_KEYS
    cell_annotation = _copy_obs(adata, annotation_keys) if load_cell_annotation else {}
    clinical_metadata = (
        _copy_obs(adata, [key for key in adata.obs.columns if key not in annotation_keys])
        if load_clinical_metadata
        else {}
    )
    return AnnDataSample(
        sample_id=str(sample_id),
        patient_id=str(patient_id),
        X=adata.X if load_normalised_expression else adata.layers["counts"],
        gene_names=gene_names,
        hvg_names=_resolve_hvg_names(adata, gene_names, n_hvg=n_hvg, hvg_names=hvg_names),
        cell_annotation=cell_annotation,
        clinical_metadata=clinical_metadata,
        label=str(label),
        n_hvg=int(n_hvg),
    )
