"""Sample-level mean-expression vectors for the MLP baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from src.train.dataset import SampleRecord


def load_sample_features(
    sample_path: str | Path,
    selected_genes: Sequence[str],
) -> np.ndarray:
    """Turn one .h5ad sample into a fixed-length mean-expression vector.

    Each sample becomes ``[gene_1_mean, ..., gene_N_mean]`` in the same gene
    order. Genes absent from the sample are filled with 0 so val/test panels
    can reuse the training HVG list.
    """
    import anndata as ad

    genes = tuple(str(name) for name in selected_genes)
    if not genes:
        raise ValueError("selected_genes must be non-empty")
    adata = ad.read_h5ad(sample_path)
    index = {str(name): column for column, name in enumerate(adata.var_names)}
    columns = [index[name] for name in genes if name in index]
    positions = [i for i, name in enumerate(genes) if name in index]
    features = np.zeros(len(genes), dtype=np.float32)
    if not columns:
        return features
    matrix = adata.X[:, columns]
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
    else:
        means = np.asarray(matrix, dtype=np.float64).mean(axis=0)
    features[np.asarray(positions, dtype=np.int64)] = means.astype(np.float32, copy=False)
    return features


def feature_matrix(
    records: Sequence[SampleRecord],
    selected_genes: Sequence[str],
) -> np.ndarray:
    """Stack one mean-expression row per sample."""
    rows = [load_sample_features(record.h5ad_path, selected_genes) for record in records]
    return np.stack(rows, axis=0)


def binary_labels(records: Sequence[SampleRecord]) -> np.ndarray:
    """R -> 1, NR -> 0."""
    labels = []
    for record in records:
        if record.label == "R":
            labels.append(1.0)
        elif record.label == "NR":
            labels.append(0.0)
        else:
            raise ValueError(f"unsupported label {record.label!r} for {record.sample_id}")
    return np.asarray(labels, dtype=np.float32)
