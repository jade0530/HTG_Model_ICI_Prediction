"""Homogeneous cell k-NN graphs for the GCN baseline.

Each sample becomes one undirected cell graph: nodes are cells, features are
HVG expression, and edges connect expression-space neighbours. There are no
gene nodes and no cell-gene edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import sparse
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected

from src.data.sampler import CellSampler
from src.train.dataset import SampleRecord


def load_cell_expression(
    sample_path: str | Path,
    selected_genes: Sequence[str],
    cell_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return cells x genes expression, 0-filling genes absent from the sample."""
    import anndata as ad

    genes = tuple(str(name) for name in selected_genes)
    if not genes:
        raise ValueError("selected_genes must be non-empty")
    adata = ad.read_h5ad(sample_path)
    n_cells = int(adata.n_obs)
    if cell_indices is None:
        cells = np.arange(n_cells, dtype=np.int64)
    else:
        cells = np.asarray(cell_indices, dtype=np.int64)
        if cells.min() < 0 or cells.max() >= n_cells:
            raise IndexError("cell index outside sample expression matrix")
    index = {str(name): column for column, name in enumerate(adata.var_names)}
    features = np.zeros((len(cells), len(genes)), dtype=np.float32)
    columns = [index[name] for name in genes if name in index]
    positions = [i for i, name in enumerate(genes) if name in index]
    if not columns:
        return features
    matrix = adata.X[cells][:, columns]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    features[:, np.asarray(positions, dtype=np.int64)] = np.asarray(matrix, dtype=np.float32)
    return features


def _knn_edges(features: np.ndarray, k_neighbors: int) -> torch.Tensor:
    n_cells = int(features.shape[0])
    k = min(int(k_neighbors), n_cells - 1)
    if n_cells < 2 or k < 1:
        return torch.zeros((2, 0), dtype=torch.long)
    norms = np.linalg.norm(features, axis=1)
    usable = features.copy()
    if np.all(norms < 1e-8):
        return torch.zeros((2, 0), dtype=torch.long)
    usable[norms < 1e-8] = 1e-6
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    knn.fit(usable)
    _, neighbours = knn.kneighbors(usable)
    src = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    dst = neighbours[:, 1:].reshape(-1).astype(np.int64, copy=False)
    return torch.tensor(np.stack((src, dst)), dtype=torch.long)


def build_cell_knn_graph(
    record: SampleRecord,
    selected_genes: Sequence[str],
    *,
    sampler: CellSampler,
    k_neighbors: int,
    training: bool,
    view_index: int = 0,
    gene_mean: np.ndarray | None = None,
    gene_std: np.ndarray | None = None,
) -> Data:
    """One homogeneous cell graph with a sample-level R/NR label."""
    import anndata as ad

    adata = ad.read_h5ad(record.h5ad_path, backed="r")
    try:
        n_cells = int(adata.n_obs)
    finally:
        if adata.isbacked:
            adata.file.close()
    cells = sampler.sample(n_cells, training=training, view_index=view_index)
    features = load_cell_expression(record.h5ad_path, selected_genes, cells)
    if gene_mean is not None and gene_std is not None:
        features = (features - gene_mean) / gene_std
    edge_index = _knn_edges(features, k_neighbors)
    if edge_index.numel():
        edge_index = to_undirected(edge_index, num_nodes=features.shape[0])
    edge_index, _ = add_self_loops(edge_index, num_nodes=features.shape[0])
    graph = Data(
        x=torch.from_numpy(np.asarray(features, dtype=np.float32)),
        edge_index=edge_index,
        y=torch.tensor([1.0 if record.label == "R" else 0.0], dtype=torch.float32),
        num_nodes=int(features.shape[0]),
    )
    graph.sample_id = str(record.sample_id)
    graph.patient_id = str(record.patient_id)
    return graph
