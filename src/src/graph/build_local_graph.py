"""Construct the minimal Stage 1 cell-gene heterogeneous graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict

import numpy as np
import torch
from scipy import sparse
from torch_geometric.data import HeteroData

from src.data.dataset import SampleData


class GraphSummary(TypedDict):
    num_cells: int
    num_genes: int
    num_expression_edges: int
    expression_density: float
    min_edge_weight: float
    max_edge_weight: float
    tensor_bytes: int


@dataclass(frozen=True)
class GeneUniverse:
    """Stable global gene identities shared by every local graph."""

    names: tuple[str, ...]

    def __init__(self, names: Sequence[str]) -> None:
        clean = tuple(str(name) for name in names)
        if not clean:
            raise ValueError("gene universe cannot be empty")
        if len(set(clean)) != len(clean):
            raise ValueError("gene universe names must be unique")
        object.__setattr__(self, "names", clean)
        object.__setattr__(self, "_index", {name: i for i, name in enumerate(clean)})

    def index(self, name: str) -> int | None:
        return self._index.get(name)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.names)


def _selected_expression(sample: SampleData, cell_indices: np.ndarray, columns: np.ndarray):
    matrix = sample.X[cell_indices][:, columns]
    if sparse.issparse(matrix):
        return matrix.tocoo()
    return sparse.coo_matrix(np.asarray(matrix))


def build_local_graph(
    sample: SampleData,
    cell_indices: Sequence[int] | np.ndarray,
    gene_universe: GeneUniverse,
) -> HeteroData:
    """Build cell->gene expression edges and their exact reverse.

    Only universe genes with at least one non-zero value in sampled cells become
    local nodes. ``gene.global_id`` preserves identity across graphs. Zero-valued
    and non-finite entries are excluded; negative processed values are rejected.
    """
    cells = np.asarray(cell_indices, dtype=np.int64)
    if cells.ndim != 1 or len(np.unique(cells)) != len(cells):
        raise ValueError("cell_indices must be a one-dimensional set of unique indices")
    if len(cells) == 0:
        raise ValueError("cannot construct a graph with no sampled cells")
    if cells.min() < 0 or cells.max() >= sample.X.shape[0]:
        raise IndexError("cell index outside sample expression matrix")

    sample_columns: list[int] = []
    global_ids: list[int] = []
    for column, name in enumerate(sample.gene_names):
        global_id = gene_universe.index(str(name))
        if global_id is not None:
            sample_columns.append(column)
            global_ids.append(global_id)
    if not sample_columns:
        raise ValueError("sample and gene universe have no genes in common")

    expression = _selected_expression(sample, cells, np.asarray(sample_columns))
    expression.sum_duplicates()
    finite = np.isfinite(expression.data)
    if np.any(expression.data[finite] < 0):
        raise ValueError("processed expression edge weights must be non-negative")
    keep = finite & (expression.data > 0)
    rows = expression.row[keep]
    universe_columns = np.asarray(global_ids, dtype=np.int64)[expression.col[keep]]
    weights = expression.data[keep].astype(np.float32, copy=False)

    expressed_global_ids = np.unique(universe_columns)
    if len(expressed_global_ids) == 0:
        raise ValueError("sampled cells have no positive expression in the gene universe")
    global_to_local = {int(gid): local for local, gid in enumerate(expressed_global_ids)}
    local_gene = np.fromiter(
        (global_to_local[int(gid)] for gid in universe_columns), dtype=np.int64
    )
    edge_index = torch.tensor(np.stack((rows, local_gene)), dtype=torch.long)
    edge_weight = torch.from_numpy(weights)

    graph = HeteroData()
    graph["cell"].num_nodes = len(cells)
    graph["cell"].source_index = torch.from_numpy(cells.copy())
    graph["gene"].num_nodes = len(expressed_global_ids)
    graph["gene"].global_id = torch.from_numpy(expressed_global_ids.copy())
    graph["cell", "expresses", "gene"].edge_index = edge_index
    graph["cell", "expresses", "gene"].edge_weight = edge_weight
    graph["gene", "expressed_by", "cell"].edge_index = edge_index.flip(0)
    graph["gene", "expressed_by", "cell"].edge_weight = edge_weight.clone()
    graph.sample_id = str(sample.sample_id)
    graph.patient_id = str(sample.patient_id)
    graph.y = torch.tensor([int(sample.label)], dtype=torch.float32)
    return graph


def summarize_local_graph(graph: HeteroData) -> GraphSummary:
    """Return Stage 1 structural and memory diagnostics for one local graph."""
    relation = graph["cell", "expresses", "gene"]
    num_cells = int(graph["cell"].num_nodes)
    num_genes = int(graph["gene"].num_nodes)
    num_edges = int(relation.edge_index.shape[1])
    weights = relation.edge_weight
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for store in graph.stores
        for value in store.values()
        if isinstance(value, torch.Tensor)
    )
    return GraphSummary(
        num_cells=num_cells,
        num_genes=num_genes,
        num_expression_edges=num_edges,
        expression_density=(num_edges / (num_cells * num_genes)),
        min_edge_weight=float(weights.min().item()),
        max_edge_weight=float(weights.max().item()),
        tensor_bytes=int(tensor_bytes),
    )
