"""Union local cell-gene graphs from one biological sample."""

from __future__ import annotations

from typing import Sequence

import torch
from torch_geometric.data import HeteroData

CELL_TO_GENE = ("cell", "expresses", "gene")
GENE_TO_CELL = ("gene", "expressed_by", "cell")


def build_sample_global_graph(graphs: Sequence[HeteroData]) -> HeteroData:
    """Merge local graphs from one sample by sharing gene nodes.

    Cells stay in their local subgraphs (no cell-cell edges). Genes with the
    same ``global_id`` become one node. The label stays a single sample-level
    ``y``. A single local graph is returned unchanged.
    """
    graphs = list(graphs)
    if not graphs:
        raise ValueError("build_sample_global_graph requires at least one local graph")
    sample_ids = {str(graph.sample_id) for graph in graphs}
    if len(sample_ids) != 1:
        raise ValueError("build_sample_global_graph requires local graphs from one sample")
    if len(graphs) == 1:
        return graphs[0]

    unique_ids = torch.cat([graph["gene"].global_id.long() for graph in graphs]).unique(sorted=True)
    gid_map = torch.full((int(unique_ids.max().item()) + 1,), -1, dtype=torch.long)
    gid_map[unique_ids] = torch.arange(unique_ids.numel(), dtype=torch.long)

    cell_features: list[torch.Tensor] = []
    source_index: list[torch.Tensor] = []
    local_graph_index: list[torch.Tensor] = []
    edge_src: list[torch.Tensor] = []
    edge_dst: list[torch.Tensor] = []
    edge_weight: list[torch.Tensor] = []
    cell_types: list[str] = []
    cell_offset = 0

    for graph_i, graph in enumerate(graphs):
        n_cells = int(graph["cell"].num_nodes)
        if n_cells <= 0:
            raise ValueError("local graph has no cells")
        mapped_gene = gid_map[graph["gene"].global_id.long()]
        if bool((mapped_gene < 0).any()):
            raise ValueError("local graph gene.global_id is outside the merged gene table")
        src, dst = graph[CELL_TO_GENE].edge_index
        edge_src.append(src + cell_offset)
        edge_dst.append(mapped_gene[dst])
        edge_weight.append(graph[CELL_TO_GENE].edge_weight)
        features = getattr(graph["cell"], "x", None)
        if features is None:
            features = torch.ones((n_cells, 1), dtype=torch.float32)
        cell_features.append(features)
        source_index.append(graph["cell"].source_index)
        existing_local = getattr(graph["cell"], "local_graph", None)
        if existing_local is None:
            existing_local = torch.full((n_cells,), graph_i, dtype=torch.long)
        local_graph_index.append(existing_local)
        cell_types.append(str(getattr(graph, "cell_type", "") or ""))
        cell_offset += n_cells

    edge_index = torch.stack((torch.cat(edge_src), torch.cat(edge_dst)))
    weights = torch.cat(edge_weight)
    merged = HeteroData()
    merged["cell"].num_nodes = cell_offset
    merged["cell"].x = torch.cat(cell_features, dim=0)
    merged["cell"].source_index = torch.cat(source_index)
    merged["cell"].local_graph = torch.cat(local_graph_index)
    merged["gene"].num_nodes = int(unique_ids.numel())
    merged["gene"].global_id = unique_ids
    merged[CELL_TO_GENE].edge_index = edge_index
    merged[CELL_TO_GENE].edge_weight = weights
    merged[GENE_TO_CELL].edge_index = edge_index.flip(0)
    merged[GENE_TO_CELL].edge_weight = weights.clone()
    merged.y = graphs[0].y.reshape(-1)
    merged.sample_id = str(graphs[0].sample_id)
    merged.patient_id = str(graphs[0].patient_id)
    merged.cell_type = ",".join(name for name in cell_types if name)
    merged.gene_strategy = getattr(graphs[0], "gene_strategy", "")
    return merged
