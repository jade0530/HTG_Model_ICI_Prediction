"""Static curated FI edges for sample HTGs, indexed by a GeneUniverse.

Predicted FI records are excluded. A pair is kept only when both genes are in
the universe. The same edge tensors are attached to every sample graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.fi.message_graph import (
    EDGE_ATTR_NAMES,
    computational_edges,
    load_fi_edges,
    select_curated_edges,
    unordered_pair,
)

FI_GRAPH_MODES = ("baseline", "fi_topology", "fi_relation_aware")
DEFAULT_FI_GRAPH = Path(
    "/media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/data/functional_interaction/fi_graph.json"
)

_CACHE: dict[tuple, "SampleFIEdges"] = {}


@dataclass
class SampleFIEdges:
    """Universe-indexed FI edges. ``edge_attr`` is set only for relation-aware mode."""

    mode: str
    edge_index: Tensor
    edge_attr: Tensor | None
    stats: dict[str, object]


def _pairs_in_universe(edges: list[dict], gene_to_id: dict[str, int]) -> list[dict]:
    kept = []
    for edge in select_curated_edges(edges, include_predicted=False):
        if edge["gene1"] in gene_to_id and edge["gene2"] in gene_to_id and edge["gene1"] != edge["gene2"]:
            kept.append(edge)
    return kept


def fi_degree(num_genes: int, pairs: set[tuple[int, int]]) -> np.ndarray:
    degree = np.zeros(num_genes, dtype=np.int64)
    for left, right in pairs:
        degree[left] += 1
        degree[right] += 1
    return degree


def summarize_fi(gene_names: tuple[str, ...] | list[str], edges: list[dict]) -> dict[str, object]:
    """Coverage and degree of curated FI pairs inside one GeneUniverse."""
    gene_to_id = {str(name): index for index, name in enumerate(gene_names)}
    kept = _pairs_in_universe(edges, gene_to_id)
    pairs = {unordered_pair(edge, gene_to_id) for edge in kept}
    degree = fi_degree(len(gene_names), pairs)
    covered = int((degree > 0).sum())
    order = np.argsort(-degree, kind="stable")
    hubs = [
        {"gene": str(gene_names[int(index)]), "degree": int(degree[int(index)])}
        for index in order[:10]
        if degree[int(index)] > 0
    ]
    relation_edges = []
    for edge in kept:
        relation_edges.extend(computational_edges(edge, gene_to_id))
    return {
        "n_gene_universe": len(gene_names),
        "n_fi_covered_genes": covered,
        "fi_coverage": (covered / len(gene_names)) if gene_names else 0.0,
        "n_fi_pairs": len(pairs),
        "n_topology_edges": len(pairs) * 2,
        "n_relation_edges": len(relation_edges),
        "degree_mean": float(degree.mean()) if len(degree) else 0.0,
        "degree_median": float(np.median(degree)) if len(degree) else 0.0,
        "degree_max": int(degree.max()) if len(degree) else 0,
        "top_hubs": hubs,
    }


def build_sample_fi_edges(
    gene_names: tuple[str, ...] | list[str],
    *,
    mode: str,
    graph_json: Path = DEFAULT_FI_GRAPH,
    edges: list[dict] | None = None,
) -> SampleFIEdges:
    """Curated FI edges in GeneUniverse index space."""
    if mode not in ("fi_topology", "fi_relation_aware"):
        raise ValueError("mode must be 'fi_topology' or 'fi_relation_aware'")
    gene_to_id = {str(name): index for index, name in enumerate(gene_names)}
    source = list(edges) if edges is not None else load_fi_edges(graph_json)
    kept = _pairs_in_universe(source, gene_to_id)
    pairs = {unordered_pair(edge, gene_to_id) for edge in kept}
    if mode == "fi_topology":
        ordered = sorted(pairs)
        src = [pair[0] for pair in ordered] + [pair[1] for pair in ordered]
        dst = [pair[1] for pair in ordered] + [pair[0] for pair in ordered]
        edge_index = torch.tensor([src, dst], dtype=torch.long) if ordered else torch.empty((2, 0), dtype=torch.long)
        edge_attr = None
    else:
        rows = []
        for edge in kept:
            rows.extend(computational_edges(edge, gene_to_id))
        if rows:
            edge_index = torch.tensor([[row[0] for row in rows], [row[1] for row in rows]], dtype=torch.long)
            edge_attr = torch.tensor([row[2] for row in rows], dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(EDGE_ATTR_NAMES)), dtype=torch.float32)
    return SampleFIEdges(
        mode=mode,
        edge_index=edge_index,
        edge_attr=edge_attr,
        stats=summarize_fi(gene_names, source),
    )


def cached_sample_fi_edges(
    gene_names: tuple[str, ...],
    *,
    mode: str,
    graph_json: Path = DEFAULT_FI_GRAPH,
) -> SampleFIEdges:
    key = (mode, str(graph_json), gene_names)
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_sample_fi_edges(gene_names, mode=mode, graph_json=graph_json)
        _CACHE[key] = cached
    return cached


def format_fi_stats(stats: dict[str, object]) -> str:
    hubs = ", ".join(f"{item['gene']} ({item['degree']})" for item in stats["top_hubs"][:8])
    return (
        f"gene_universe={stats['n_gene_universe']} "
        f"fi_covered={stats['n_fi_covered_genes']} "
        f"coverage={float(stats['fi_coverage']):.4f} "
        f"fi_pairs={stats['n_fi_pairs']} "
        f"topology_edges={stats['n_topology_edges']} "
        f"relation_edges={stats['n_relation_edges']} "
        f"degree_mean={float(stats['degree_mean']):.2f} "
        f"degree_median={float(stats['degree_median']):.1f} "
        f"degree_max={stats['degree_max']} "
        f"hubs={hubs}"
    )
