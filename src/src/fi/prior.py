"""Align frozen FI gene embeddings to an HTG gene universe.

Genes with no FI vector stay zero, so the HTG base embedding is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

GENE_PRIOR_MODES = ("none", "fi_topology_frozen", "fi_semantic_frozen")

_WORKTREE = Path(__file__).resolve().parents[3]
DEFAULT_FI_EMBEDDINGS = {
    "fi_topology_frozen": _WORKTREE
    / "outputs"
    / "fi_encoder_experiments"
    / "topology_only"
    / "fi_gene_embeddings.pt",
    "fi_semantic_frozen": _WORKTREE
    / "outputs"
    / "fi_encoder_experiments"
    / "topology_semantics"
    / "fi_gene_embeddings.pt",
}


def resolve_fi_embedding_path(gene_prior: str, fi_embedding_path: str = "") -> Path:
    if fi_embedding_path:
        return Path(fi_embedding_path)
    if gene_prior not in DEFAULT_FI_EMBEDDINGS:
        raise ValueError(f"gene_prior must be one of {GENE_PRIOR_MODES}")
    return DEFAULT_FI_EMBEDDINGS[gene_prior]


def load_fi_embeddings(path: Path) -> tuple[Tensor, list[str]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].detach().float().cpu()
    genes = [str(gene) for gene in payload["genes"]]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(genes):
        raise ValueError(f"FI embedding file does not match its gene list: {path}")
    return embeddings, genes


def align_fi_prior(
    gene_names: Sequence[str],
    path: Path,
) -> tuple[Tensor, dict[str, object]]:
    """One FI row per HTG gene. Missing genes are a zero vector of the FI width."""
    embeddings, fi_genes = load_fi_embeddings(path)
    fi_index = {gene: index for index, gene in enumerate(fi_genes)}
    table = torch.zeros(len(gene_names), embeddings.shape[1], dtype=torch.float32)
    matched: list[str] = []
    missing: list[str] = []
    for row, name in enumerate(gene_names):
        source = fi_index.get(str(name))
        if source is None:
            missing.append(str(name))
            continue
        table[row] = embeddings[source]
        matched.append(str(name))
    htg = set(str(name) for name in gene_names)
    overlap = sum(gene in htg for gene in fi_genes)
    coverage = {
        "fi_embedding_path": str(path),
        "n_htg_genes": len(gene_names),
        "n_fi_genes": len(fi_genes),
        "overlap": overlap,
        "n_with_fi": len(matched),
        "n_without_fi": len(missing),
        "coverage": (len(matched) / len(gene_names)) if gene_names else 0.0,
        "examples": [
            {"gene": gene, "htg_id": gene_names.index(gene), "fi_id": fi_index[gene]}
            for gene in matched[:8]
        ],
    }
    return table, coverage
