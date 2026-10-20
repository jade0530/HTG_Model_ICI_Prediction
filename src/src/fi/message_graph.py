"""Turn curated biological FI records into a split-safe message-passing graph.

Predicted FI edges are left out unless requested. Undirected and bidirectional
records stay one biological relation, then both computational directions are
added for message passing. A directed reverse edge is only that computational
copy: it is not a new biological mechanism.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.fi.parse import SEMANTIC_ATTRIBUTES

CURATED_SEMANTICS = tuple(name for name in SEMANTIC_ATTRIBUTES if name != "predicted")
EDGE_ATTR_NAMES = (
    *CURATED_SEMANTICS,
    "confidence",
    "is_directed",
    "is_undirected",
    "is_bidirectional",
    "is_reverse",
)


def load_fi_edges(path: Path) -> list[dict]:
    """Read biological FI edges. The JSON file is not rewritten."""
    payload = json.loads(Path(path).read_text())
    return list(payload["edges"])


def select_curated_edges(edges: list[dict], *, include_predicted: bool) -> list[dict]:
    """Drop records whose predicted bit is set, unless the flag keeps them."""
    if include_predicted:
        return list(edges)
    return [edge for edge in edges if int(edge["semantic_attributes"]["predicted"]) == 0]


def build_gene_index(edges: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Contiguous ids for genes that remain on at least one kept FI record."""
    genes = sorted({edge["gene1"] for edge in edges} | {edge["gene2"] for edge in edges})
    return genes, {gene: index for index, gene in enumerate(genes)}


def unordered_pair(edge: dict, gene_to_id: dict[str, int]) -> tuple[int, int]:
    left = gene_to_id[edge["gene1"]]
    right = gene_to_id[edge["gene2"]]
    return (left, right) if left < right else (right, left)


def split_biological_pairs(
    edges: list[dict],
    gene_to_id: dict[str, int],
    *,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, set[tuple[int, int]]]:
    """Assign every unordered gene pair to one split. Records of a pair stay together."""
    pairs = sorted({unordered_pair(edge, gene_to_id) for edge in edges})
    order = np.random.default_rng(seed).permutation(len(pairs))
    n_val = int(round(len(pairs) * val_fraction))
    n_test = int(round(len(pairs) * (1.0 - train_fraction - val_fraction)))
    val_ids = {pairs[index] for index in order[:n_val]}
    test_ids = {pairs[index] for index in order[n_val : n_val + n_test]}
    train_ids = {pairs[index] for index in order[n_val + n_test :]}
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def edge_attr_vector(edge: dict, *, is_reverse: int) -> list[float]:
    """Semantic bits, confidence, direction flags, and the computational reverse bit."""
    semantics = edge["semantic_attributes"]
    confidence = edge["confidence"]
    direction = edge["direction_type"]
    return [
        float(semantics[name]) for name in CURATED_SEMANTICS
    ] + [
        0.0 if confidence is None else float(confidence),
        float(direction == "directed"),
        float(direction == "undirected"),
        float(direction == "bidirectional"),
        float(is_reverse),
    ]


def computational_edges(
    edge: dict,
    gene_to_id: dict[str, int],
) -> list[tuple[int, int, list[float]]]:
    """Both message-passing directions for one biological FI record."""
    left = gene_to_id[edge["gene1"]]
    right = gene_to_id[edge["gene2"]]
    direction = edge["direction_type"]
    if direction == "directed":
        if edge["source"] == edge["gene1"] and edge["target"] == edge["gene2"]:
            source, target = left, right
        elif edge["source"] == edge["gene2"] and edge["target"] == edge["gene1"]:
            source, target = right, left
        else:
            return []
        return [
            (source, target, edge_attr_vector(edge, is_reverse=0)),
            (target, source, edge_attr_vector(edge, is_reverse=1)),
        ]
    if direction in {"undirected", "bidirectional"}:
        forward = edge_attr_vector(edge, is_reverse=0)
        return [(left, right, forward), (right, left, list(forward))]
    return []


def stack_edges(rows: list[tuple[int, int, list[float]]]) -> tuple[Tensor, Tensor]:
    if not rows:
        empty_index = torch.empty((2, 0), dtype=torch.long)
        empty_attr = torch.empty((0, len(EDGE_ATTR_NAMES)), dtype=torch.float32)
        return empty_index, empty_attr
    src = [row[0] for row in rows]
    dst = [row[1] for row in rows]
    attr = [row[2] for row in rows]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    return edge_index, edge_attr


def pairs_tensor(pairs: set[tuple[int, int]]) -> Tensor:
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    ordered = sorted(pairs)
    return torch.tensor(ordered, dtype=torch.long)


def semantic_examples(
    edges: list[dict],
    gene_to_id: dict[str, int],
    allowed: set[tuple[int, int]],
) -> tuple[Tensor, Tensor]:
    """One multi-label target per biological record in ``allowed`` pairs."""
    pair_rows = []
    targets = []
    for edge in edges:
        pair = unordered_pair(edge, gene_to_id)
        if pair not in allowed:
            continue
        pair_rows.append(pair)
        targets.append([float(edge["semantic_attributes"][name]) for name in CURATED_SEMANTICS])
    if not pair_rows:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0, len(CURATED_SEMANTICS)))
    return torch.tensor(pair_rows, dtype=torch.long), torch.tensor(targets, dtype=torch.float32)


def sample_negative_pairs(
    num_pairs: int,
    num_genes: int,
    banned: set[tuple[int, int]],
    rng: np.random.Generator,
) -> Tensor:
    """Unordered gene pairs that are not known positive FI pairs."""
    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(chosen) < num_pairs:
        draw = rng.integers(0, num_genes, size=(max(num_pairs, 1) * 2, 2))
        for left, right in draw.tolist():
            if left == right:
                continue
            pair = (left, right) if left < right else (right, left)
            if pair in banned or pair in seen:
                continue
            seen.add(pair)
            chosen.append(pair)
            if len(chosen) == num_pairs:
                break
    return torch.tensor(chosen, dtype=torch.long)


def assert_message_split(
    train_edge_index: Tensor,
    splits: dict[str, set[tuple[int, int]]],
    negatives: dict[str, Tensor],
    positive: set[tuple[int, int]],
) -> None:
    """Held-out biological pairs, including reverse copies, stay out of training."""
    train_pairs = splits["train"]
    held_out = splits["val"] | splits["test"]
    if train_pairs & held_out:
        raise RuntimeError("a biological FI pair was assigned to more than one split")
    for src, dst in train_edge_index.t().tolist():
        pair = (src, dst) if src < dst else (dst, src)
        if pair not in train_pairs:
            raise RuntimeError("training adjacency contains a non-training FI pair")
        if pair in held_out:
            raise RuntimeError("validation or test FI pair leaked into training adjacency")
    for name, pairs in negatives.items():
        for left, right in pairs.tolist():
            pair = (left, right) if left < right else (right, left)
            if pair in positive:
                raise RuntimeError(f"{name} negative pairs overlap a known positive FI pair")


def split_symbol_pairs(
    splits: dict[str, set[tuple[int, int]]],
    genes: list[str],
) -> dict[str, list[list[str]]]:
    return {
        name: [[genes[left], genes[right]] for left, right in sorted(pairs)]
        for name, pairs in splits.items()
    }
