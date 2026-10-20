from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.fi.encoder import FIPretrainer
from src.fi.message_graph import (
    EDGE_ATTR_NAMES,
    computational_edges,
    sample_negative_pairs,
    select_curated_edges,
    split_biological_pairs,
    unordered_pair,
)
from src.fi.pretrain import parse_args, prepare_graph, train_fi_encoder
from src.fi.parse import SEMANTIC_ATTRIBUTES
import numpy as np


def _edge(gene1, gene2, direction, source, target, **bits):
    semantics = {name: 0 for name in SEMANTIC_ATTRIBUTES}
    semantics.update(bits)
    return {
        "gene1": gene1,
        "gene2": gene2,
        "direction_type": direction,
        "source": source,
        "target": target,
        "confidence": 1.0,
        "semantic_attributes": semantics,
    }


def test_predicted_edges_are_excluded_by_default():
    edges = [
        _edge("A", "B", "directed", "A", "B", activation=1),
        _edge("C", "D", "undirected", None, None, predicted=1),
    ]
    kept = select_curated_edges(edges, include_predicted=False)
    assert [edge["gene1"] for edge in kept] == ["A"]
    assert len(select_curated_edges(edges, include_predicted=True)) == 2


def test_directed_reverse_is_computational_only():
    genes = {"A": 0, "B": 1}
    edge = _edge("A", "B", "directed", "A", "B", inhibition=1)
    rows = computational_edges(edge, genes)
    assert [(row[0], row[1], row[2][-1]) for row in rows] == [(0, 1, 0.0), (1, 0, 1.0)]
    assert rows[0][2][:-1] == rows[1][2][:-1]
    assert rows[0][2][EDGE_ATTR_NAMES.index("is_directed")] == 1.0


def test_undirected_and_bidirectional_keep_their_flags():
    genes = {"A": 0, "B": 1}
    undirected = computational_edges(_edge("A", "B", "undirected", None, None, complex=1), genes)
    bidirectional = computational_edges(_edge("A", "B", "bidirectional", None, None, activation=1), genes)
    assert [row[2][-1] for row in undirected] == [0.0, 0.0]
    assert undirected[0][2][EDGE_ATTR_NAMES.index("is_undirected")] == 1.0
    assert [row[2][-1] for row in bidirectional] == [0.0, 0.0]
    assert bidirectional[0][2][EDGE_ATTR_NAMES.index("is_bidirectional")] == 1.0


def test_held_out_pairs_and_reverse_edges_stay_out_of_training():
    edges = [
        _edge("A", "B", "directed", "A", "B", activation=1),
        _edge("A", "C", "directed", "C", "A", inhibition=1),
        _edge("B", "C", "undirected", None, None, complex=1),
        _edge("D", "E", "bidirectional", None, None, catalysis=1),
        _edge("E", "F", "directed", "E", "F", reaction_association=1),
    ]
    gene_to_id = {gene: index for index, gene in enumerate(["A", "B", "C", "D", "E", "F"])}
    splits = split_biological_pairs(edges, gene_to_id, seed=0, train_fraction=0.6, val_fraction=0.2)
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    train_pairs = {unordered_pair(edge, gene_to_id) for edge in edges if unordered_pair(edge, gene_to_id) in splits["train"]}
    for edge in edges:
        pair = unordered_pair(edge, gene_to_id)
        if pair not in train_pairs:
            continue
        for src, dst, _attr in computational_edges(edge, gene_to_id):
            message_pair = (src, dst) if src < dst else (dst, src)
            assert message_pair == pair
            assert message_pair not in splits["val"]
            assert message_pair not in splits["test"]


def test_negative_pairs_are_not_known_fi_pairs():
    banned = {(0, 1), (1, 2)}
    negatives = sample_negative_pairs(20, 8, banned, np.random.default_rng(0))
    assert all((min(a, b), max(a, b)) not in banned for a, b in negatives.tolist())


def test_encoder_uses_edge_attributes():
    model = FIPretrainer(num_genes=4, hidden_dim=8, num_layers=2, dropout=0.0)
    assert model.encoder.convs[0].lin is not None
    assert model.encoder.convs[0].lin.weight.shape[1] == len(EDGE_ATTR_NAMES)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    base = torch.zeros((2, len(EDGE_ATTR_NAMES)))
    changed = base.clone()
    changed[0, 0] = 4.0
    model.eval()
    first = model.encode(edge_index, base)
    second = model.encode(edge_index, changed)
    assert not torch.allclose(first, second)


def test_pretrain_writes_embeddings(tmp_path: Path):
    edges = []
    genes = [f"G{i}" for i in range(12)]
    for index in range(11):
        edges.append(_edge(genes[index], genes[index + 1], "directed", genes[index], genes[index + 1], activation=1))
    graph_path = tmp_path / "fi_graph.json"
    graph_path.write_text(json.dumps({"edges": edges}))
    out = tmp_path / "out"
    args = parse_args(
        [
            "--graph-json",
            str(graph_path),
            "--output-dir",
            str(out),
            "--epochs",
            "1",
            "--hidden-dim",
            "8",
            "--device",
            "cpu",
        ]
    )
    train_fi_encoder(args)
    saved = torch.load(out / "fi_gene_embeddings.pt", map_location="cpu")
    assert saved["embeddings"].shape[0] == len(saved["genes"])
    assert (out / "fi_encoder_best.pt").exists()
    assert (out / "fi_split.json").exists()
    prepared = prepare_graph(graph_path, include_predicted=False, seed=42)
    assert prepared["n_kept_edges"] == 11
