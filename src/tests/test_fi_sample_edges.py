from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.data.data_loader import AnnDataSample
from src.fi.message_graph import EDGE_ATTR_NAMES
from src.fi.parse import SEMANTIC_ATTRIBUTES
from src.fi.sample_edges import build_sample_fi_edges
from src.graph.build_local_graph import GeneUniverse, build_local_graph
from src.train.model import FIRelationConv, FITopologyConv, SampleGraphClassifier


def _edge(gene1, gene2, direction="undirected", source=None, target=None, **bits):
    semantics = {name: 0 for name in SEMANTIC_ATTRIBUTES}
    semantics.update(bits)
    return {
        "gene1": gene1,
        "gene2": gene2,
        "direction_type": direction,
        "source": source,
        "target": target,
        "confidence": 0.9,
        "semantic_attributes": semantics,
    }


def _sample(values):
    matrix = np.asarray(values, dtype=np.float32)
    return AnnDataSample(
        sample_id="S1",
        patient_id="P1",
        X=matrix,
        gene_names=("A", "B", "C"),
        hvg_names=("A", "B", "C"),
        cell_annotation={},
        clinical_metadata={},
        label="R",
        n_hvg=3,
    )


def test_topology_edges_are_symmetric_and_drop_uncovered_genes():
    edges = [
        _edge("A", "B", "directed", "A", "B", activation=1),
        _edge("C", "D", "undirected", predicted=1),
        _edge("B", "Z", "undirected", complex=1),
    ]
    prepared = build_sample_fi_edges(("A", "B", "C"), mode="fi_topology", edges=edges)
    assert prepared.edge_attr is None
    assert prepared.stats["n_fi_pairs"] == 1
    assert prepared.stats["n_fi_covered_genes"] == 2
    assert prepared.stats["n_gene_universe"] == 3
    pairs = {tuple(sorted(pair)) for pair in prepared.edge_index.t().tolist()}
    assert pairs == {(0, 1)}
    assert prepared.edge_index.shape[1] == 2
    assert 2 not in prepared.edge_index.flatten().tolist()


def test_relation_edges_keep_direction_and_semantics():
    edges = [_edge("A", "B", "directed", "A", "B", inhibition=1)]
    prepared = build_sample_fi_edges(("A", "B", "C"), mode="fi_relation_aware", edges=edges)
    assert prepared.edge_attr is not None
    assert prepared.edge_attr.shape == (2, len(EDGE_ATTR_NAMES))
    reverse = EDGE_ATTR_NAMES.index("is_reverse")
    assert sorted(prepared.edge_attr[:, reverse].tolist()) == [0.0, 1.0]
    assert prepared.edge_attr[0, EDGE_ATTR_NAMES.index("inhibition")] == 1.0


def test_uncovered_gene_stays_in_the_sample_graph_without_fi_messages():
    edges = [_edge("A", "B", "undirected", complex=1)]
    fi_edges = build_sample_fi_edges(("A", "B", "C"), mode="fi_topology", edges=edges)
    graph = build_local_graph(
        _sample([[1.0, 0.0, 0.0]]),
        [0],
        GeneUniverse(["A", "B", "C"]),
        gene_strategy="hvg",
        fi_edges=fi_edges,
    )
    assert graph["gene"].global_id.tolist() == [0, 1, 2]
    expressed = graph["cell", "expresses", "gene"].edge_index[1].tolist()
    assert expressed == [0]
    incident = set(graph["gene", "fi", "gene"].edge_index.flatten().tolist())
    assert 2 not in incident
    assert incident == {0, 1}


def test_fi_messages_are_zero_without_neighbors_and_relation_uses_edge_attr():
    topology = FITopologyConv(4)
    relation = FIRelationConv(4, len(EDGE_ATTR_NAMES))
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    topo = topology(x, edge_index)
    assert torch.equal(topo[2], torch.zeros(4))
    base = torch.zeros((2, len(EDGE_ATTR_NAMES)))
    changed = base.clone()
    changed[0, 0] = 3.0
    first = relation(x[:2], edge_index, base)
    second = relation(x[:2], edge_index, changed)
    assert not torch.allclose(first, second)
    isolated = relation(x, torch.tensor([[0], [1]]), base[:1])
    assert torch.equal(isolated[2], torch.zeros(4))


def test_classifier_accepts_shared_fi_edges():
    edges = [_edge("A", "B", "undirected", activation=1)]
    fi_edges = build_sample_fi_edges(("A", "B", "C"), mode="fi_relation_aware", edges=edges)
    graph = build_local_graph(
        _sample([[1.0, 2.0, 0.0]]),
        [0],
        GeneUniverse(["A", "B", "C"]),
        gene_strategy="hvg",
        fi_edges=fi_edges,
    )
    model = SampleGraphClassifier(3, hidden_dim=8, encoder="sage", fi_mode="fi_relation_aware")
    assert model(graph).shape == (1,)
