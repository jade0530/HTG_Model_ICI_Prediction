import numpy as np
import pytest
from scipy import sparse

from src.data.dataset import AnnDataSample
from src.data.sampler import CellSampler
from src.graph.build_local_graph import GeneUniverse, build_local_graph, summarize_local_graph


def make_sample(matrix):
    return AnnDataSample(
        sample_id="S1",
        patient_id="P1",
        X=matrix,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        cell_metadata={"cell_type": np.array(["T", "B", "T"])},
        label=1,
    )


@pytest.mark.parametrize("as_sparse", [False, True])
def test_graph_has_consistent_gene_ids_and_nonzero_weighted_reverse_edges(as_sparse):
    matrix = np.array([[2.0, 9.0, 0.0, 1.0], [0.0, 8.0, 3.0, 0.0], [0, 7, 0, 0]])
    sample = make_sample(sparse.csr_matrix(matrix) if as_sparse else matrix)
    graph = build_local_graph(sample, [0, 1], GeneUniverse(["G1", "G2", "G3", "G4"]))

    assert graph.node_types == ["cell", "gene"]
    assert graph["cell"].num_nodes == 2
    assert graph["gene"].global_id.tolist() == [0, 1, 2]
    forward = graph["cell", "expresses", "gene"]
    reverse = graph["gene", "expressed_by", "cell"]
    assert forward.edge_index.shape == (2, 3)
    assert sorted(forward.edge_weight.tolist()) == [1.0, 2.0, 3.0]
    assert (forward.edge_weight > 0).all()
    assert reverse.edge_index.equal(forward.edge_index.flip(0))
    assert reverse.edge_weight.equal(forward.edge_weight)
    assert graph.sample_id == "S1"
    assert graph.patient_id == "P1"
    assert graph.y.tolist() == [1.0]
    summary = summarize_local_graph(graph)
    assert summary["num_cells"] == 2
    assert summary["num_genes"] == 3
    assert summary["num_expression_edges"] == 3
    assert summary["expression_density"] == 0.5
    assert summary["min_edge_weight"] == 1.0
    assert summary["max_edge_weight"] == 3.0
    assert summary["tensor_bytes"] > 0


def test_validation_sampling_is_reproducible_and_small_samples_use_all_cells():
    sampler = CellSampler(num_cells=2, seed=13)
    assert np.array_equal(
        sampler.sample(10, training=False), sampler.sample(10, training=False)
    )
    assert np.array_equal(sampler.sample(2, training=False), np.array([0, 1]))


def test_different_validation_views_are_deterministic():
    sampler = CellSampler(num_cells=3, seed=13)
    first = sampler.sample(20, training=False, view_index=1)
    assert np.array_equal(first, sampler.sample(20, training=False, view_index=1))
    assert not np.array_equal(first, sampler.sample(20, training=False, view_index=2))


def test_zero_expression_graph_is_rejected():
    sample = make_sample(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="no positive expression"):
        build_local_graph(sample, [0, 1], GeneUniverse(["G1", "G2", "G3"]))
