import numpy as np
import pytest
from scipy import sparse

from src.data.data_loader import AnnDataSample, sample_from_anndata, select_sample_hvgs
from src.data.sampler import CellSampler
from src.graph.build_local_graph import GeneUniverse, build_local_graph, summarize_local_graph


def make_sample(matrix, hvg_names=("G2", "G1", "G3")):
    return AnnDataSample(
        sample_id="S1",
        patient_id="P1",
        X=matrix,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        hvg_names=hvg_names,
        cell_annotation={
            "predicted_labels": np.array(["T", "B", "T"]),
            "conf_score": np.array([0.9, 0.8, 0.7]),
        },
        clinical_metadata={"treatment": np.array(["ICI", "ICI", "ICI"])},
        label="R",
        n_hvg=3000,
    )


@pytest.mark.parametrize("as_sparse", [False, True])
def test_graph_has_consistent_gene_ids_and_nonzero_weighted_reverse_edges(as_sparse):
    matrix = np.array([[2.0, 9.0, 0.0, 1.0], [0.0, 8.0, 3.0, 0.0], [0, 7, 0, 0]])
    sample = make_sample(sparse.csr_matrix(matrix) if as_sparse else matrix)
    graph = build_local_graph(
        sample, [0, 1], GeneUniverse(["G1", "G2", "G3", "G4"]), gene_strategy="expressed"
    )

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
        build_local_graph(
            sample, [0, 1], GeneUniverse(["G1", "G2", "G3"]), gene_strategy="expressed"
        )


class _Obs:
    def __init__(self, columns: dict[str, np.ndarray]):
        self._columns = columns

    @property
    def columns(self):
        return list(self._columns)

    def __getitem__(self, key: str):
        return self._columns[key]

    def __contains__(self, key: str) -> bool:
        return key in self._columns


class _AnnData:
    def __init__(self, obs: dict[str, np.ndarray]):
        self.X = np.ones((3, 2), dtype=np.float32)
        self.layers = {"counts": np.arange(6, dtype=np.float32).reshape(3, 2)}
        self.var_names = ["g1", "g2"]
        self.obs = _Obs(obs)


def test_sample_from_anndata_splits_annotation_and_clinical_columns():
    adata = _AnnData(
        {
            "predicted_labels": np.array(["T", "B", "T"]),
            "conf_score": np.array([0.9, 0.4, 0.8]),
            "age": np.array([50, 50, 51]),
            "treatment": np.array(["ICI", "ICI", "ICI"]),
        }
    )
    sample = sample_from_anndata(
        adata,
        sample_id="S1",
        patient_id="P1",
        label="NR",
        load_cell_annotation=True,
        load_clinical_metadata=True,
        hvg_names=("g1", "g2"),
    )
    assert set(sample.cell_annotation) == {"predicted_labels", "conf_score"}
    assert set(sample.clinical_metadata) == {"age", "treatment"}
    assert list(sample.cell_annotation["predicted_labels"]) == ["T", "B", "T"]
    assert list(sample.clinical_metadata["age"]) == [50, 50, 51]
    assert sample.n_hvg == 3000
    assert set(sample.hvg_names) <= {"g1", "g2"}
    assert len(sample.hvg_names) == 2


def test_cell_type_proportional_sample_is_stratified_and_reproducible():
    labels = np.array(["T"] * 6 + ["B"] * 4)
    scores = np.linspace(0.1, 1.0, 10)
    annotation = {"predicted_labels": labels, "conf_score": scores}
    sampler = CellSampler(num_cells=5, seed=13)
    first = sampler.cell_type_proportional_sample(
        10, training=False, cell_annotation=annotation
    )
    again = sampler.cell_type_proportional_sample(
        10, training=False, cell_annotation=annotation
    )
    other_view = sampler.cell_type_proportional_sample(
        10, training=False, view_index=1, cell_annotation=annotation
    )
    sampled_labels = labels[first]
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_view)
    assert (sampled_labels == "T").sum() == 3
    assert (sampled_labels == "B").sum() == 2


def test_sample_by_cell_type_builds_type_specific_indices():
    labels = np.array(["T"] * 6 + ["B"] * 4)
    scores = np.linspace(0.1, 1.0, 10)
    annotation = {"predicted_labels": labels, "conf_score": scores}
    sampler = CellSampler(num_cells=3, seed=13)
    t_cells = sampler.sample_by_cell_type(
        10, cell_type="T", training=False, cell_annotation=annotation
    )
    all_types = sampler.sample_all_cell_types(
        10, training=False, cell_annotation=annotation
    )
    assert set(all_types) == {"B", "T"}
    assert np.array_equal(t_cells, all_types["T"])
    assert set(labels[t_cells]) == {"T"}
    assert set(labels[all_types["B"]]) == {"B"}
    assert len(t_cells) == 3
    assert len(all_types["B"]) == 3


def test_sample_by_cell_type_requires_annotation_columns():
    sampler = CellSampler(num_cells=2)
    with pytest.raises(ValueError, match="cell-type sampling requires"):
        sampler.sample_by_cell_type(4, cell_type="T", training=False, cell_annotation={})
    with pytest.raises(ValueError, match="no cells with predicted_labels"):
        sampler.sample_by_cell_type(
            3,
            cell_type="NK",
            training=False,
            cell_annotation={
                "predicted_labels": np.array(["T", "B", "T"]),
                "conf_score": np.array([0.9, 0.8, 0.7]),
            },
        )


def test_select_sample_hvgs_uses_scanpy_on_normalised_x():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    rng = np.random.default_rng(0)
    n_cells, n_genes = 80, 40
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    X[:, 0] *= 25
    adata = anndata.AnnData(X)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    hvgs = select_sample_hvgs(adata, n_hvg=5)
    assert "highly_variable" in adata.var.columns
    assert len(hvgs) == 5
    assert set(hvgs) == set(adata.var_names[adata.var["highly_variable"]].astype(str))
    sample_adata = adata[:20].copy()
    sample = sample_from_anndata(
        sample_adata,
        sample_id="S1",
        patient_id="P1",
        label="R",
        load_normalised_expression=True,
        n_hvg=5,
    )
    assert sample.hvg_names == hvgs


def test_hvg_graph_keeps_only_sample_hvgs():
    matrix = np.array([[2.0, 9.0, 0.0, 1.0], [0.0, 8.0, 3.0, 0.0], [0, 7, 0, 0]])
    sample = make_sample(matrix, hvg_names=("G1", "G3"))
    universe = GeneUniverse(["G1", "G2", "G3", "G4"])
    expressed = build_local_graph(sample, [0, 1], universe, gene_strategy="expressed")
    hvg = build_local_graph(sample, [0, 1], universe, gene_strategy="hvg")
    assert expressed["gene"].global_id.tolist() == [0, 1, 2]
    assert sorted(expressed["cell", "expresses", "gene"].edge_weight.tolist()) == [1.0, 2.0, 3.0]
    assert hvg["gene"].global_id.tolist() == [0, 2]
    assert sorted(hvg["cell", "expresses", "gene"].edge_weight.tolist()) == [1.0, 3.0]
    assert hvg.gene_strategy == "hvg"


def test_sample_from_anndata_respects_n_hvg():
    adata = _AnnData(
        {
            "predicted_labels": np.array(["T", "B", "T"]),
            "conf_score": np.array([0.9, 0.4, 0.8]),
        }
    )
    sample = sample_from_anndata(
        adata, sample_id="S1", patient_id="P1", label="NR", n_hvg=1, hvg_names=("g1",)
    )
    assert sample.n_hvg == 1
    assert len(sample.hvg_names) == 1
    assert sample.hvg_names[0] in {"g1", "g2"}
