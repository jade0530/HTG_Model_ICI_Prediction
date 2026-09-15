from pathlib import Path

import numpy as np
import pytest

from src.data.data_loader import AnnDataSample
from src.graph.build_local_graph import GeneUniverse
from src.train.dataset import (
    SampleRecord,
    assert_patient_disjoint,
    load_sample_manifest,
    patient_key,
    split_by_patient,
)
from src.train.loop import TrainConfig, fit
from src.train.metrics import classification_metrics, confusion_counts, select_threshold


def _toy_sample(sample_id: str, patient_id: str, label: str, seed: int) -> AnnDataSample:
    rng = np.random.default_rng(seed)
    matrix = rng.random((6, 4)).astype(np.float32) + 0.1
    return AnnDataSample(
        sample_id=sample_id,
        patient_id=patient_id,
        X=matrix,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        hvg_names=("G1", "G2", "G3"),
        cell_annotation={},
        clinical_metadata={},
        label=label,
        n_hvg=3,
    )


def test_classification_metrics_acc_auprc_f1():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_prob)
    assert metrics["acc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    matrix = confusion_counts(y_true, y_prob)
    assert matrix.tolist() == [[2, 0], [0, 2]]


def test_select_threshold_max_f1_when_scores_are_below_half():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.20, 0.30, 0.40])
    assert classification_metrics(y_true, y_prob, threshold=0.5)["f1"] == 0.0
    chosen = select_threshold(y_true, y_prob, strategy="max_f1")
    assert chosen < 0.5
    assert classification_metrics(y_true, y_prob, threshold=chosen)["f1"] == 1.0
    assert select_threshold(y_true, y_prob, strategy="fixed", fixed=0.5) == 0.5
    youden = select_threshold(y_true, y_prob, strategy="youden")
    assert classification_metrics(y_true, y_prob, threshold=youden)["f1"] == 1.0


def test_split_by_patient_is_disjoint():
    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P1", "R", 2),
        _toy_sample("S3", "P2", "NR", 3),
        _toy_sample("S4", "P3", "NR", 4),
    ]
    train, val = split_by_patient(samples, val_fraction=0.34, seed=0)
    train_patients = {sample.patient_id for sample in train}
    val_patients = {sample.patient_id for sample in val}
    assert train and val
    assert train_patients.isdisjoint(val_patients)
    assert train_patients | val_patients == {"P1", "P2", "P3"}


def test_split_by_patient_balances_responder_labels():
    samples = [
        _toy_sample(f"R{i}", f"PR{i}", "R", i)
        for i in range(8)
    ] + [
        _toy_sample(f"N{i}", f"PN{i}", "NR", 100 + i)
        for i in range(8)
    ]
    train, val = split_by_patient(samples, val_fraction=0.25, seed=0)
    again_train, again_val = split_by_patient(samples, val_fraction=0.25, seed=0)
    other_train, other_val = split_by_patient(samples, val_fraction=0.25, seed=1)
    assert_patient_disjoint(train, val)
    assert {item.patient_id for item in train} == {item.patient_id for item in again_train}
    assert {item.patient_id for item in val} == {item.patient_id for item in again_val}
    train_labels = [item.label for item in train]
    val_labels = [item.label for item in val]
    assert train_labels.count("R") == 6
    assert train_labels.count("NR") == 6
    assert val_labels.count("R") == 2
    assert val_labels.count("NR") == 2
    assert {item.patient_id for item in train} != {item.patient_id for item in other_train}

    train3, val3, test3 = split_by_patient(
        samples, val_fraction=0.25, test_fraction=0.25, seed=0
    )
    assert_patient_disjoint(train3, val3, test3)
    assert [item.label for item in train3].count("R") == 4
    assert [item.label for item in val3].count("R") == 2
    assert [item.label for item in test3].count("R") == 2
    assert [item.label for item in train3].count("NR") == 4
    assert [item.label for item in val3].count("NR") == 2
    assert [item.label for item in test3].count("NR") == 2


def test_split_keeps_mixed_label_patient_together():
    samples = [
        _toy_sample("A1", "Pmix", "R", 1),
        _toy_sample("A2", "Pmix", "NR", 2),
        _toy_sample("B1", "P2", "NR", 3),
        _toy_sample("C1", "P3", "R", 4),
        _toy_sample("D1", "P4", "NR", 5),
    ]
    train, val = split_by_patient(samples, val_fraction=0.25, seed=0)
    assert_patient_disjoint(train, val)
    mix = [item for item in train + val if item.patient_id == "Pmix"]
    assert len(mix) == 2
    mix_fold = train if mix[0] in train else val
    assert {item.sample_id for item in mix_fold if item.patient_id == "Pmix"} == {"A1", "A2"}


def _record(dataset_id: str, patient_id: str, sample_id: str) -> SampleRecord:
    return SampleRecord(
        h5ad_path=Path("missing.h5ad"),
        metadata_path=None,
        patient_id=patient_id,
        sample_id=sample_id,
        dataset_id=dataset_id,
        label="R",
        tissue="Tumor",
        ici_phase="pre",
        cancer_type="BCC",
        output_file=f"sample_h5ad/Tumor/pre/{sample_id}.h5ad",
    )


def test_same_patient_samples_stay_in_one_split():
    records = [
        _record("Au_et_al", "A01", "A01_neg"),
        _record("Au_et_al", "A01", "A01_pos"),
        _record("GSE123813", "su001", "su001_pre"),
        _record("GSE123813", "su002", "su002_pre"),
        _record("GSE123813", "su003", "su003_pre"),
    ]
    train, val = split_by_patient(records, val_fraction=0.4, seed=0)
    assert_patient_disjoint(train, val)
    a01 = [item for item in train + val if item.patient_id == "A01"]
    assert len(a01) == 2
    assert {patient_key(item) for item in a01} == {"Au_et_al::A01"}


def test_same_patient_id_in_different_datasets_is_not_merged():
    records = [
        _record("StudyA", "P1", "S1"),
        _record("StudyB", "P1", "S2"),
        _record("StudyA", "P2", "S3"),
        _record("StudyB", "P3", "S4"),
    ]
    train, val = split_by_patient(records, val_fraction=0.25, seed=1)
    assert_patient_disjoint(train, val)
    keys = {patient_key(item) for item in train + val}
    assert keys == {"StudyA::P1", "StudyB::P1", "StudyA::P2", "StudyB::P3"}


def test_load_sample_manifest_filters_tumor_pre(tmp_path):
    manifest = tmp_path / "sample_manifest.csv"
    manifest.write_text(
        "output_file,metadata_file,source_file,Cancer type,patient_id,sample_id,"
        "dataset_id,Response,Response definition,Tissue,ICI_phase\n"
        "sample_h5ad/Tumor/pre/a.h5ad,a.metadata.csv,src.h5ad,BCC,su001,su001_pre,"
        "GSE123813,R,RECIST,Tumor,pre\n"
        "sample_h5ad/Tumor/on/b.h5ad,b.metadata.csv,src.h5ad,BCC,su001,su001_on,"
        "GSE123813,R,RECIST,Tumor,on\n"
        "sample_h5ad/PBMC/pre/c.h5ad,c.metadata.csv,src.h5ad,CRC,p2,p2_pre,"
        "GSE130157,NR,RECIST,PBMC,pre\n"
        "sample_h5ad/Tumor/pre/d.h5ad,d.metadata.csv,src.h5ad,MCC,p3,p3_pre,"
        "GSE235090,Intermediate,Path,Tumor,pre\n"
    )
    records = load_sample_manifest(tmp_path, manifest, skip_missing=False)
    assert [item.sample_id for item in records] == ["su001_pre"]
    assert records[0].label == "R"


def test_real_manifest_tumor_pre_split_has_no_patient_leak():
    records = load_sample_manifest("/does/not/need/to/exist", skip_missing=False)
    train, val, test = split_by_patient(
        records, val_fraction=0.15, test_fraction=0.15, seed=0
    )
    assert_patient_disjoint(train, val, test)
    assert {patient_key(item) for item in train + val + test} == {
        patient_key(item) for item in records
    }


def test_build_gene_universe_keeps_shared_order():
    from src.train.dataset import build_gene_universe

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "NR", 2),
    ]
    universe = build_gene_universe(samples)
    assert universe.names == ("G2", "OFF_UNIVERSE", "G1", "G3")


def test_records_from_h5ad_dir_reads_metadata(tmp_path):
    from src.train.dataset import records_from_h5ad_dir

    (tmp_path / "s1.h5ad").write_bytes(b"")
    (tmp_path / "s1.metadata.csv").write_text(
        "Cancer type,patient_id,sample_id,dataset_id,Response,Tissue,ICI_phase\n"
        "BCC,p1,s1,StudyA,R,Tumor,pre\n"
    )
    records = records_from_h5ad_dir(tmp_path)
    assert len(records) == 1
    assert records[0].label == "R"
    assert records[0].patient_id == "p1"
    assert records[0].dataset_id == "StudyA"


def test_dataset_whole_sample_uses_every_cell_and_shares_builder():
    from src.data.sampler import CellSampler
    from src.graph.build_local_graph import build_local_graph
    from src.train.dataset import SampleGraphDataset

    rng = np.random.default_rng(0)
    matrix = rng.random((6, 4)).astype(np.float32) + 0.1
    sample = AnnDataSample(
        sample_id="S1",
        patient_id="P1",
        X=matrix,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        hvg_names=("G1", "G2", "G3"),
        cell_annotation={},
        clinical_metadata={},
        label="R",
        n_hvg=3,
    )
    universe = GeneUniverse(["G1", "G2", "G3", "G4"])
    whole = SampleGraphDataset(
        [sample],
        sampler=CellSampler(num_cells=2, seed=0),
        gene_universe=universe,
        sampling_mode="whole_sample",
        training=True,
    )
    random = SampleGraphDataset(
        [sample],
        sampler=CellSampler(num_cells=2, seed=0),
        gene_universe=universe,
        sampling_mode="random",
        training=False,
    )
    whole_graph = whole[0]
    random_graph = random[0]
    assert len(whole) == 1
    assert whole_graph.sample_id == "S1"
    assert int(whole_graph["cell"].num_nodes) == 6
    assert np.array_equal(whole_graph["cell"].source_index.numpy(), np.arange(6))
    assert int(random_graph["cell"].num_nodes) == 2
    assert whole_graph.node_types == random_graph.node_types == ["cell", "gene"]
    assert set(whole_graph.edge_types) == set(random_graph.edge_types)
    assert ("cell", "expresses", "gene") in whole_graph.edge_types
    assert ("gene", "expressed_by", "cell") in whole_graph.edge_types
    assert "sample" not in whole_graph.node_types
    assert whole_graph.y.tolist() == [1.0]
    again = whole[0]
    assert np.array_equal(again["cell"].source_index.numpy(), np.arange(6))

    calls: list[int] = []
    real_builder = build_local_graph

    def wrapped(sample_arg, cell_indices, *args, **kwargs):
        calls.append(len(np.asarray(cell_indices)))
        return real_builder(sample_arg, cell_indices, *args, **kwargs)

    import src.train.dataset as dataset_mod

    original = dataset_mod.build_local_graph
    dataset_mod.build_local_graph = wrapped
    try:
        whole[0]
        random[0]
    finally:
        dataset_mod.build_local_graph = original
    assert calls == [6, 2]


def test_dataset_proportional_sampling_builds_a_graph():
    from src.data.sampler import CellSampler
    from src.train.dataset import SampleGraphDataset

    rng = np.random.default_rng(0)
    matrix = rng.random((6, 4)).astype(np.float32) + 0.1
    sample = AnnDataSample(
        sample_id="S1",
        patient_id="P1",
        X=matrix,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        hvg_names=("G1", "G2", "G3"),
        cell_annotation={
            "predicted_labels": np.array(["T", "T", "T", "T", "B", "B"]),
            "conf_score": np.linspace(0.2, 0.9, 6),
        },
        clinical_metadata={},
        label="R",
        n_hvg=3,
    )
    dataset = SampleGraphDataset(
        [sample],
        sampler=CellSampler(num_cells=4, seed=0),
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        sampling_mode="proportional",
        training=False,
    )
    graph = dataset[0]
    assert int(graph["cell"].num_nodes) == 4
    assert graph.y.tolist() == [1.0]


def test_dataset_graphs_use_sample_level_hvgs():
    from src.data.sampler import CellSampler
    from src.train.dataset import SampleGraphDataset

    first = _toy_sample("S1", "P1", "R", 1)
    rng = np.random.default_rng(2)
    second = AnnDataSample(
        sample_id="S2",
        patient_id="P2",
        X=rng.random((6, 4)).astype(np.float32) + 0.1,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        hvg_names=("G1",),
        cell_annotation={},
        clinical_metadata={},
        label="NR",
        n_hvg=1,
    )
    dataset = SampleGraphDataset(
        [first, second],
        sampler=CellSampler(num_cells=4, seed=0),
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        training=False,
    )
    genes = {graph.sample_id: set(graph["gene"].global_id.tolist()) for graph in (dataset[0], dataset[1])}
    assert genes["S2"] <= genes["S1"]
    assert genes["S2"] == {0}
    assert dataset.frozen_hvgs() == {"S1": ("G1", "G2", "G3"), "S2": ("G1",)}


def test_dataset_applies_frozen_train_hvgs_to_every_sample():
    from src.data.sampler import CellSampler
    from src.train.dataset import SampleGraphDataset

    first = _toy_sample("S1", "P1", "R", 1)
    second = _toy_sample("S2", "P2", "NR", 2)
    dataset = SampleGraphDataset(
        [first, second],
        sampler=CellSampler(num_cells=4, seed=0),
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        hvg_names=("G1",),
        training=False,
    )
    assert dataset.frozen_hvgs() == {"S1": ("G1",), "S2": ("G1",)}
    for graph in (dataset[0], dataset[1]):
        assert graph["gene"].global_id.tolist() == [0]


def _annotated_toy_sample(sample_id: str, patient_id: str, label: str, seed: int) -> AnnDataSample:
    rng = np.random.default_rng(seed)
    matrix = rng.random((6, 4)).astype(np.float32) + 0.1
    return AnnDataSample(
        sample_id=sample_id,
        patient_id=patient_id,
        X=matrix,
        gene_names=("G2", "OFF_UNIVERSE", "G1", "G3"),
        hvg_names=("G1", "G2", "G3"),
        cell_annotation={
            "predicted_labels": np.array(["T", "T", "T", "T", "B", "B"]),
            "conf_score": np.linspace(0.2, 0.9, 6),
        },
        clinical_metadata={},
        label=label,
        n_hvg=3,
    )


def test_dataset_by_cell_type_aggregates_into_one_sample_graph():
    from src.data.sampler import CellSampler
    from src.train.dataset import SampleGraphDataset

    sample = _annotated_toy_sample("S1", "P1", "R", 1)
    dataset = SampleGraphDataset(
        [sample],
        sampler=CellSampler(num_cells=4, seed=0),
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        sampling_mode="by_cell_type",
        training=False,
    )
    assert len(dataset) == 1
    graph = dataset[0]
    assert graph.sample_id == "S1"
    assert graph.y.tolist() == [1.0]
    assert int(graph["cell"].num_nodes) == 6
    assert "sample" not in graph.node_types
    assert set(graph.cell_type.split(",")) == {"B", "T"}
    assert graph["cell"].local_graph.max().item() == 1
    only_t = SampleGraphDataset(
        [sample],
        sampler=CellSampler(num_cells=4, seed=0),
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        sampling_mode="by_cell_type",
        cell_types=("T",),
        training=False,
    )
    assert len(only_t) == 1
    assert only_t[0].cell_type == "T"
    assert int(only_t[0]["cell"].num_nodes) == 4


def test_fit_runs_with_by_cell_type_sampling():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    samples = [
        _annotated_toy_sample("S1", "P1", "R", 1),
        _annotated_toy_sample("S2", "P2", "R", 2),
        _annotated_toy_sample("S3", "P3", "NR", 3),
        _annotated_toy_sample("S4", "P4", "NR", 4),
    ]
    history = fit(
        samples,
        GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            sampling_mode="by_cell_type",
            batch_size=2,
            epochs=1,
            val_fraction=0.5,
            seed=0,
        ),
        log=False,
    )
    assert len(history) == 1
    assert history[0].train["loss"] >= 0.0


def test_train_writes_checkpoints_and_history(tmp_path):
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    result = train(
        samples,
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=2,
            val_fraction=0.5,
            seed=0,
        ),
        output_dir=tmp_path,
        log=False,
    )
    assert len(result.history) == 2
    unique_hvgs = set(result.hvg_by_sample.values())
    assert len(unique_hvgs) == 1
    assert set(result.hvg_by_sample) == {"S1", "S2", "S3", "S4"}
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "split.json").is_file()
    assert (tmp_path / "hvgs.json").is_file()
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "threshold.json").is_file()
    assert (tmp_path / "early_stopping.json").is_file()
    assert 0.0 <= result.threshold <= 1.0
    assert result.output_dir == tmp_path

    from src.train.loop import load_checkpoint, predict

    saved = load_checkpoint(tmp_path)
    assert saved.threshold == result.threshold
    assert saved.gene_universe.names == result.gene_universe.names
    scored = predict(samples, saved, output_dir=tmp_path / "infer", log=False)
    assert (tmp_path / "infer" / "predictions.csv").is_file()
    assert len(scored["rows"]) == 4
    assert {row["pred"] for row in scored["rows"]} <= {"R", "NR"}


def test_train_builds_universe_and_hvgs_from_training_fold_only():
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    result = train(
        samples,
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=1,
            val_fraction=0.5,
            n_hvg=3,
            seed=0,
        ),
        log=False,
    )
    unique_hvgs = set(result.hvg_by_sample.values())
    assert len(unique_hvgs) == 1
    frozen = next(iter(unique_hvgs))
    assert result.gene_universe.names == frozen
    assert len(frozen) <= 3


def test_cli_parse_defaults():
    from src.train.__main__ import parse_args

    args = parse_args([])
    assert args.gene_strategy == "hvg"
    assert args.sampling_mode == "random"
    assert args.cell_type_level == "fine"
    assert args.pooling == "mean"
    assert args.readout == "cell"
    assert args.encoder == "sage"
    assert args.threshold_strategy == "max_f1"
    assert args.test_fraction == 0.0
    assert args.profile is False
    assert args.epochs == 30
    assert args.patience == 5
    assert args.min_delta == 0.005
    assert args.no_early_stopping is False


def test_cli_accepts_whole_sample_and_profile():
    from src.train.__main__ import parse_args

    args = parse_args(["--sampling-mode", "whole_sample", "--profile", "--no-early-stopping"])
    assert args.sampling_mode == "whole_sample"
    assert args.profile is True
    assert args.no_early_stopping is True


def test_attention_pool_normalises_per_graph():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from torch_geometric.nn.pool import global_add_pool
    from torch_geometric.utils import softmax

    from src.train.model import CellAttentionPool

    pool = CellAttentionPool(4)
    x = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1])
    pooled, weights = pool(x, batch)
    assert pooled.shape == (2, 4)
    assert weights.shape == (5,)
    mass = global_add_pool(weights.unsqueeze(-1), batch).reshape(-1)
    assert torch.allclose(mass, torch.ones(2), atol=1e-6)
    scores = pool.gate(x).squeeze(-1)
    assert torch.allclose(weights, softmax(scores, batch))


def test_attention_pool_is_not_uniform_mean():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    from src.train.model import CellAttentionPool

    pool = CellAttentionPool(2)
    with torch.no_grad():
        for param in pool.parameters():
            param.zero_()
        pool.gate[0].weight[0, 0] = 1.0
        pool.gate[2].weight[0, 0] = 5.0
    x = torch.tensor([[2.0, 0.0], [0.0, 0.0], [-2.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    pooled, weights = pool(x, batch)
    assert weights[0] > weights[1] > weights[2]
    assert not torch.allclose(pooled.squeeze(0), x.mean(0))


def test_classifier_encoders_return_one_logit():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    from src.graph.build_local_graph import build_local_graph
    from src.train.model import SampleGraphClassifier

    sample = _toy_sample("S1", "P1", "R", 1)
    graph = build_local_graph(
        sample, [0, 1, 2], GeneUniverse(["G1", "G2", "G3", "G4"]), gene_strategy="hvg"
    )
    for encoder in ("placeholder", "sage", "gat"):
        model = SampleGraphClassifier(4, hidden_dim=8, encoder=encoder, num_layers=2)
        logit = model(graph)
        assert logit.shape == (1,), encoder
        cell_x, gene_x, cell_batch, gene_batch = model.encode(graph)
        assert cell_x.shape[0] == graph["cell"].num_nodes
        assert gene_x.shape[0] == graph["gene"].num_nodes
        assert cell_batch.shape[0] == cell_x.shape[0]
        assert gene_batch.shape[0] == gene_x.shape[0]


def test_classifier_readout_both_uses_gene_states():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from src.graph.build_local_graph import build_local_graph
    from src.train.model import SampleGraphClassifier

    sample = _toy_sample("S1", "P1", "R", 1)
    graph = build_local_graph(
        sample, [0, 1, 2], GeneUniverse(["G1", "G2", "G3", "G4"]), gene_strategy="hvg"
    )
    model = SampleGraphClassifier(4, hidden_dim=8, encoder="sage", readout="both")
    logit = model(graph)
    assert logit.shape == (1,)
    assert model.head[0].in_features == 16
    _, gene_x, _, gene_batch = model.encode(graph)
    pooled, _ = model.pool_genes(gene_x, gene_batch)
    assert pooled.shape == (1, 8)


def test_classifier_on_global_batch_returns_one_logit_per_sample():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from torch_geometric.data import Batch

    from src.graph.build_local_graph import build_local_graph
    from src.train.model import SampleGraphClassifier

    graphs = []
    for sample_id, patient_id, label, seed in (
        ("S1", "P1", "R", 1),
        ("S2", "P2", "NR", 2),
    ):
        graphs.append(
            build_local_graph(
                _toy_sample(sample_id, patient_id, label, seed),
                [0, 1, 2],
                GeneUniverse(["G1", "G2", "G3", "G4"]),
                gene_strategy="hvg",
            )
        )
    batch = Batch.from_data_list(graphs)
    for encoder in ("placeholder", "sage"):
        model = SampleGraphClassifier(4, hidden_dim=8, encoder=encoder, readout="both")
        logit = model(batch)
        assert logit.shape == (2,), encoder
        assert batch.y.shape[0] == 2


def test_fit_runs_with_sage_encoder():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    history = fit(
        samples,
        GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            encoder="sage",
            pooling="attention",
            batch_size=2,
            epochs=1,
            val_fraction=0.5,
            seed=0,
        ),
        log=False,
    )
    assert len(history) == 1
    assert history[0].train["loss"] >= 0.0


def test_classifier_attention_pooling_returns_one_logit():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    from src.graph.build_local_graph import build_local_graph
    from src.train.model import SampleGraphClassifier

    sample = _toy_sample("S1", "P1", "R", 1)
    graph = build_local_graph(
        sample, [0, 1, 2], GeneUniverse(["G1", "G2", "G3", "G4"]), gene_strategy="hvg"
    )
    model = SampleGraphClassifier(4, hidden_dim=8, pooling="attention", encoder="placeholder")
    logit = model(graph)
    assert logit.shape == (1,)
    cell_x, batch = model.encode_cells(graph)
    _, weights = model.pool_cells(cell_x, batch)
    assert weights is not None
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_fit_runs_one_epoch_and_reports_metrics():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    history = fit(
        samples,
        GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=1,
            val_fraction=0.5,
            seed=0,
        ),
        log=False,
    )
    assert len(history) == 1
    for split_metrics in (history[0].train, history[0].val):
        assert set(split_metrics) >= {"acc", "auroc", "auprc", "f1", "loss"}
        assert 0.0 <= split_metrics["acc"] <= 1.0
        assert 0.0 <= split_metrics["f1"] <= 1.0
        assert split_metrics["loss"] >= 0.0


def test_whole_sample_batch_returns_one_logit_per_sample():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader

    from src.data.sampler import CellSampler
    from src.graph.build_local_graph import per_sample_graph_stats
    from src.train.dataset import SampleGraphDataset
    from src.train.loop import describe_graph_batch
    from src.train.model import SampleGraphClassifier

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "NR", 2),
    ]
    dataset = SampleGraphDataset(
        samples,
        sampler=CellSampler(num_cells=2, seed=0),
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        sampling_mode="whole_sample",
        training=False,
    )
    graphs = [dataset[0], dataset[1]]
    assert [int(graph["cell"].num_nodes) for graph in graphs] == [6, 6]
    batch = Batch.from_data_list(graphs)
    model = SampleGraphClassifier(4, hidden_dim=8, encoder="sage")
    logit = model(batch)
    assert logit.shape == (2,)
    assert batch.y.shape[0] == 2
    cell_counts = batch["cell"].batch.bincount().tolist()
    assert cell_counts == [6, 6]
    stats = per_sample_graph_stats(batch)
    assert [row["sample_id"] for row in stats] == ["S1", "S2"]
    assert [row["num_cells"] for row in stats] == [6, 6]
    assert stats[0]["num_reverse_edges"] == stats[0]["num_expression_edges"]
    message = describe_graph_batch(batch)
    assert "S1" in message and "S2" in message
    assert "cells=6" in message

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batched = next(iter(loader))
    assert model(batched).shape == (2,)


def test_fit_and_train_hvgs_with_whole_sample(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    history = fit(
        samples,
        GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=2,
            sampling_mode="whole_sample",
            batch_size=2,
            epochs=1,
            val_fraction=0.5,
            seed=0,
            profile=True,
        ),
        log=False,
    )
    assert len(history) == 1
    assert history[0].train["loss"] >= 0.0

    result = train(
        samples,
        config=TrainConfig(
            hidden_dim=8,
            num_cells=2,
            sampling_mode="whole_sample",
            batch_size=2,
            epochs=1,
            val_fraction=0.5,
            n_hvg=3,
            seed=0,
            profile=True,
            device="cpu",
        ),
        output_dir=tmp_path,
        log=False,
    )
    unique_hvgs = set(result.hvg_by_sample.values())
    assert len(unique_hvgs) == 1
    frozen = next(iter(unique_hvgs))
    assert result.gene_universe.names == frozen
    assert len(frozen) <= 3
    profile_path = tmp_path / "graph_profile.csv"
    assert profile_path.is_file()
    text = profile_path.read_text()
    assert "num_cells" in text
    assert "num_reverse_edges" in text
    assert "forward_ms" in text
    assert "S1" in text or "S2" in text or "S3" in text or "S4" in text


def test_auprc_improved_ignores_nan_and_min_delta():
    from src.train.loop import auprc_improved

    assert auprc_improved(0.50, float("-inf"), 0.005)
    assert not auprc_improved(float("nan"), float("-inf"), 0.005)
    assert not auprc_improved(float("nan"), 0.50, 0.005)
    assert not auprc_improved(0.504, 0.50, 0.005)
    assert not auprc_improved(0.505, 0.50, 0.005)
    assert auprc_improved(0.506, 0.50, 0.005)
    assert not auprc_improved(0.50, 0.50, 0.005)


def _scripted_val_auprc(monkeypatch, values: list[float]):
    import src.train.loop as loop_mod

    real = loop_mod.run_epoch
    val_states: list[dict] = []

    def fake(model, loader, *, optimizer=None, **kwargs):
        metrics = real(model, loader, optimizer=optimizer, **kwargs)
        if optimizer is None:
            metrics = dict(metrics)
            metrics["auprc"] = float(values[len(val_states)])
            val_states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        return metrics

    monkeypatch.setattr(loop_mod, "run_epoch", fake)
    return val_states


def test_early_stopping_patience_and_best_checkpoint_reload(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import json

    import torch

    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    val_auprcs = [0.20, 0.50, 0.501, 0.502, 0.503, 0.504, 0.504]
    states = _scripted_val_auprc(monkeypatch, val_auprcs)
    result = train(
        samples,
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=20,
            patience=5,
            min_delta=0.005,
            val_fraction=0.5,
            seed=0,
            device="cpu",
            hvg_names=("G1", "G2", "G3"),
        ),
        output_dir=tmp_path,
        log=False,
    )
    assert [row.val["auprc"] for row in result.history] == val_auprcs
    assert [row.train["loss"] for row in result.history]
    assert all(row.train["loss"] >= 0.0 for row in result.history)
    assert result.best_epoch == 1
    assert result.stop_epoch == 6
    assert result.stopped_early is True
    assert result.best_val_auprc == pytest.approx(0.50)
    assert len(result.history) == 7
    for key, value in states[1].items():
        assert torch.equal(result.model.state_dict()[key].cpu(), value)
    payload = json.loads((tmp_path / "early_stopping.json").read_text())
    assert payload == {
        "best_epoch": 1,
        "best_val_auprc": pytest.approx(0.50),
        "stop_epoch": 6,
        "early_stopping": True,
        "patience": 5,
        "min_delta": 0.005,
        "stopped_early": True,
        "max_epochs": 20,
    }
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 6
    assert best["epoch"] == 1
    history_text = (tmp_path / "history.csv").read_text()
    assert history_text.splitlines()[0] == "epoch,split,acc,auroc,auprc,f1,loss"
    assert history_text.count(",train,") == 7
    assert history_text.count(",val,") == 7


def test_early_stopping_nan_auprc_is_not_improvement(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    values = [float("nan"), float("nan"), 0.40, 0.401, 0.402]
    _scripted_val_auprc(monkeypatch, values)
    result = train(
        samples,
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=5,
            patience=5,
            min_delta=0.005,
            val_fraction=0.5,
            seed=0,
            device="cpu",
            hvg_names=("G1", "G2", "G3"),
        ),
        log=False,
    )
    assert result.best_epoch == 2
    assert result.best_val_auprc == pytest.approx(0.40)
    assert result.stop_epoch == 4
    assert result.stopped_early is False
    assert len(result.history) == 5


def test_reaches_max_epochs_without_early_stopping(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    _scripted_val_auprc(monkeypatch, [0.20, 0.40, 0.60])
    result = train(
        samples,
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=3,
            patience=5,
            min_delta=0.005,
            val_fraction=0.5,
            seed=0,
            device="cpu",
            hvg_names=("G1", "G2", "G3"),
        ),
        log=False,
    )
    assert result.stopped_early is False
    assert result.stop_epoch == 2
    assert result.best_epoch == 2
    assert result.best_val_auprc == pytest.approx(0.60)
    assert len(result.history) == 3


def test_no_early_stopping_flag_runs_all_epochs(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    values = [0.20, 0.50, 0.501, 0.502, 0.503, 0.504, 0.504]
    _scripted_val_auprc(monkeypatch, values)
    result = train(
        samples,
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=7,
            early_stopping=False,
            patience=5,
            min_delta=0.005,
            val_fraction=0.5,
            seed=0,
            device="cpu",
            hvg_names=("G1", "G2", "G3"),
        ),
        log=False,
    )
    assert result.stopped_early is False
    assert result.stop_epoch == 6
    assert result.best_epoch == 1
    assert result.best_val_auprc == pytest.approx(0.50)
    assert len(result.history) == 7
    assert [row.val["auprc"] for row in result.history] == values


def test_no_early_stopping_flag_runs_all_epochs(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train.loop import train

    samples = [
        _toy_sample("S1", "P1", "R", 1),
        _toy_sample("S2", "P2", "R", 2),
        _toy_sample("S3", "P3", "NR", 3),
        _toy_sample("S4", "P4", "NR", 4),
    ]
    values = [0.20, 0.50, 0.501, 0.502, 0.503, 0.504, 0.504]
    _scripted_val_auprc(monkeypatch, values)
    result = train(
        samples,
        gene_universe=GeneUniverse(["G1", "G2", "G3", "G4"]),
        config=TrainConfig(
            hidden_dim=8,
            num_cells=4,
            batch_size=2,
            epochs=7,
            early_stopping=False,
            patience=5,
            min_delta=0.005,
            val_fraction=0.5,
            seed=0,
            device="cpu",
            hvg_names=("G1", "G2", "G3"),
        ),
        log=False,
    )
    assert result.stopped_early is False
    assert result.stop_epoch == 6
    assert result.best_epoch == 1
    assert result.best_val_auprc == pytest.approx(0.50)
    assert len(result.history) == 7
    assert [row.val["auprc"] for row in result.history] == values


