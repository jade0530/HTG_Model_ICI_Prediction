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
from src.train.metrics import classification_metrics


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
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "split.json").is_file()
    assert result.output_dir == tmp_path


def test_cli_parse_defaults():
    from src.train.__main__ import parse_args

    args = parse_args([])
    assert args.gene_strategy == "hvg"
    assert args.sampling_mode == "random"
    assert args.test_fraction == 0.0


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
        assert set(split_metrics) >= {"acc", "auprc", "f1", "loss"}
        assert 0.0 <= split_metrics["acc"] <= 1.0
        assert 0.0 <= split_metrics["f1"] <= 1.0
        assert split_metrics["loss"] >= 0.0
