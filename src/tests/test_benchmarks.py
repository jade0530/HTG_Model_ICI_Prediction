from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.train.benchmark_data import (
    align_cell_matrix,
    align_mean_vector,
    assign_split,
    score_metrics,
    write_predict_report,
)
from src.train.dataset import SampleRecord


def _record(sample_id: str, label: str) -> SampleRecord:
    return SampleRecord(
        h5ad_path=Path(f"{sample_id}.h5ad"),
        metadata_path=None,
        patient_id=sample_id,
        sample_id=sample_id,
        dataset_id="D",
        label=label,
        tissue="Tumor",
        ici_phase="pre",
        cancer_type="",
        output_file=f"{sample_id}.h5ad",
    )


def test_assign_split_reads_sample_ids(tmp_path: Path) -> None:
    records = [_record("a", "R"), _record("b", "NR"), _record("c", "R")]
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": [{"sample_id": "a", "patient_key": "D::a", "label": "R"}],
                "val": [{"sample_id": "b", "patient_key": "D::b", "label": "NR"}],
                "test": [{"sample_id": "c", "patient_key": "D::c", "label": "R"}],
            }
        )
    )
    folds = assign_split(records, split_json=split_path)
    assert [item.sample_id for item in folds["train"]] == ["a"]
    assert [item.sample_id for item in folds["val"]] == ["b"]
    assert [item.sample_id for item in folds["test"]] == ["c"]


def test_assign_split_reads_id_lists(tmp_path: Path) -> None:
    records = [_record("a", "R"), _record("b", "NR")]
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    train_ids.write_text("a\n")
    val_ids.write_text("b\n")
    folds = assign_split(records, train_ids=train_ids, val_ids=val_ids)
    assert folds["train"][0].sample_id == "a"
    assert folds["val"][0].sample_id == "b"
    assert folds["test"] == []


def test_score_metrics_include_htg_keys() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = score_metrics(y, scores, threshold=0.5, loss=0.2)
    assert set(metrics) >= {"acc", "auroc", "auprc", "f1", "loss"}
    assert metrics["loss"] == 0.2
    assert metrics["acc"] == 1.0


def test_linear_sgd_writes_epoch_history() -> None:
    import benchmark_linear_regression as lin

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 8))
    y_train = np.array([0] * 10 + [1] * 10)
    X_val = rng.normal(size=(8, 8))
    y_val = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    model = lin.make_model({"alpha": 0.01, "eta0": 0.01, "penalty": "l2", "l1_ratio": 0.15}, seed=0)
    history = lin.run_epochs(model, X_train, y_train, X_val, y_val, epochs=3, seed=0)
    assert [row.epoch for row in history] == [0, 1, 2]
    assert set(history[-1].val) >= {"acc", "auroc", "auprc", "f1", "loss"}


def test_random_forest_grows_tree_history() -> None:
    import benchmark_random_forest as rf

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 8))
    y_train = np.array([0] * 10 + [1] * 10)
    X_val = rng.normal(size=(8, 8))
    y_val = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    model, history = rf.fit_history(
        {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
        X_train,
        y_train,
        X_val,
        y_val,
        seed=0,
        tree_step=10,
    )
    assert [row.epoch for row in history] == [10, 20]
    assert model.n_estimators == 20
    assert set(history[-1].train) >= {"acc", "auroc", "auprc", "f1", "loss"}


def test_mlp_writes_epoch_history() -> None:
    import benchmark_mlp as mlp

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 8))
    y_train = np.array([0] * 10 + [1] * 10)
    X_val = rng.normal(size=(8, 8))
    y_val = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    model = mlp.make_model(
        {"hidden_layer_sizes": (8,), "alpha": 1e-3, "learning_rate_init": 1e-2, "activation": "relu"},
        seed=0,
    )
    history = mlp.run_epochs(model, X_train, y_train, X_val, y_val, epochs=3, seed=0)
    assert [row.epoch for row in history] == [0, 1, 2]
    assert set(history[-1].val) >= {"acc", "auroc", "auprc", "f1", "loss"}
    assert model.predict_proba(X_val).shape == (8, 2)


def test_align_mean_vector_maps_overlapping_genes() -> None:
    mean = np.array([1.0, 2.0, 3.0])
    aligned = align_mean_vector(mean, ["gB", "gC", "gE"], ["gA", "gB", "gC", "gD"])
    assert aligned.tolist() == [0.0, 1.0, 2.0, 0.0]


def test_write_predict_report_writes_metrics_and_confusion(tmp_path: Path) -> None:
    records = [_record("a", "NR"), _record("b", "R"), _record("c", "R"), _record("d", "NR")]
    scores = np.array([0.1, 0.8, 0.9, 0.2])
    metrics = write_predict_report(
        tmp_path,
        records,
        scores,
        threshold=0.5,
        checkpoint="model.joblib",
        threshold_strategy="max_f1",
    )
    assert metrics is not None
    assert metrics["acc"] == 1.0
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "confusion_matrix_predict.csv").is_file()
    assert (tmp_path / "confusion_matrix_predict.png").is_file()
    assert (tmp_path / "roc_pr_predict.png").is_file()


def test_random_forest_predict_on_unaligned_h5ad(tmp_path: Path) -> None:
    import anndata as ad
    import benchmark_random_forest as rf
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    genes = ("gA", "gB", "gC", "gD")
    X_train = rng.normal(size=(24, 4))
    y_train = np.array([0] * 12 + [1] * 12)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dump({"model": model, "genes": genes, "threshold": 0.5}, run_dir / "model.joblib")

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    for sample_id, label, values in (
        ("s1", "R", [1.0, 2.0, 9.0]),
        ("s2", "NR", [0.1, 0.2, 0.3]),
    ):
        adata = ad.AnnData(np.array([values, values], dtype=np.float64))
        adata.var_names = ["gB", "gC", "gE"]
        adata.obs["patient_id"] = sample_id
        adata.obs["sample_id"] = sample_id
        adata.obs["dataset_id"] = "D"
        adata.obs["Response"] = label
        adata.write(new_dir / f"{sample_id}.h5ad")

    out = tmp_path / "pred"
    result = rf.predict(run_dir, new_dir, out, manifest=tmp_path / "missing.csv")
    assert (out / "predictions.csv").is_file()
    assert (out / "results.csv").is_file()
    assert (out / "confusion_matrix_predict.csv").is_file()
    assert result["scores"].shape == (2,)


def test_align_cell_matrix_maps_overlapping_genes() -> None:
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    aligned = align_cell_matrix(X, ["gB", "gC", "gE"], ["gA", "gB", "gC", "gD"])
    assert aligned.shape == (2, 4)
    assert aligned[0].tolist() == [0.0, 1.0, 2.0, 0.0]
    assert aligned[1].tolist() == [0.0, 4.0, 5.0, 0.0]


def test_gnn_knn_and_epoch_history() -> None:
    import benchmark_gnn as gnn

    rng = np.random.default_rng(0)
    train_cells = [rng.normal(size=(8, 6)).astype(np.float32) for _ in range(6)]
    val_cells = [rng.normal(size=(8, 6)).astype(np.float32) for _ in range(4)]
    y_train = np.array([0, 0, 0, 1, 1, 1])
    y_val = np.array([0, 0, 1, 1])
    graph = gnn.make_knn_graph(train_cells[0], 1, k=3)
    assert graph.x.shape == (8, 6)
    assert graph.edge_index.size(0) == 2
    assert graph.edge_index.size(1) > 0

    device = gnn.torch.device("cpu")
    model = gnn.make_model(
        6, {"hidden_dim": 8, "num_layers": 2, "dropout": 0.0, "encoder": "sage", "pooling": "mean"}
    ).to(device)
    train_loader = gnn.make_loader(train_cells, y_train, k=3, batch_size=2, shuffle=False)
    val_loader = gnn.make_loader(val_cells, y_val, k=3, batch_size=2, shuffle=False)
    history = gnn.run_epochs(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=2,
        lr=1e-2,
        weight=None,
    )
    assert [row.epoch for row in history] == [0, 1]
    assert set(history[-1].val) >= {"acc", "auroc", "auprc", "f1", "loss"}
    scores = gnn.predict_scores(model, val_loader, device)
    assert scores.shape == (4,)


def test_gnn_predict_on_unaligned_h5ad(tmp_path: Path) -> None:
    import anndata as ad
    import benchmark_gnn as gnn
    from joblib import dump
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    genes = ("gA", "gB", "gC", "gD")
    train_cells = [rng.normal(size=(8, 4)).astype(np.float32) for _ in range(6)]
    y_train = np.array([0, 0, 0, 1, 1, 1])
    scaler = StandardScaler().fit(np.vstack(train_cells))
    train_cells = gnn.scale_cells(train_cells, scaler)
    device = gnn.torch.device("cpu")
    params = {
        "k": 3,
        "hidden_dim": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "encoder": "sage",
        "pooling": "mean",
        "lr": 1e-2,
    }
    model = gnn.make_model(4, params).to(device)
    loader = gnn.make_loader(train_cells, y_train, k=3, batch_size=2, shuffle=False)
    gnn.run_epochs(model, loader, loader, device=device, epochs=1, lr=1e-2, weight=None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gnn.torch.save(model.state_dict(), run_dir / "model.pt")
    dump(
        {
            "genes": genes,
            "scaler": scaler,
            "threshold": 0.5,
            "best_params": params,
            "num_cells": 8,
            "in_dim": 4,
        },
        run_dir / "model.joblib",
    )

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    for sample_id, label in (("s1", "R"), ("s2", "NR")):
        values = rng.normal(size=(8, 3)).astype(np.float32)
        adata = ad.AnnData(values)
        adata.var_names = ["gB", "gC", "gE"]
        adata.obs["patient_id"] = sample_id
        adata.obs["sample_id"] = sample_id
        adata.obs["dataset_id"] = "D"
        adata.obs["Response"] = label
        adata.write(new_dir / f"{sample_id}.h5ad")

    out = tmp_path / "pred"
    result = gnn.predict(run_dir, new_dir, out, manifest=tmp_path / "missing.csv", device="cpu")
    assert (out / "predictions.csv").is_file()
    assert (out / "results.csv").is_file()
    assert (out / "confusion_matrix_predict.csv").is_file()
    assert result["scores"].shape == (2,)
