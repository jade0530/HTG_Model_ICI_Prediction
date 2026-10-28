from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from benchmark.run_table_benchmark import aggregate_search, choose_best_seed, summarise_metric
from benchmark.table_grids import EVAL_SEEDS, TUNING_SEEDS, parameter_grid


def test_table_grid_sizes_and_gnn_values() -> None:
    assert [len(parameter_grid(model)) for model in ("random_forest", "logistic_regression", "gnn", "mlp")] == [
        216,
        16,
        96,
        48,
    ]
    forest = parameter_grid("random_forest")
    assert "class_weight" not in forest[0]
    assert forest[0]["min_samples_split"] == 2
    assert {row["n_estimators"] for row in forest} == {100, 300, 500}
    assert {row["max_features"] for row in forest} == {"sqrt", "log2"}

    logistic = parameter_grid("logistic_regression")
    assert {row["penalty"] for row in logistic} == {"l1", "l2"}
    assert {row["solver"] for row in logistic} == {"liblinear", "saga"}

    gnn = parameter_grid("gnn")
    assert {row["k"] for row in gnn} == {5, 10}
    assert {row["hidden_dim"] for row in gnn} == {32, 128}
    assert {row["num_layers"] for row in gnn} == {2, 4}
    assert {row["dropout"] for row in gnn} == {0.0, 0.3}
    assert {row["lr"] for row in gnn} == {1e-3, 5e-4}
    assert {row["weight_decay"] for row in gnn} == {0.0, 1e-4, 1e-3}
    assert {row["encoder"] for row in gnn} == {"sage"}
    assert {row["pooling"] for row in gnn} == {"mean"}

    mlp = parameter_grid("mlp")
    assert {row["learning_rate_init"] for row in mlp} == {1e-3, 5e-4}
    assert len(EVAL_SEEDS) == 10
    assert set(EVAL_SEEDS).isdisjoint(TUNING_SEEDS)


def test_search_selects_highest_mean_auprc_then_lower_std() -> None:
    configs = [{"name": "a"}, {"name": "b"}]
    raw = [
        {"config": 1, "val_auprc": 0.80, "val_auroc": 0.70, "val_f1": 0.60, "val_acc": 0.60, "val_loss": 0.40},
        {"config": 1, "val_auprc": 0.60, "val_auroc": 0.70, "val_f1": 0.60, "val_acc": 0.60, "val_loss": 0.40},
        {"config": 2, "val_auprc": 0.70, "val_auroc": 0.70, "val_f1": 0.60, "val_acc": 0.60, "val_loss": 0.40},
        {"config": 2, "val_auprc": 0.70, "val_auroc": 0.70, "val_f1": 0.60, "val_acc": 0.60, "val_loss": 0.40},
    ]
    ranked = aggregate_search(configs, raw)
    assert ranked[0]["config"] == 2
    assert ranked[0]["val_auprc_mean"] == 0.70
    assert ranked[1]["val_auprc_mean"] == 0.70


def test_eval_summary_and_best_seed() -> None:
    summary = summarise_metric([0.2, 0.4, 0.6])
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["min"] == pytest.approx(0.2)
    assert summary["max"] == pytest.approx(0.6)
    assert summary["n"] == 3
    best = choose_best_seed(
        [
            {"training_seed": 46, "auprc": 0.80, "auroc": 0.70, "acc": 0.60, "f1": 0.50},
            {"training_seed": 45, "auprc": 0.80, "auroc": 0.90, "acc": 0.60, "f1": 0.50},
            {"training_seed": 47, "auprc": 0.70, "auroc": 0.99, "acc": 0.60, "f1": 0.50},
        ]
    )
    assert best["training_seed"] == 45


def test_logistic_regression_grid_fits() -> None:
    from benchmark import benchmark_logistic_regression as logistic

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(24, 6))
    y_train = np.array([0] * 12 + [1] * 12)
    X_val = rng.normal(size=(8, 6))
    y_val = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    params, rows = logistic.tune(X_train, y_train, X_val, y_val, seed=0)
    assert len(rows) == 16
    assert set(params) == {"C", "penalty", "solver"}
    assert set(rows[-1]) >= {"val_acc", "val_auroc", "val_auprc", "val_f1", "val_loss"}
