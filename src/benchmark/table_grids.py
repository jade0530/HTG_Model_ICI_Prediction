"""Hyperparameter grids for the split-8173 benchmark.

The legacy grids in the individual benchmark modules are unchanged.
These grids are the table values: random forest, logistic regression,
the cell-KNN GNN, and the MLP. Selection still uses mean validation
AUPRC, with the lower AUPRC standard deviation as the tie break.
"""

from __future__ import annotations

from itertools import product

SPLIT_SEED = 8173
TUNING_SEEDS = (42, 43, 44)
EVAL_SEEDS = (45, 46, 47, 48, 49, 50, 51, 52, 53, 54)
MODELS = ("random_forest", "logistic_regression", "gnn", "mlp")


def random_forest_grid() -> list[dict]:
    configs = []
    for n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features in product(
        (100, 300, 500),
        (None, 10, 20, 30),
        (2, 5, 10),
        (1, 2, 4),
        ("sqrt", "log2"),
    ):
        configs.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
        )
    return configs


def logistic_regression_grid() -> list[dict]:
    return [
        {"C": c_value, "penalty": penalty, "solver": solver}
        for c_value, penalty, solver in product(
            (0.01, 0.1, 1.0, 10.0),
            ("l1", "l2"),
            ("liblinear", "saga"),
        )
    ]


def gnn_grid() -> list[dict]:
    configs = []
    for k, hidden_dim, num_layers, dropout, learning_rate, weight_decay in product(
        (5, 10),
        (32, 128),
        (2, 4),
        (0.0, 0.3),
        (1e-3, 5e-4),
        (0.0, 1e-4, 1e-3),
    ):
        configs.append(
            {
                "k": k,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "encoder": "sage",
                "pooling": "mean",
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    return configs


def mlp_grid() -> list[dict]:
    return [
        {
            "hidden_layer_sizes": hidden,
            "alpha": alpha,
            "learning_rate_init": learning_rate,
            "activation": activation,
        }
        for hidden, alpha, learning_rate, activation in product(
            ((64,), (128,), (64, 32), (128, 64)),
            (1e-4, 1e-3, 1e-2),
            (1e-3, 5e-4),
            ("relu", "tanh"),
        )
    ]


def parameter_grid(model: str) -> list[dict]:
    grids = {
        "random_forest": random_forest_grid,
        "logistic_regression": logistic_regression_grid,
        "gnn": gnn_grid,
        "mlp": mlp_grid,
    }
    if model not in grids:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    return grids[model]()


def json_ready(params: dict) -> dict:
    ready = {}
    for key, value in params.items():
        ready[key] = list(value) if isinstance(value, tuple) else value
    return ready
