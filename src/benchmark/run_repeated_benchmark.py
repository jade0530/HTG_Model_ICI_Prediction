"""Tune one benchmark over fixed splits and training seeds, then score unseen data once."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.train.dataset import DEFAULT_MANIFEST


def parameter_grid(model: str) -> list[dict]:
    if model == "random_forest":
        from benchmark import benchmark_random_forest

        return [
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "class_weight": class_weight,
            }
            for n_estimators, max_depth, min_samples_leaf, max_features, class_weight in product(
                benchmark_random_forest.N_ESTIMATORS,
                benchmark_random_forest.MAX_DEPTH,
                benchmark_random_forest.MIN_SAMPLES_LEAF,
                benchmark_random_forest.MAX_FEATURES,
                benchmark_random_forest.CLASS_WEIGHT,
            )
        ]
    if model == "linear_regression":
        from benchmark import benchmark_linear_regression

        configs = []
        for alpha, eta0, penalty in product(
            benchmark_linear_regression.ALPHAS,
            benchmark_linear_regression.ETA0S,
            benchmark_linear_regression.PENALTIES,
        ):
            l1_values = (
                benchmark_linear_regression.L1_RATIOS
                if penalty == "elasticnet"
                else (0.15,)
            )
            for l1_ratio in l1_values:
                configs.append(
                    {
                        "alpha": alpha,
                        "eta0": eta0,
                        "penalty": penalty,
                        "l1_ratio": l1_ratio,
                    }
                )
        return configs
    if model == "mlp":
        from benchmark import benchmark_mlp

        return [
            {
                "hidden_layer_sizes": list(hidden),
                "alpha": alpha,
                "learning_rate_init": learning_rate,
                "activation": activation,
            }
            for hidden, alpha, learning_rate, activation in product(
                benchmark_mlp.HIDDEN_LAYERS,
                benchmark_mlp.ALPHAS,
                benchmark_mlp.LEARNING_RATES,
                benchmark_mlp.ACTIVATIONS,
            )
        ]
    from benchmark import benchmark_gnn

    return [
        {
            "k": k,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "encoder": "sage",
            "pooling": "mean",
            "lr": 1e-3,
        }
        for k, hidden_dim, num_layers, dropout in product(
            benchmark_gnn.K_VALUES,
            benchmark_gnn.HIDDEN_DIMS,
            benchmark_gnn.NUM_LAYERS,
            benchmark_gnn.DROPOUTS,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("random_forest", "linear_regression", "mlp", "gnn"),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-cells", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--final-training-seed", type=int, default=42)
    parser.add_argument("--unseen-dataset-root", type=Path, default=None)
    args = parser.parse_args()

    scripts = {
        "random_forest": "benchmark_random_forest.py",
        "linear_regression": "benchmark_linear_regression.py",
        "mlp": "benchmark_mlp.py",
        "gnn": "benchmark_gnn.py",
    }
    script = Path(__file__).with_name(scripts[args.model])
    split_files = sorted(args.split_dir.glob("split_*.json"))
    configs = parameter_grid(args.model)
    config_dir = args.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for config_index, params in enumerate(configs, start=1):
        (config_dir / f"config_{config_index:03d}.json").write_text(
            json.dumps(params, indent=2) + "\n"
        )
    raw_rows: list[dict] = []

    for split_index, split_path in enumerate(split_files, start=1):
        split_seed = json.loads(split_path.read_text())["split_seed"]
        for training_seed in args.training_seeds:
            run_dir = (
                args.output_dir
                / "search_runs"
                / f"split_{split_index:03d}"
                / f"seed_{training_seed}"
            )
            command = [
                sys.executable,
                str(script),
                "--dataset-root",
                str(args.dataset_root),
                "--manifest",
                str(args.manifest),
                "--tissue",
                args.tissue,
                "--ici-phase",
                args.ici_phase,
                "--split-json",
                str(split_path),
                "--output-dir",
                str(run_dir),
                "--n-hvg",
                str(args.n_hvg),
                "--seed",
                str(training_seed),
            ]
            if args.model in ("linear_regression", "mlp", "gnn"):
                command.extend(["--epochs", str(args.epochs)])
            if args.model == "gnn":
                command.extend(
                    [
                        "--num-cells",
                        str(args.num_cells),
                        "--batch-size",
                        str(args.batch_size),
                        "--device",
                        args.device,
                    ]
                )
            subprocess.run(command, check=True)
            with (run_dir / "hyperparam_search.csv").open(newline="") as handle:
                search_rows = list(csv.DictReader(handle))
            for config_index, (params, search_row) in enumerate(
                zip(configs, search_rows, strict=True), start=1
            ):
                raw_rows.append(
                    {
                        "config": config_index,
                        "params": json.dumps(params, sort_keys=True),
                        "split": split_index,
                        "split_seed": split_seed,
                        "training_seed": training_seed,
                        **{
                            f"val_{key}": float(search_row[f"val_{key}"])
                            for key in ("acc", "auroc", "auprc", "f1", "loss")
                        },
                    }
                )

    raw_columns = list(raw_rows[0])
    with (args.output_dir / "raw_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_columns)
        writer.writeheader()
        writer.writerows(raw_rows)

    metric_names = ("auprc", "auroc", "f1", "acc", "loss")
    aggregate_rows = []
    for config_index, params in enumerate(configs, start=1):
        config_rows = [row for row in raw_rows if row["config"] == config_index]
        aggregate = {
            "config": config_index,
            "params": json.dumps(params, sort_keys=True),
            "n_runs": len(config_rows),
        }
        for metric in metric_names:
            values = np.asarray(
                [row[f"val_{metric}"] for row in config_rows], dtype=np.float64
            )
            aggregate[f"val_{metric}_mean"] = float(np.mean(values))
            aggregate[f"val_{metric}_std"] = float(np.std(values))
        aggregate_rows.append(aggregate)

    aggregate_rows.sort(
        key=lambda row: (-row["val_auprc_mean"], row["val_auprc_std"])
    )
    aggregate_columns = list(aggregate_rows[0])
    with (args.output_dir / "aggregate_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    selected_index = int(aggregate_rows[0]["config"])
    selected_params = configs[selected_index - 1]
    selected_path = args.output_dir / "selected_params.json"
    selected_path.write_text(json.dumps(selected_params, indent=2) + "\n")

    final_dir = args.output_dir / "final_model"
    final_command = [
        sys.executable,
        str(script),
        "--dataset-root",
        str(args.dataset_root),
        "--manifest",
        str(args.manifest),
        "--tissue",
        args.tissue,
        "--ici-phase",
        args.ici_phase,
        "--split-json",
        str(split_files[0]),
        "--output-dir",
        str(final_dir),
        "--params-json",
        str(selected_path),
        "--n-hvg",
        str(args.n_hvg),
        "--seed",
        str(args.final_training_seed),
    ]
    if args.model in ("linear_regression", "mlp", "gnn"):
        final_command.extend(["--epochs", str(args.epochs)])
    if args.model == "gnn":
        final_command.extend(
            [
                "--num-cells",
                str(args.num_cells),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
        )
    subprocess.run(final_command, check=True)

    if args.unseen_dataset_root is not None:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "predict",
                "--checkpoint",
                str(final_dir),
                "--dataset-root",
                str(args.unseen_dataset_root),
                "--output-dir",
                str(args.output_dir / "unseen"),
                "--tissue",
                "all",
                "--ici-phase",
                "all",
            ],
            check=True,
        )

    print(
        f"selected_config={selected_index} "
        f"mean_val_auprc={aggregate_rows[0]['val_auprc_mean']:.6f} "
        f"std_val_auprc={aggregate_rows[0]['val_auprc_std']:.6f}"
    )


if __name__ == "__main__":
    main()
