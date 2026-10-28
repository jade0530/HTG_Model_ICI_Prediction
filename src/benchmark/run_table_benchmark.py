"""Split-8173 search, then the shared 10-seed evaluation.

Search uses training seeds 42, 43, and 44. The saved configuration is the
one with the highest mean validation AUPRC. The standard deviation is the
population standard deviation, matching the earlier repeated benchmark, and
a lower value breaks ties.

``evaluate`` retrains that configuration on seeds 45 through 54. Those ten
seeds are the same for every architecture. Accuracy and F1 in the evaluation
table come from the saved validation report, whose cutoff maximises F1.
AUPRC and AUROC do not use that cutoff. The seed with the highest validation
AUPRC is recorded as the unseen-cohort checkpoint. This script does not
score the unseen cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from benchmark.table_grids import (
    EVAL_SEEDS,
    MODELS,
    TUNING_SEEDS,
    json_ready,
    parameter_grid,
)
from src.train.dataset import DEFAULT_MANIFEST

SEARCH_METRICS = ("auprc", "auroc", "f1", "acc", "loss")
EVAL_METRICS = ("auprc", "auroc", "acc", "f1")
SCRIPTS = {
    "random_forest": "benchmark_random_forest.py",
    "logistic_regression": "benchmark_logistic_regression.py",
    "mlp": "benchmark_mlp.py",
    "gnn": "benchmark_gnn.py",
}


def aggregate_search(configs: list[dict], raw_rows: list[dict]) -> list[dict]:
    rows = []
    for config_index, params in enumerate(configs, start=1):
        config_rows = [row for row in raw_rows if row["config"] == config_index]
        aggregate = {
            "config": config_index,
            "params": json.dumps(json_ready(params), sort_keys=True),
            "n_runs": len(config_rows),
        }
        for metric in SEARCH_METRICS:
            values = np.asarray([row[f"val_{metric}"] for row in config_rows], dtype=np.float64)
            aggregate[f"val_{metric}_mean"] = float(np.mean(values))
            aggregate[f"val_{metric}_std"] = float(np.std(values))
        rows.append(aggregate)
    rows.sort(key=lambda row: (-row["val_auprc_mean"], row["val_auprc_std"], row["config"]))
    return rows


def summarise_metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else float("nan"),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": float(len(array)),
    }


def choose_best_seed(per_run: list[dict]) -> dict:
    return sorted(
        per_run,
        key=lambda row: (-float(row["auprc"]), -float(row["auroc"]), int(row["training_seed"])),
    )[0]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_val_metrics(run_dir: Path) -> dict[str, float]:
    with (run_dir / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    val = next(row for row in rows if row["split"] == "val")
    return {metric: float(val[metric]) for metric in EVAL_METRICS}


def plot_distribution(per_run: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [[float(row[metric]) for row in per_run] for metric in EVAL_METRICS]
    ax.boxplot(series, tick_labels=list(EVAL_METRICS), showmeans=True)
    ax.set_ylabel("validation score")
    ax.set_title("Ten-seed validation distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _script(model: str) -> Path:
    return Path(__file__).with_name(SCRIPTS[model])


def _common_command(args, output_dir: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(_script(args.model)),
        "--dataset-root",
        str(args.dataset_root),
        "--manifest",
        str(args.manifest),
        "--tissue",
        args.tissue,
        "--ici-phase",
        args.ici_phase,
        "--split-json",
        str(args.split_json),
        "--output-dir",
        str(output_dir),
        "--n-hvg",
        str(args.n_hvg),
        "--seed",
        str(seed),
    ]


def _model_flags(args) -> list[str]:
    flags: list[str] = []
    if args.model in ("mlp", "gnn"):
        flags.extend(["--epochs", str(args.epochs)])
    if args.model == "gnn":
        flags.extend(
            [
                "--num-cells",
                str(args.num_cells),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
        )
    return flags


def _check_split(path: Path) -> int:
    payload = json.loads(path.read_text())
    if "split_seed" not in payload:
        raise SystemExit(f"{path} has no split_seed")
    return int(payload["split_seed"])


def run_search(args) -> None:
    split_seed = _check_split(args.split_json)
    configs = parameter_grid(args.model)
    config_dir = args.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for config_index, params in enumerate(configs, start=1):
        (config_dir / f"config_{config_index:03d}.json").write_text(
            json.dumps(json_ready(params), indent=2) + "\n"
        )
    raw_rows: list[dict] = []
    for training_seed in args.training_seeds:
        run_dir = args.output_dir / "search_runs" / f"seed_{training_seed}"
        command = _common_command(args, run_dir, training_seed)
        command.extend(["--search-grid", "table", "--search-only", *_model_flags(args)])
        subprocess.run(command, check=True)
        with (run_dir / "hyperparam_search.csv").open(newline="") as handle:
            search_rows = list(csv.DictReader(handle))
        if len(search_rows) != len(configs):
            raise SystemExit(
                f"{run_dir} wrote {len(search_rows)} rows; the {args.model} grid has {len(configs)}"
            )
        for config_index, (params, search_row) in enumerate(zip(configs, search_rows, strict=True), start=1):
            raw_rows.append(
                {
                    "config": config_index,
                    "params": json.dumps(json_ready(params), sort_keys=True),
                    "split_seed": split_seed,
                    "training_seed": training_seed,
                    **{f"val_{key}": float(search_row[f"val_{key}"]) for key in SEARCH_METRICS},
                }
            )
    _write_csv(args.output_dir / "raw_results.csv", raw_rows)
    aggregate_rows = aggregate_search(configs, raw_rows)
    _write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    selected_index = int(aggregate_rows[0]["config"])
    selected = json_ready(configs[selected_index - 1])
    selected_path = args.output_dir / "selected_params.json"
    selected_path.write_text(json.dumps(selected, indent=2) + "\n")
    print(
        f"selected_config={selected_index} "
        f"mean_val_auprc={aggregate_rows[0]['val_auprc_mean']:.6f} "
        f"std_val_auprc={aggregate_rows[0]['val_auprc_std']:.6f} "
        f"params={selected_path}"
    )


def run_evaluate(args) -> None:
    split_seed = _check_split(args.split_json)
    selected_path = args.output_dir / "selected_params.json"
    if not selected_path.is_file():
        raise SystemExit(f"missing {selected_path}; run --stage search first")
    per_run = []
    for training_seed in args.eval_seeds:
        run_dir = args.output_dir / "eval_runs" / f"seed_{training_seed}"
        command = _common_command(args, run_dir, training_seed)
        command.extend(["--params-json", str(selected_path), *_model_flags(args)])
        subprocess.run(command, check=True)
        metrics = _read_val_metrics(run_dir)
        per_run.append({"training_seed": training_seed, "checkpoint": str(run_dir), **metrics})
    _write_csv(args.output_dir / "eval_per_run.csv", per_run)
    summary = [{"metric": metric, **summarise_metric([float(row[metric]) for row in per_run])} for metric in EVAL_METRICS]
    _write_csv(args.output_dir / "eval_summary.csv", summary)
    plot_distribution(per_run, args.output_dir / "eval_distribution.png")
    best = choose_best_seed(per_run)
    (args.output_dir / "best_seed.json").write_text(
        json.dumps(
            {
                "split_seed": split_seed,
                "training_seed": int(best["training_seed"]),
                "checkpoint": best["checkpoint"],
                "selection": "highest validation AUPRC, then AUROC, then the smaller seed",
                **{metric: float(best[metric]) for metric in EVAL_METRICS},
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"best_seed={best['training_seed']} val_auprc={float(best['auprc']):.6f} "
        f"checkpoint={best['checkpoint']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "evaluate"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-cells", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=list(TUNING_SEEDS))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "search":
        run_search(args)
        return
    run_evaluate(args)


if __name__ == "__main__":
    main()
