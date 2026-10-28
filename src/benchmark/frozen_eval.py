"""Ten-seed evaluation for the four table architectures.

Training seeds 1 through 10 are frozen and shared. Each architecture is
retrained from the hyperparameters selected by its split-8173 search.
Accuracy, AUROC, AUPRC, and F1 are read from the validation report.
Accuracy and F1 use the cutoff that maximises validation F1. AUROC and
AUPRC do not use that cutoff. The reported standard deviation is the
population standard deviation.

This module builds the commands, the metric table, and the AUPRC
distribution. Importing it does not train anything.
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

from benchmark.table_grids import SPLIT_SEED

FROZEN_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
REPORT_MODELS = ("random_forest", "logistic_regression", "mlp", "gnn")
METRICS = ("acc", "auroc", "auprc", "f1")
MODEL_LABELS = {
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
    "mlp": "MLP",
    "gnn": "GNN",
}


def seed_payload(split_seed: int = SPLIT_SEED) -> dict:
    return {
        "seeds": list(FROZEN_SEEDS),
        "role": "training_seed",
        "models": list(REPORT_MODELS),
        "split_seed": int(split_seed),
        "note": (
            "Frozen training seeds shared by random forest, logistic regression, "
            f"MLP, and the cell-KNN GNN. The patient split stays seed {int(split_seed)}."
        ),
    }


def freeze_eval_seeds(path: Path, split_seed: int = SPLIT_SEED) -> dict:
    """Write the shared seed list, or reject a file that differs from it."""
    payload = seed_payload(split_seed)
    path = Path(path)
    if path.is_file():
        found = json.loads(path.read_text())
        if tuple(int(seed) for seed in found["seeds"]) != FROZEN_SEEDS:
            raise ValueError(f"{path} freezes seeds {found['seeds']}; expected {list(FROZEN_SEEDS)}")
        if int(found["split_seed"]) != int(split_seed):
            raise ValueError(f"{path} freezes split_seed {found['split_seed']}; expected {int(split_seed)}")
        return found
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _require_frozen_runs(model: str, runs: list[dict]) -> list[dict]:
    ordered = sorted(runs, key=lambda row: int(row["training_seed"]))
    found = tuple(int(row["training_seed"]) for row in ordered)
    if found != FROZEN_SEEDS:
        raise ValueError(f"{model} has training seeds {found}; expected {FROZEN_SEEDS}")
    return ordered


def _population_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def format_mean_pm_sd(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def metric_rows(model: str, runs: list[dict]) -> list[dict]:
    """Per-seed validation scores, then mean and population SD."""
    ordered = _require_frozen_runs(model, runs)
    rows = [
        {
            "model": model,
            "record": "run",
            "training_seed": int(row["training_seed"]),
            **{metric: float(row[metric]) for metric in METRICS},
        }
        for row in ordered
    ]
    for record, reducer in (("mean", np.mean), ("sd", _population_sd)):
        rows.append(
            {
                "model": model,
                "record": record,
                "training_seed": "",
                **{
                    metric: float(reducer([float(row[metric]) for row in ordered]))
                    for metric in METRICS
                },
            }
        )
    return rows


def summary_row(model: str, runs: list[dict]) -> dict:
    ordered = _require_frozen_runs(model, runs)
    row: dict[str, object] = {"model": model, "n_seeds": len(ordered), "sd": "population"}
    for metric in METRICS:
        values = [float(item[metric]) for item in ordered]
        mean = float(np.mean(values))
        sd = _population_sd(values)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[metric] = format_mean_pm_sd(mean, sd)
    return row


def write_metric_tables(output_dir: Path, by_model: dict[str, list[dict]], split_seed: int = SPLIT_SEED) -> None:
    missing = [model for model in REPORT_MODELS if model not in by_model]
    if missing:
        raise ValueError(f"missing frozen-seed results for {missing}")
    long_rows = [row for model in REPORT_MODELS for row in metric_rows(model, by_model[model])]
    summary_rows = [summary_row(model, by_model[model]) for model in REPORT_MODELS]
    _write_csv(output_dir / "metrics_by_seed.csv", long_rows)
    _write_csv(output_dir / "metrics_summary.csv", summary_rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "split_seed": int(split_seed),
                "training_seeds": list(FROZEN_SEEDS),
                "sd": "population standard deviation (ddof=0)",
                "metrics": list(METRICS),
                "models": summary_rows,
            },
            indent=2,
        )
        + "\n"
    )


def plot_auprc_distribution(by_model: dict[str, list[dict]], path: Path, split_seed: int = SPLIT_SEED) -> None:
    """Box plot of validation AUPRC for seeds 1–10, one box per architecture."""
    import matplotlib.pyplot as plt

    series = []
    labels = []
    for model in REPORT_MODELS:
        ordered = _require_frozen_runs(model, by_model[model])
        series.append([float(row["auprc"]) for row in ordered])
        labels.append(MODEL_LABELS[model])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.boxplot(series, tick_labels=labels, showmeans=True, widths=0.55)
    offsets = np.linspace(-0.15, 0.15, len(FROZEN_SEEDS))
    for index, values in enumerate(series, start=1):
        ax.scatter(np.full(len(values), index) + offsets, values, s=28, zorder=3)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Validation AUPRC")
    ax.set_title("Validation AUPRC across frozen training seeds 1–10")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        f"Selected hyperparameters, split {int(split_seed)} validation fold. Points are seeds 1–10 in order. The triangle marks the mean.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def evaluate_command(
    python: str,
    model: str,
    *,
    dataset_root: Path,
    manifest: Path,
    split_json: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    """One evaluate command. This does not launch it."""
    if model not in REPORT_MODELS:
        raise ValueError(f"frozen evaluation does not include {model}")
    return [
        python,
        "-u",
        "benchmark/run_table_benchmark.py",
        "--stage",
        "evaluate",
        "--model",
        model,
        "--dataset-root",
        str(dataset_root),
        "--manifest",
        str(manifest),
        "--split-json",
        str(split_json),
        "--output-dir",
        str(output_dir),
        "--n-hvg",
        "3000",
        "--epochs",
        "40",
        "--num-cells",
        "128",
        "--batch-size",
        "2",
        "--device",
        device,
        "--eval-seeds",
        *[str(seed) for seed in FROZEN_SEEDS],
    ]


def _selected_params(table_root: Path, model: str) -> Path:
    path = table_root / model / "selected_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; finish the {model} search before the frozen-seed evaluation")
    return path


def _read_eval_runs(model_dir: Path) -> list[dict]:
    with (model_dir / "eval_per_run.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_frozen_evaluation(args: argparse.Namespace) -> None:
    """Retrain the selected configurations on seeds 1–10 and write the report."""
    split_seed = int(json.loads(Path(args.split_json).read_text())["split_seed"])
    freeze_eval_seeds(args.frozen_seeds, split_seed)
    for model in REPORT_MODELS:
        _selected_params(args.table_root, model)
    for model in REPORT_MODELS:
        command = evaluate_command(
            sys.executable,
            model,
            dataset_root=args.dataset_root,
            manifest=args.manifest,
            split_json=args.split_json,
            output_dir=args.table_root / model,
            device=args.device,
        )
        subprocess.run(command, check=True, cwd=_PACKAGE_ROOT)
    by_model = {model: _read_eval_runs(args.table_root / model) for model in REPORT_MODELS}
    write_metric_tables(args.output_dir, by_model, split_seed)
    plot_auprc_distribution(by_model, args.output_dir / "auprc_distribution.png", split_seed)
    print(f"frozen_eval={args.output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-seeds", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_frozen_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
