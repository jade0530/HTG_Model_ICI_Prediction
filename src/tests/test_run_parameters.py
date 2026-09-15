"""Stage-1 parameter-sweep experiments.

Experiment 1 compares hidden_dim 32 vs 64 on each baseline sampling mode
(whole_sample, random, proportional). Every run uses 40 epochs, seed 42,
and --no-early-stopping. All other knobs stay at the ``python -m src.train``
CLI defaults.

Check the job matrix (no training):

    pytest tests/test_run_parameters.py

Launch the six training runs:

    python tests/test_run_parameters.py
    python tests/test_run_parameters.py --dataset-root /path/to/h5ads --output-root /path/to/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import pytest

from src.train.__main__ import main_train, parse_args
from src.train.dataset import REPO_ROOT

EXPERIMENT_1_HIDDEN_DIMS = (32, 64)
EXPERIMENT_1_SAMPLING_MODES = ("whole_sample", "random", "proportional")
EXPERIMENT_1_EPOCHS = 40
EXPERIMENT_1_SEED = 42
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "exp1"


def experiment_1_jobs(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ``(run_name, argv)`` pairs for Experiment 1."""
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    jobs: list[tuple[str, list[str]]] = []
    for hidden_dim in EXPERIMENT_1_HIDDEN_DIMS:
        for sampling_mode in EXPERIMENT_1_SAMPLING_MODES:
            name = f"hidden{hidden_dim}_{sampling_mode}"
            argv = [
                "--hidden-dim",
                str(hidden_dim),
                "--sampling-mode",
                sampling_mode,
                "--epochs",
                str(EXPERIMENT_1_EPOCHS),
                "--seed",
                str(EXPERIMENT_1_SEED),
                "--no-early-stopping",
                "--output-dir",
                str(root / name),
            ]
            if dataset_root is not None:
                argv.extend(["--dataset-root", str(dataset_root)])
            jobs.append((name, argv))
    return jobs


def run_experiment_1(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[str]:
    """Train every Experiment 1 job in order. Returns output directories."""
    outputs: list[str] = []
    for name, argv in experiment_1_jobs(dataset_root=dataset_root, output_root=output_root):
        print(f"\n=== Experiment 1: {name} ===")
        print("python -m src.train " + " ".join(argv))
        main_train(argv)
        outputs.append(str(Path(argv[argv.index("--output-dir") + 1])))
    return outputs


@pytest.mark.parametrize("hidden_dim", EXPERIMENT_1_HIDDEN_DIMS)
@pytest.mark.parametrize("sampling_mode", EXPERIMENT_1_SAMPLING_MODES)
def test_experiment_1_uses_requested_knobs_and_cli_defaults(hidden_dim: int, sampling_mode: str) -> None:
    jobs = dict(experiment_1_jobs())
    name = f"hidden{hidden_dim}_{sampling_mode}"
    assert name in jobs
    args = parse_args(jobs[name])
    assert args.hidden_dim == hidden_dim
    assert args.sampling_mode == sampling_mode
    assert args.epochs == EXPERIMENT_1_EPOCHS
    assert args.no_early_stopping is True
    assert args.encoder == "sage"
    assert args.pooling == "mean"
    assert args.readout == "cell"
    assert args.num_gnn_layers == 2
    assert args.dropout == 0.1
    assert args.gene_strategy == "hvg"
    assert args.num_cells == 256
    assert args.n_hvg == 500
    assert args.batch_size == 2
    assert args.val_fraction == 0.25
    assert args.test_fraction == 0.0
    assert args.lr == 1e-3
    assert args.seed == EXPERIMENT_1_SEED
    assert args.device == "auto"
    assert args.tissue == "Tumor"
    assert args.ici_phase == "pre"


def test_experiment_1_has_six_runs() -> None:
    names = [name for name, _ in experiment_1_jobs()]
    assert names == [
        "hidden32_whole_sample",
        "hidden32_random",
        "hidden32_proportional",
        "hidden64_whole_sample",
        "hidden64_random",
        "hidden64_proportional",
    ]


def _parse_runner_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-1 parameter-sweep experiments")
    parser.add_argument("--experiment", default="1", choices=("1",), help="Which experiment to launch")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Passed through to src.train. Omit to use the CLI default (data/test)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent folder for the six Experiment 1 run directories",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_runner_args()
    if args.experiment == "1":
        run_experiment_1(dataset_root=args.dataset_root, output_root=args.output_root)
