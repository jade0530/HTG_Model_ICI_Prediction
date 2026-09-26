"""Stage-1 parameter-sweep experiments.

Experiment 1 compares hidden_dim 32 vs 64 on each baseline sampling mode
(whole_sample, random, proportional). Every run uses 40 epochs, seed 42,
and --no-early-stopping. All other knobs stay at the ``python -m src.train``
CLI defaults.

Experiment 2 trains for 100 epochs to inspect overfitting curves. It uses the
CLI default hidden_dim (64), 128 sampled cells, seed 42, and
--no-early-stopping on whole_sample, random, and proportional.

Experiment 3 checks dropout 0 vs 0.3. It uses the CLI default hidden_dim (64),
128 sampled cells, seed 42, and --no-early-stopping on whole_sample, random,
and proportional.

Experiment 4 compares mean vs attention pooling on whole_sample, random, and
proportional. Every run uses 40 epochs, seed 42, and --no-early-stopping.
Random and proportional use 128 sampled cells; whole_sample keeps all cells.

Check the job matrix (no training):

    pytest tests/test_run_parameters.py

Launch a sweep:

    python tests/test_run_parameters.py --experiment 1
    python tests/test_run_parameters.py --experiment 2
    python tests/test_run_parameters.py --experiment 3
    python tests/test_run_parameters.py --experiment 4
    python tests/test_run_parameters.py --experiment 4 --dataset-root /path/to/h5ads --output-root /path/to/results
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

BASELINE_SAMPLING_MODES = ("whole_sample", "random", "proportional")
EXPERIMENT_SEED = 42

EXPERIMENT_1_HIDDEN_DIMS = (32, 64)
EXPERIMENT_1_SAMPLING_MODES = BASELINE_SAMPLING_MODES
EXPERIMENT_1_EPOCHS = 40
EXPERIMENT_1_SEED = EXPERIMENT_SEED
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "exp1"

EXPERIMENT_2_SAMPLING_MODES = BASELINE_SAMPLING_MODES
EXPERIMENT_2_EPOCHS = 100
EXPERIMENT_2_NUM_CELLS = 128
EXPERIMENT_2_SEED = EXPERIMENT_SEED
EXPERIMENT_2_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "exp2"

EXPERIMENT_3_SAMPLING_MODES = BASELINE_SAMPLING_MODES
EXPERIMENT_3_DROPOUTS = (0.0, 0.3)
EXPERIMENT_3_NUM_CELLS = 128
EXPERIMENT_3_SEED = EXPERIMENT_SEED
EXPERIMENT_3_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "exp3"

EXPERIMENT_4_SAMPLING_MODES = BASELINE_SAMPLING_MODES
EXPERIMENT_4_POOLINGS = ("mean", "attention")
EXPERIMENT_4_EPOCHS = 40
EXPERIMENT_4_NUM_CELLS = 128
EXPERIMENT_4_SEED = EXPERIMENT_SEED
EXPERIMENT_4_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "exp4"


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


def experiment_2_jobs(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ``(run_name, argv)`` pairs for Experiment 2 (long overfitting runs)."""
    root = Path(output_root) if output_root is not None else EXPERIMENT_2_OUTPUT_ROOT
    jobs: list[tuple[str, list[str]]] = []
    for sampling_mode in EXPERIMENT_2_SAMPLING_MODES:
        name = f"overfit_100ep_cells128_{sampling_mode}"
        argv = [
            "--sampling-mode",
            sampling_mode,
            "--num-cells",
            str(EXPERIMENT_2_NUM_CELLS),
            "--epochs",
            str(EXPERIMENT_2_EPOCHS),
            "--seed",
            str(EXPERIMENT_2_SEED),
            "--no-early-stopping",
            "--output-dir",
            str(root / name),
        ]
        if dataset_root is not None:
            argv.extend(["--dataset-root", str(dataset_root)])
        jobs.append((name, argv))
    return jobs


def experiment_3_jobs(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ``(run_name, argv)`` pairs for Experiment 3 (dropout 0 vs 0.3)."""
    root = Path(output_root) if output_root is not None else EXPERIMENT_3_OUTPUT_ROOT
    jobs: list[tuple[str, list[str]]] = []
    for dropout in EXPERIMENT_3_DROPOUTS:
        tag = "0" if dropout == 0.0 else "0p3"
        for sampling_mode in EXPERIMENT_3_SAMPLING_MODES:
            name = f"dropout{tag}_cells128_{sampling_mode}"
            argv = [
                "--sampling-mode",
                sampling_mode,
                "--dropout",
                str(dropout),
                "--num-cells",
                str(EXPERIMENT_3_NUM_CELLS),
                "--seed",
                str(EXPERIMENT_3_SEED),
                "--no-early-stopping",
                "--output-dir",
                str(root / name),
            ]
            if dataset_root is not None:
                argv.extend(["--dataset-root", str(dataset_root)])
            jobs.append((name, argv))
    return jobs


def experiment_4_jobs(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ``(run_name, argv)`` pairs for Experiment 4 (mean vs attention pooling)."""
    root = Path(output_root) if output_root is not None else EXPERIMENT_4_OUTPUT_ROOT
    jobs: list[tuple[str, list[str]]] = []
    for pooling in EXPERIMENT_4_POOLINGS:
        for sampling_mode in EXPERIMENT_4_SAMPLING_MODES:
            name = f"pool{pooling}_{sampling_mode}"
            argv = [
                "--sampling-mode",
                sampling_mode,
                "--pooling",
                pooling,
                "--epochs",
                str(EXPERIMENT_4_EPOCHS),
                "--seed",
                str(EXPERIMENT_4_SEED),
                "--no-early-stopping",
                "--output-dir",
                str(root / name),
            ]
            if sampling_mode in ("random", "proportional"):
                argv.extend(["--num-cells", str(EXPERIMENT_4_NUM_CELLS)])
            if dataset_root is not None:
                argv.extend(["--dataset-root", str(dataset_root)])
            jobs.append((name, argv))
    return jobs


def _run_jobs(experiment: str, jobs: list[tuple[str, list[str]]]) -> list[str]:
    outputs: list[str] = []
    for name, argv in jobs:
        print(f"\n=== Experiment {experiment}: {name} ===")
        print("python -m src.train " + " ".join(argv))
        main_train(argv)
        outputs.append(str(Path(argv[argv.index("--output-dir") + 1])))
    return outputs


def run_experiment_1(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[str]:
    """Train every Experiment 1 job in order. Returns output directories."""
    return _run_jobs("1", experiment_1_jobs(dataset_root=dataset_root, output_root=output_root))


def run_experiment_2(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[str]:
    """Train every Experiment 2 job in order. Returns output directories."""
    return _run_jobs("2", experiment_2_jobs(dataset_root=dataset_root, output_root=output_root))


def run_experiment_3(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[str]:
    """Train every Experiment 3 job in order. Returns output directories."""
    return _run_jobs("3", experiment_3_jobs(dataset_root=dataset_root, output_root=output_root))


def run_experiment_4(
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[str]:
    """Train every Experiment 4 job in order. Returns output directories."""
    return _run_jobs("4", experiment_4_jobs(dataset_root=dataset_root, output_root=output_root))


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


@pytest.mark.parametrize("sampling_mode", EXPERIMENT_2_SAMPLING_MODES)
def test_experiment_2_uses_requested_knobs_and_cli_defaults(sampling_mode: str) -> None:
    jobs = dict(experiment_2_jobs())
    name = f"overfit_100ep_cells128_{sampling_mode}"
    assert name in jobs
    args = parse_args(jobs[name])
    assert args.hidden_dim == 64
    assert args.sampling_mode == sampling_mode
    assert args.epochs == EXPERIMENT_2_EPOCHS
    assert args.num_cells == EXPERIMENT_2_NUM_CELLS
    assert args.no_early_stopping is True
    assert args.encoder == "sage"
    assert args.pooling == "mean"
    assert args.readout == "cell"
    assert args.num_gnn_layers == 2
    assert args.dropout == 0.1
    assert args.gene_strategy == "hvg"
    assert args.n_hvg == 500
    assert args.batch_size == 2
    assert args.val_fraction == 0.25
    assert args.test_fraction == 0.0
    assert args.lr == 1e-3
    assert args.seed == EXPERIMENT_2_SEED
    assert args.device == "auto"
    assert args.tissue == "Tumor"
    assert args.ici_phase == "pre"


def test_experiment_2_has_three_runs() -> None:
    names = [name for name, _ in experiment_2_jobs()]
    assert names == [
        "overfit_100ep_cells128_whole_sample",
        "overfit_100ep_cells128_random",
        "overfit_100ep_cells128_proportional",
    ]


@pytest.mark.parametrize("dropout,tag", [(0.0, "0"), (0.3, "0p3")])
@pytest.mark.parametrize("sampling_mode", EXPERIMENT_3_SAMPLING_MODES)
def test_experiment_3_uses_requested_knobs_and_cli_defaults(
    dropout: float, tag: str, sampling_mode: str
) -> None:
    jobs = dict(experiment_3_jobs())
    name = f"dropout{tag}_cells128_{sampling_mode}"
    assert name in jobs
    args = parse_args(jobs[name])
    assert args.hidden_dim == 64
    assert args.sampling_mode == sampling_mode
    assert args.dropout == dropout
    assert args.num_cells == EXPERIMENT_3_NUM_CELLS
    assert args.epochs == 30
    assert args.no_early_stopping is True
    assert args.encoder == "sage"
    assert args.pooling == "mean"
    assert args.readout == "cell"
    assert args.num_gnn_layers == 2
    assert args.gene_strategy == "hvg"
    assert args.n_hvg == 500
    assert args.batch_size == 2
    assert args.val_fraction == 0.25
    assert args.test_fraction == 0.0
    assert args.lr == 1e-3
    assert args.seed == EXPERIMENT_3_SEED
    assert args.device == "auto"
    assert args.tissue == "Tumor"
    assert args.ici_phase == "pre"


def test_experiment_3_has_six_runs() -> None:
    names = [name for name, _ in experiment_3_jobs()]
    assert names == [
        "dropout0_cells128_whole_sample",
        "dropout0_cells128_random",
        "dropout0_cells128_proportional",
        "dropout0p3_cells128_whole_sample",
        "dropout0p3_cells128_random",
        "dropout0p3_cells128_proportional",
    ]


@pytest.mark.parametrize("pooling", EXPERIMENT_4_POOLINGS)
@pytest.mark.parametrize("sampling_mode", EXPERIMENT_4_SAMPLING_MODES)
def test_experiment_4_uses_requested_knobs_and_cli_defaults(
    pooling: str, sampling_mode: str
) -> None:
    jobs = dict(experiment_4_jobs())
    name = f"pool{pooling}_{sampling_mode}"
    assert name in jobs
    args = parse_args(jobs[name])
    assert args.hidden_dim == 64
    assert args.sampling_mode == sampling_mode
    assert args.pooling == pooling
    assert args.epochs == EXPERIMENT_4_EPOCHS
    assert args.no_early_stopping is True
    if sampling_mode in ("random", "proportional"):
        assert args.num_cells == EXPERIMENT_4_NUM_CELLS
    else:
        assert args.num_cells == 256
    assert args.encoder == "sage"
    assert args.readout == "cell"
    assert args.num_gnn_layers == 2
    assert args.dropout == 0.1
    assert args.gene_strategy == "hvg"
    assert args.n_hvg == 500
    assert args.batch_size == 2
    assert args.val_fraction == 0.25
    assert args.test_fraction == 0.0
    assert args.lr == 1e-3
    assert args.seed == EXPERIMENT_4_SEED
    assert args.device == "auto"
    assert args.tissue == "Tumor"
    assert args.ici_phase == "pre"


def test_experiment_4_has_six_runs() -> None:
    names = [name for name, _ in experiment_4_jobs()]
    assert names == [
        "poolmean_whole_sample",
        "poolmean_random",
        "poolmean_proportional",
        "poolattention_whole_sample",
        "poolattention_random",
        "poolattention_proportional",
    ]


def _parse_runner_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-1 parameter-sweep experiments")
    parser.add_argument(
        "--experiment",
        default="1",
        choices=("1", "2", "3", "4"),
        help="Which experiment to launch",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Passed through to src.train. Omit to use the CLI default (data/test)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent folder for this experiment's run directories",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_runner_args()
    if args.experiment == "1":
        run_experiment_1(dataset_root=args.dataset_root, output_root=args.output_root)
    elif args.experiment == "2":
        run_experiment_2(dataset_root=args.dataset_root, output_root=args.output_root)
    elif args.experiment == "3":
        run_experiment_3(dataset_root=args.dataset_root, output_root=args.output_root)
    elif args.experiment == "4":
        run_experiment_4(dataset_root=args.dataset_root, output_root=args.output_root)
