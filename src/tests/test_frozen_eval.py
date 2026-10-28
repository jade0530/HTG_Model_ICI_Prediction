from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from benchmark.frozen_eval import (
    FROZEN_SEEDS,
    REPORT_MODELS,
    evaluate_command,
    format_mean_pm_sd,
    freeze_eval_seeds,
    metric_rows,
    plot_auprc_distribution,
    summary_row,
    write_metric_tables,
)


def _runs() -> list[dict]:
    scores = {
        1: (0.50, 0.60, 0.20, 0.40),
        2: (0.60, 0.70, 0.40, 0.50),
        3: (0.70, 0.80, 0.60, 0.60),
        4: (0.50, 0.60, 0.20, 0.40),
        5: (0.60, 0.70, 0.40, 0.50),
        6: (0.70, 0.80, 0.60, 0.60),
        7: (0.50, 0.60, 0.20, 0.40),
        8: (0.60, 0.70, 0.40, 0.50),
        9: (0.70, 0.80, 0.60, 0.60),
        10: (0.60, 0.70, 0.40, 0.50),
    }
    return [
        {
            "training_seed": seed,
            "acc": acc,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
        }
        for seed, (acc, auroc, auprc, f1) in scores.items()
    ]


def test_frozen_seeds_are_one_through_ten() -> None:
    assert FROZEN_SEEDS == tuple(range(1, 11))
    assert REPORT_MODELS == ("random_forest", "logistic_regression", "mlp", "gnn")


def test_freeze_writes_the_shared_list_and_rejects_a_different_list(tmp_path: Path) -> None:
    path = tmp_path / "frozen_eval_seeds.json"
    payload = freeze_eval_seeds(path)
    assert payload["seeds"] == list(range(1, 11))
    assert freeze_eval_seeds(path)["split_seed"] == 8173
    path.write_text(json.dumps({"seeds": [45], "split_seed": 8173}) + "\n")
    with pytest.raises(ValueError, match="expected"):
        freeze_eval_seeds(path)


def test_metric_table_reports_every_seed_then_mean_and_population_sd() -> None:
    rows = metric_rows("mlp", _runs())
    assert [row["record"] for row in rows] == ["run"] * 10 + ["mean", "sd"]
    assert [row["training_seed"] for row in rows[:10]] == list(range(1, 11))
    auprc = [0.20, 0.40, 0.60, 0.20, 0.40, 0.60, 0.20, 0.40, 0.60, 0.40]
    assert rows[10]["auprc"] == pytest.approx(float(np.mean(auprc)))
    assert rows[11]["auprc"] == pytest.approx(float(np.std(auprc, ddof=0)))
    summary = summary_row("mlp", _runs())
    assert summary["auprc"] == format_mean_pm_sd(rows[10]["auprc"], rows[11]["auprc"])
    assert " ± " in summary["acc"]


def test_commands_use_the_frozen_seeds_and_skip_biomesh(tmp_path: Path) -> None:
    command = evaluate_command(
        "python",
        "gnn",
        dataset_root=tmp_path,
        manifest=tmp_path / "manifest.csv",
        split_json=tmp_path / "split.json",
        output_dir=tmp_path / "gnn",
        device="cuda",
    )
    assert command[command.index("--stage") + 1] == "evaluate"
    assert command[command.index("--eval-seeds") + 1 : command.index("--eval-seeds") + 11] == [
        str(seed) for seed in range(1, 11)
    ]
    assert "biomesh" not in command
    with pytest.raises(ValueError):
        evaluate_command(
            "python",
            "biomesh",
            dataset_root=tmp_path,
            manifest=tmp_path / "manifest.csv",
            split_json=tmp_path / "split.json",
            output_dir=tmp_path / "biomesh",
            device="cpu",
        )


def test_report_writes_the_table_and_auprc_distribution(tmp_path: Path) -> None:
    by_model = {model: _runs() for model in REPORT_MODELS}
    write_metric_tables(tmp_path, by_model)
    plot_auprc_distribution(by_model, tmp_path / "auprc_distribution.png")
    assert (tmp_path / "metrics_by_seed.csv").is_file()
    assert (tmp_path / "metrics_summary.csv").is_file()
    assert (tmp_path / "auprc_distribution.png").stat().st_size > 0
    assert (tmp_path / "auprc_distribution.svg").stat().st_size > 0
    saved = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert saved["training_seeds"] == list(range(1, 11))
    assert saved["sd"].startswith("population")
