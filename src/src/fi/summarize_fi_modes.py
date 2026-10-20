"""Collect FI-mode validation and unseen-test metrics into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODES = ("baseline", "fi_topology", "fi_relation_aware")
METRICS = (
    "val_auprc",
    "val_auroc",
    "test_auprc",
    "test_auroc",
    "test_f1",
    "test_balanced_acc",
)


def _val_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["split"] == "val")


def _run_row(run_dir: Path, mode: str, seed: int) -> dict[str, object]:
    val = _val_row(run_dir / "results.csv")
    early = json.loads((run_dir / "early_stopping.json").read_text())
    predict = json.loads((run_dir / "unseen" / "predict.json").read_text())["metrics"]
    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": int(early["best_epoch"]),
        "val_auprc": float(val["auprc"]),
        "val_auroc": float(val["auroc"]),
        "test_auprc": float(predict["auprc"]),
        "test_auroc": float(predict["auroc"]),
        "test_f1": float(predict["f1"]),
        "test_balanced_acc": float(predict["balanced_acc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for mode in MODES:
            run_dir = args.root / f"seed_{seed}" / mode
            rows.append(_run_row(run_dir, mode, seed))
    fieldnames = ["mode", "seed", "best_epoch", *METRICS]
    out = args.root / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for mode in MODES:
        chosen = [row for row in rows if row["mode"] == mode]
        summary = {"mode": mode, "n_seeds": len(chosen)}
        for metric in (*METRICS,):
            values = np.asarray([row[metric] for row in chosen], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    summary_path = args.root / "summary_mean_sd.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
