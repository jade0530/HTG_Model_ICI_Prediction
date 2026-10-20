"""Audit tumor-pre cancer type and ICI treatment, then save tables and figures separately."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

DATA_ROOT = Path("/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre")
OUT_ROOT = Path(__file__).resolve().parents[2] / "reports" / "tumor_pre_cohort"
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"

BAR = "#1F4E79"
ICI_COLORS = {
    "Anti-pd-1": "#1F4E79",
    "Anti-pd-l1": "#2F6F8F",
    "Anti-pd-1+aCTLA-4": "#C46B2C",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def _load_rows() -> list[dict[str, str]]:
    rows = []
    for path in sorted(DATA_ROOT.glob("*.metadata.csv")):
        with path.open(newline="") as handle:
            record = next(csv.DictReader(handle))
        record["n_cells"] = int(record["n_cells"])
        record["patient_key"] = f"{record['dataset_id']}::{record['patient_id']}"
        rows.append(record)
    if not rows:
        raise FileNotFoundError(f"no metadata csv files in {DATA_ROOT}")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _group(rows: list[dict], *keys: str) -> list[dict]:
    buckets: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        bucket = buckets.setdefault(
            key,
            {
                **{name: row[name] for name in keys},
                "datasets": set(),
                "samples": 0,
                "patients": set(),
                "cells": 0,
            },
        )
        bucket["datasets"].add(row["dataset_id"])
        bucket["samples"] += 1
        bucket["patients"].add(row["patient_key"])
        bucket["cells"] += row["n_cells"]
    summary = []
    for bucket in buckets.values():
        item = {name: bucket[name] for name in keys}
        if len(keys) == 1:
            item["n_datasets"] = len(bucket["datasets"])
        item["n_samples"] = bucket["samples"]
        item["n_patients"] = len(bucket["patients"])
        item["n_cells"] = bucket["cells"]
        summary.append(item)
    summary.sort(key=lambda item: (-item["n_cells"], item[keys[0]]))
    return summary


def _horizontal_bars(labels: list[str], values: list[int], *, title: str, xlabel: str, stem: str, color: str = BAR) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(labels) + 1.2)))
    positions = list(range(len(labels) - 1, -1, -1))
    ax.barh(positions, values, color=color)
    ax.set_yticks(positions, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, stem)


def main() -> None:
    _style()
    rows = _load_rows()
    cancer = _group(rows, "Cancer type")
    dataset_cancer = _group(rows, "dataset_id", "Cancer type")
    ici = _group(rows, "ICI treatment type")
    cancer_ici = _group(rows, "Cancer type", "ICI treatment type")
    drug = _group(rows, "ICI treatment drug")
    dataset_drug = _group(rows, "dataset_id", "ICI treatment type", "ICI treatment drug")

    _write_csv(TABLES / "cancer_type.csv", cancer)
    _write_csv(TABLES / "dataset_by_cancer_type.csv", dataset_cancer)
    _write_csv(TABLES / "ici_treatment_type.csv", ici)
    _write_csv(TABLES / "cancer_type_by_ici_treatment_type.csv", cancer_ici)
    _write_csv(TABLES / "ici_treatment_drug.csv", drug)
    _write_csv(TABLES / "dataset_by_ici_drug.csv", dataset_drug)

    _horizontal_bars(
        [row["Cancer type"] for row in cancer],
        [row["n_cells"] for row in cancer],
        title="Tumor pre cells by cancer type",
        xlabel="Cells",
        stem="cells_by_cancer_type",
    )
    _horizontal_bars(
        [row["Cancer type"] for row in cancer],
        [row["n_samples"] for row in cancer],
        title="Tumor pre samples by cancer type",
        xlabel="Samples",
        stem="samples_by_cancer_type",
    )
    _horizontal_bars(
        [row["ICI treatment type"] for row in ici],
        [row["n_cells"] for row in ici],
        title="Tumor pre cells by ICI treatment type",
        xlabel="Cells",
        stem="cells_by_ici_treatment_type",
    )

    dataset_cells: dict[str, int] = {}
    for row in dataset_cancer:
        dataset_cells[row["dataset_id"]] = dataset_cells.get(row["dataset_id"], 0) + row["n_cells"]
    dataset_order = sorted(dataset_cells, key=lambda name: -dataset_cells[name])
    cancers = [row["Cancer type"] for row in cancer]
    palette = ["#1F4E79", "#2F6F8F", "#7A9BB5", "#C46B2C", "#8C5A3C", "#4E4E4E", "#8A8F98", "#C5CAD3"]
    colors = {name: palette[index % len(palette)] for index, name in enumerate(cancers)}
    lookup = {(row["dataset_id"], row["Cancer type"]): row["n_cells"] for row in dataset_cancer}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    left = [0] * len(dataset_order)
    positions = list(range(len(dataset_order) - 1, -1, -1))
    for name in cancers:
        values = [lookup.get((dataset, name), 0) for dataset in dataset_order]
        if not any(values):
            continue
        ax.barh(positions, values, left=left, color=colors[name], label=name)
        left = [base + value for base, value in zip(left, values)]
    ax.set_yticks(positions, dataset_order)
    ax.set_xlabel("Cells")
    ax.set_title("Tumor pre cells by dataset and cancer type")
    ax.legend(title="Cancer type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, "cells_by_dataset_and_cancer_type")

    print(f"tables={TABLES}")
    print(f"figures={FIGURES}")
    print(f"samples={len(rows)} cells={sum(row['n_cells'] for row in rows)}")


if __name__ == "__main__":
    main()
