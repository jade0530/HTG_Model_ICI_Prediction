"""Compare selected benchmark architectures with bioMESH across the same splits.

Reads only saved validation and unseen files. A metric that is absent from a
file is left blank. Nothing is imputed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

BENCH_ROOT = Path("/media/rokny/DATA3/Jade/HTG_Training_Results/1016_selected_benchmarks")
BASE_ROOT = Path("/media/rokny/DATA3/Anish/HTG_Model_ICI_Results/seeds_splits_exp/baseline")
OUT_ROOT = Path(__file__).resolve().parents[2] / "reports" / "benchmark_seed_comparison"
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"

SPLIT_SEEDS = (9485, 1043, 1045, 8173, 2873)
TRAINING_SEEDS = (42, 43, 44)
FOLDER_FOR_SPLIT = {9485: 1, 1043: 2, 1045: 3, 8173: 4, 2873: 5}
MODELS = (
    ("bioMESH", None),
    ("Random forest", "random_forest"),
    ("Linear regression", "linear_regression"),
    ("MLP", "mlp"),
    ("GNN", "gnn"),
)
METRICS = ("acc", "balanced_acc", "auroc", "auprc", "f1")
PLOT_METRICS = ("acc", "auroc", "auprc", "f1")
METRIC_LABELS = {
    "acc": "Accuracy",
    "balanced_acc": "Balanced accuracy",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1",
}
COLORS = {
    "bioMESH": "#1F4E79",
    "Random forest": "#2F6F8F",
    "Linear regression": "#C46B2C",
    "MLP": "#4E4E4E",
    "GNN": "#7A9BB5",
    "FI topology": "#6B4C9A",
    "FI relation": "#2E7D4F",
    "Encoder topology": "#8C6A2F",
    "Encoder semantics": "#8F3D4B",
}
FI_GRAPH_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "fi_splits"
FI_GRAPH_MODES = (
    ("FI topology", "fi_topology"),
    ("FI relation", "fi_relation_aware"),
)
SHORT = {
    "bioMESH": "bioMESH",
    "Random forest": "RF",
    "Linear regression": "Linreg",
    "MLP": "MLP",
    "GNN": "GNN",
    "FI topology": "FI-top",
    "FI relation": "FI-rel",
    "Encoder topology": "Enc-top",
    "Encoder semantics": "Enc-sem",
}
ENCODER_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "fi_prior"
ENCODER_RUNS = (
    ("Encoder topology", "fi_topology_frozen"),
    ("Encoder semantics", "fi_semantic_frozen"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(row: dict[str, str] | None, key: str) -> str:
    if row is None or key not in row or row[key] == "":
        return ""
    return row[key]


def _val_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    for row in _read_csv_rows(path):
        if row.get("split") == "val":
            return row
    return None


def _unseen_metrics(run_dir: Path) -> dict[str, str] | None:
    predict_path = run_dir / "unseen" / "predict.json"
    if predict_path.is_file():
        payload = json.loads(predict_path.read_text())
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return {key: "" if metrics.get(key) is None else str(metrics[key]) for key in METRICS}
    results_path = run_dir / "unseen" / "results.csv"
    if not results_path.is_file():
        return None
    for row in _read_csv_rows(results_path):
        if row.get("split") == "predict":
            return {key: _metric(row, key) for key in METRICS}
    return None


def _collect() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    for label, folder in MODELS:
        for split_seed in SPLIT_SEEDS:
            for training_seed in TRAINING_SEEDS:
                if folder is None:
                    run_dir = BASE_ROOT / f"seed_{training_seed}_split_{FOLDER_FOR_SPLIT[split_seed]}"
                    note = ""
                    if split_seed == 2873 and training_seed in (43, 44):
                        note = "gene_universe_is_symbols"
                else:
                    run_dir = BENCH_ROOT / folder / f"split_{split_seed}" / f"seed_{training_seed}"
                    note = ""
                val = _val_row(run_dir / "results.csv")
                unseen = _unseen_metrics(run_dir)
                record = {
                    "architecture": label,
                    "split_seed": str(split_seed),
                    "training_seed": str(training_seed),
                    "run_dir": str(run_dir),
                    "note": note,
                    "val_file": "present" if val is not None else "missing",
                    "unseen_file": "present" if unseen is not None else "missing",
                }
                for key in METRICS:
                    record[f"val_{key}"] = _metric(val, key)
                    record[f"unseen_{key}"] = "" if unseen is None else unseen.get(key, "")
                rows.append(record)
                for cohort, present in (("val", val is not None), ("unseen", unseen is not None)):
                    if not present:
                        audit.append(
                            {
                                "architecture": label,
                                "cohort": cohort,
                                "split_seed": str(split_seed),
                                "training_seed": str(training_seed),
                                "status": "run file missing",
                            }
                        )
                        continue
                    for key in METRICS:
                        if record[f"{cohort}_{key}"] == "":
                            audit.append(
                                {
                                    "architecture": label,
                                    "cohort": cohort,
                                    "split_seed": str(split_seed),
                                    "training_seed": str(training_seed),
                                    "status": f"{key} not in saved file",
                                }
                            )
    return rows, audit


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wide(rows: list[dict[str, str]], cohort: str, metric: str) -> list[dict[str, str]]:
    columns = [f"{split_seed}_seed_{training_seed}" for split_seed in SPLIT_SEEDS for training_seed in TRAINING_SEEDS]
    wide = []
    for label, _folder in MODELS:
        item = {"architecture": label}
        for row in rows:
            if row["architecture"] != label:
                continue
            item[f"{row['split_seed']}_seed_{row['training_seed']}"] = row[f"{cohort}_{metric}"]
        for column in columns:
            item.setdefault(column, "")
        wide.append(item)
    return wide


def _summary(
    rows: list[dict[str, str]],
    models: tuple[tuple[str, str | None], ...] = MODELS,
) -> list[dict[str, str]]:
    summary = []
    for label, _folder in models:
        for cohort in ("val", "unseen"):
            for metric in METRICS:
                values = []
                for row in rows:
                    if row["architecture"] != label:
                        continue
                    text = row[f"{cohort}_{metric}"]
                    if text == "":
                        continue
                    values.append(float(text))
                item = {
                    "architecture": label,
                    "cohort": cohort,
                    "metric": metric,
                    "n": str(len(values)),
                    "mean": "" if not values else f"{mean(values):.6f}",
                    "sample_sd": "" if len(values) < 2 else f"{stdev(values):.6f}",
                    "variance": "" if len(values) < 2 else f"{stdev(values) ** 2:.6f}",
                }
                summary.append(item)
    return summary


def _fi_graph_rows() -> list[dict[str, str]]:
    """Saved 5,000-HVG inbuilt FI-graph runs. Missing split/seed files are omitted."""
    rows: list[dict[str, str]] = []
    for label, mode in FI_GRAPH_MODES:
        for split_seed in SPLIT_SEEDS:
            for training_seed in TRAINING_SEEDS:
                run_dir = FI_GRAPH_ROOT / f"split_{split_seed}" / f"seed_{training_seed}" / mode
                val = _val_row(run_dir / "results.csv")
                if val is None:
                    continue
                record = {
                    "architecture": label,
                    "split_seed": str(split_seed),
                    "training_seed": str(training_seed),
                    "run_dir": str(run_dir),
                    "note": "5000_hvg_inbuilt_fi_graph",
                }
                for key in METRICS:
                    record[f"val_{key}"] = _metric(val, key)
                    record[f"unseen_{key}"] = ""
                rows.append(record)
    return rows


def _encoder_rows() -> list[dict[str, str]]:
    """Frozen FI-encoder priors. Only saved runs are included."""
    rows: list[dict[str, str]] = []
    for label, folder in ENCODER_RUNS:
        run_dir = ENCODER_ROOT / folder
        val = _val_row(run_dir / "results.csv")
        if val is None:
            continue
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        record = {
            "architecture": label,
            "split_seed": str(config.get("split_seed", "")),
            "training_seed": str(config.get("training_seed", "")),
            "run_dir": str(run_dir),
            "note": "frozen_fi_encoder_prior",
        }
        for key in METRICS:
            record[f"val_{key}"] = _metric(val, key)
            record[f"unseen_{key}"] = ""
        rows.append(record)
    return rows


def _fi_coverage_note(extra: list[dict[str, str]]) -> str:
    covered = sorted({(row["split_seed"], row["training_seed"]) for row in extra})
    by_split: dict[str, list[str]] = {}
    for split_seed, training_seed in covered:
        by_split.setdefault(split_seed, []).append(training_seed)
    parts = [
        f"split {split_seed} seeds {', '.join(seeds)}"
        for split_seed, seeds in by_split.items()
    ]
    n = len(covered)
    return (
        " FI-top and FI-rel are 5,000-HVG inbuilt FI graphs from saved runs only"
        f" (n={n}: {'; '.join(parts)})."
    )


def _boxplots(
    rows: list[dict[str, str]],
    cohort: str,
    extra: list[dict[str, str]] | None = None,
    encoders: list[dict[str, str]] | None = None,
) -> None:
    plot_rows = list(rows)
    order = list(MODELS)
    if extra:
        plot_rows.extend(extra)
        order = [*order, *FI_GRAPH_MODES]
    if encoders:
        plot_rows.extend(encoders)
        order = [*order, *ENCODER_RUNS]
    width = 19 if encoders else (16.5 if extra else 13)
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(width, 4.4), sharey=True)
    for ax, metric in zip(axes, PLOT_METRICS):
        series = []
        labels = []
        colors = []
        for label, _folder in order:
            values = [
                float(row[f"{cohort}_{metric}"])
                for row in plot_rows
                if row["architecture"] == label and row[f"{cohort}_{metric}"] != ""
            ]
            if not values:
                continue
            series.append(values)
            labels.append(label)
            colors.append(COLORS[label])
        if not series:
            ax.set_title(METRIC_LABELS[metric])
            ax.text(0.5, 0.5, "no saved values", ha="center", va="center", transform=ax.transAxes)
            continue
        plotted = ax.boxplot(
            series,
            tick_labels=[SHORT[label] for label in labels],
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        )
        for patch, color in zip(plotted["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Score")
    fig.suptitle(f"{cohort.capitalize()} scores across 5 splits and seeds 42, 43, 44")
    note = "Diamond is the mean."
    if extra:
        note += _fi_coverage_note(extra)
    if encoders:
        covered = sorted({(row["split_seed"], row["training_seed"]) for row in encoders})
        where = ", ".join(f"split {split_seed} seed {training_seed}" for split_seed, training_seed in covered)
        note += f" Enc-top and Enc-sem are frozen FI-encoder priors ({where} only)."
    fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{cohort}_boxplots.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{cohort}_boxplots.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    rows, audit = _collect()
    _write_csv(TABLES / "per_run.csv", rows)
    _write_csv(TABLES / "missing.csv", audit)
    summary = _summary(rows)
    _write_csv(TABLES / "mean_sd.csv", summary)
    fi_rows = _fi_graph_rows()
    encoder_rows = _encoder_rows()
    val_table = [
        item
        for item in _summary(rows + fi_rows + encoder_rows, (*MODELS, *FI_GRAPH_MODES, *ENCODER_RUNS))
        if item["cohort"] == "val" and item["metric"] in PLOT_METRICS
    ]
    _write_csv(TABLES / "val_with_fi_graph.csv", val_table)
    for cohort in ("val", "unseen"):
        for metric in METRICS:
            _write_csv(TABLES / f"{cohort}_{metric}.csv", _wide(rows, cohort, metric))
        _boxplots(
            rows,
            cohort,
            extra=fi_rows if cohort == "val" else None,
            encoders=encoder_rows if cohort == "val" else None,
        )
    print(f"fi_graph_val_runs={len(fi_rows)} encoder_val_runs={len(encoder_rows)}")
    print(f"runs={len(rows)} missing_cells={len(audit)}")
    print(f"tables={TABLES}")
    print(f"figures={FIGURES}")


if __name__ == "__main__":
    main()
