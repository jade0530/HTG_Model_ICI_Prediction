"""Write a short FI benchmark summary from saved experiment files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.fi.encoder import FIPretrainer
from src.fi.message_graph import CURATED_SEMANTICS
from src.fi.pretrain import prepare_graph
from src.train.metrics import classification_metrics

WORKTREE = Path(__file__).resolve().parents[3]
ENCODER_ROOT = WORKTREE / "outputs" / "fi_encoder_experiments"
PRIOR_ROOT = WORKTREE / "outputs" / "fi_prior"
SUMMARY_ROOT = WORKTREE / "outputs" / "fi_benchmark_summary"
GRAPH_JSON = Path(
    "/media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/data/functional_interaction/fi_graph.json"
)

ENCODER_RUNS = (
    ("Topology only", ENCODER_ROOT / "topology_only"),
    ("Topology + semantics", ENCODER_ROOT / "topology_semantics"),
)
PRIOR_RUNS = (
    ("No FI", PRIOR_ROOT / "none"),
    ("Topology FI", PRIOR_ROOT / "fi_topology_frozen"),
    ("Semantic FI", PRIOR_ROOT / "fi_semantic_frozen"),
)
RELATION_LABELS = {
    "activation": "Activation",
    "inhibition": "Inhibition",
    "catalysis": "Catalysis",
    "expression_regulation": "Expression regulation",
    "complex": "Complex",
    "co_input": "Co-input",
    "reaction_association": "Reaction association",
}
ICI_METRICS = (
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
    ("f1", "F1"),
    ("balanced_acc", "Balanced accuracy"),
)
COLORS = {
    "prevalence": "#C5CAD3",
    "auprc": "#1F4E79",
    "No FI": "#4E4E4E",
    "Topology FI": "#2F6F8F",
    "Semantic FI": "#C46B2C",
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _encoder_rows() -> list[dict]:
    rows = []
    for name, folder in ENCODER_RUNS:
        history = list(csv.DictReader((folder / "fi_training_metrics.csv").open()))
        test = next(row for row in history if row["split"] == "test")
        epoch = test["epoch"]
        val = next(row for row in history if row["split"] == "val" and row["epoch"] == epoch)
        rows.append(
            {
                "model": name,
                "best_epoch": int(epoch),
                "val_auprc": float(val["link_auprc"]),
                "val_auroc": float(val["link_auroc"]),
                "test_auprc": float(test["link_auprc"]),
                "test_auroc": float(test["link_auroc"]),
                "source": str(folder / "fi_training_metrics.csv"),
            }
        )
    return rows


def _semantic_rows() -> list[dict]:
    folder = ENCODER_ROOT / "topology_semantics"
    config = json.loads((folder / "fi_training_config.json").read_text())
    graph = prepare_graph(GRAPH_JSON, include_predicted=False, seed=int(config["seed"]))
    try:
        checkpoint = torch.load(folder / "fi_encoder_best.pt", map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(folder / "fi_encoder_best.pt", map_location="cpu")
    model = FIPretrainer(
        len(graph["genes"]),
        int(config["hidden_dim"]),
        int(config["num_layers"]),
        float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph["edge_index"], graph["edge_attr"])
        pairs, targets = graph["semantics"]["test"]
        probs = torch.sigmoid(model.semantic_logits(embeddings, pairs)).numpy()
        labels = targets.numpy()
    rows = []
    for column, name in enumerate(CURATED_SEMANTICS):
        metrics = classification_metrics(labels[:, column], probs[:, column])
        rows.append(
            {
                "relation": name,
                "label": RELATION_LABELS[name],
                "test_prevalence": float(labels[:, column].mean()),
                "test_auprc": metrics["auprc"],
                "test_auroc": metrics["auroc"],
                "source": str(folder / "fi_encoder_best.pt"),
            }
        )
    return rows


def _prior_rows() -> list[dict]:
    rows = []
    for name, folder in PRIOR_RUNS:
        if not (folder / "results.csv").is_file():
            continue
        config = json.loads((folder / "config.json").read_text())
        result = next(row for row in csv.DictReader((folder / "results.csv").open()) if row["split"] == "val")
        stopping = json.loads((folder / "early_stopping.json").read_text())
        rows.append(
            {
                "condition": name,
                "training_seed": int(config["training_seed"]),
                "split_seed": int(config["split_seed"]),
                "best_epoch": int(stopping["best_epoch"]),
                "auprc": float(result["auprc"]),
                "auroc": float(result["auroc"]),
                "f1": float(result["f1"]),
                "balanced_acc": float(result["balanced_acc"]),
                "source": str(folder / "results.csv"),
            }
        )
    return rows


def _plot_semantic(rows: list[dict], stem: Path) -> None:
    labels = [row["label"] for row in rows][::-1]
    prevalence = [row["test_prevalence"] for row in rows][::-1]
    auprc = [row["test_auprc"] for row in rows][::-1]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(y + 0.18, prevalence, height=0.34, color=COLORS["prevalence"], label="Test prevalence")
    ax.barh(y - 0.18, auprc, height=0.34, color=COLORS["auprc"], label="Test AUPRC")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Semantic reconstruction on held-out FI pairs")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    _save(fig, stem)


def _plot_ici(rows: list[dict], stem: Path) -> None:
    conditions = [name for name, _folder in PRIOR_RUNS if any(row["condition"] == name for row in rows)]
    metric_keys = [key for key, _label in ICI_METRICS]
    metric_labels = [label for _key, label in ICI_METRICS]
    seeds = sorted({row["training_seed"] for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(metric_labels))
    if len(seeds) == 1:
        width = 0.24
        for index, condition in enumerate(conditions):
            match = next(row for row in rows if row["condition"] == condition)
            values = [match[key] for key in metric_keys]
            ax.bar(
                x + (index - (len(conditions) - 1) / 2) * width,
                values,
                width=width,
                color=COLORS[condition],
                label=condition,
            )
        ax.set_title(f"ICI validation, training seed {seeds[0]}")
    else:
        width = 0.24
        for index, condition in enumerate(conditions):
            selected = [row for row in rows if row["condition"] == condition]
            means = []
            stds = []
            for key_index, key in enumerate(metric_keys):
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
                ax.scatter(
                    np.full(len(values), x[key_index] + (index - (len(conditions) - 1) / 2) * width),
                    values,
                    color=COLORS[condition],
                    s=28,
                    zorder=3,
                )
            ax.errorbar(
                x + (index - (len(conditions) - 1) / 2) * width,
                means,
                yerr=stds,
                fmt="o",
                color=COLORS[condition],
                capsize=3,
                label=condition,
            )
        ax.set_title("ICI validation, mean ± SD across training seeds")
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, stem)


def main() -> None:
    _style()
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    encoder = _encoder_rows()
    semantic = _semantic_rows()
    prior = _prior_rows()
    _write_csv(SUMMARY_ROOT / "fi_encoder_summary.csv", encoder)
    _write_csv(SUMMARY_ROOT / "fi_semantic_reconstruction_test.csv", semantic)
    _plot_semantic(semantic, SUMMARY_ROOT / "fi_semantic_reconstruction_test")
    if prior:
        _write_csv(SUMMARY_ROOT / "ici_fi_prior_summary.csv", prior)
        _plot_ici(prior, SUMMARY_ROOT / "ici_fi_prior_validation")
    print(f"wrote {SUMMARY_ROOT}")
    print(f"downstream_runs={len(prior)}")


if __name__ == "__main__":
    main()
