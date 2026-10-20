"""Self-supervised pretraining of the Reactome FI gene encoder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.fi.encoder import FIPretrainer
from src.fi.message_graph import (
    CURATED_SEMANTICS,
    EDGE_ATTR_NAMES,
    assert_message_split,
    build_gene_index,
    computational_edges,
    load_fi_edges,
    pairs_tensor,
    sample_negative_pairs,
    select_curated_edges,
    semantic_examples,
    split_biological_pairs,
    split_symbol_pairs,
    stack_edges,
    unordered_pair,
)
from src.train.loop import resolve_device, seed_everything
from src.train.metrics import classification_metrics

DEFAULT_GRAPH = Path(
    "/media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/data/functional_interaction/fi_graph.json"
)
DEFAULT_OUTPUT = Path("/media/rokny/DATA3/Jade/HTG-interpretability/outputs/fi_encoder")


def prepare_graph(path: Path, *, include_predicted: bool, seed: int):
    edges = load_fi_edges(path)
    kept = select_curated_edges(edges, include_predicted=include_predicted)
    genes, gene_to_id = build_gene_index(kept)
    splits = split_biological_pairs(kept, gene_to_id, seed=seed)
    train_rows = []
    for edge in kept:
        if unordered_pair(edge, gene_to_id) in splits["train"]:
            train_rows.extend(computational_edges(edge, gene_to_id))
    edge_index, edge_attr = stack_edges(train_rows)
    positive = splits["train"] | splits["val"] | splits["test"]
    pos_pairs = {name: pairs_tensor(pairs) for name, pairs in splits.items()}
    semantics = {name: semantic_examples(kept, gene_to_id, pairs) for name, pairs in splits.items()}
    return {
        "genes": genes,
        "gene_to_id": gene_to_id,
        "splits": splits,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "positive": positive,
        "pos_pairs": pos_pairs,
        "semantics": semantics,
        "n_input_edges": len(edges),
        "n_kept_edges": len(kept),
    }


def _link_loss(model: FIPretrainer, embeddings, pos_pairs, neg_pairs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.cat([model.link_logits(embeddings, pos_pairs), model.link_logits(embeddings, neg_pairs)])
    labels = torch.cat(
        [
            torch.ones(pos_pairs.shape[0], device=embeddings.device),
            torch.zeros(neg_pairs.shape[0], device=embeddings.device),
        ]
    )
    return F.binary_cross_entropy_with_logits(logits, labels), logits.detach(), labels.detach()


def _semantic_scores(model: FIPretrainer, embeddings, pairs, targets):
    logits = model.semantic_logits(embeddings, pairs)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss, torch.sigmoid(logits).detach().cpu().numpy(), targets.detach().cpu().numpy()


def _semantic_metrics(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    if probs.size == 0:
        return float("nan"), float("nan")
    scored = classification_metrics(targets.reshape(-1), probs.reshape(-1))
    return scored["auroc"], scored["auprc"]


@torch.no_grad()
def evaluate_split(model, edge_index, edge_attr, pos_pairs, neg_pairs, sem_pairs, sem_targets) -> dict[str, float]:
    model.eval()
    embeddings = model.encode(edge_index, edge_attr)
    loss, logits, labels = _link_loss(model, embeddings, pos_pairs, neg_pairs)
    scores = classification_metrics(labels.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())
    sem_loss, sem_probs, sem_true = _semantic_scores(model, embeddings, sem_pairs, sem_targets)
    sem_auroc, sem_auprc = _semantic_metrics(sem_probs, sem_true)
    return {
        "loss": float(loss) + float(sem_loss),
        "link_loss": float(loss),
        "semantic_loss": float(sem_loss),
        "link_auroc": scores["auroc"],
        "link_auprc": scores["auprc"],
        "semantic_auroc": sem_auroc,
        "semantic_auprc": sem_auprc,
    }


def train_fi_encoder(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    graph = prepare_graph(args.graph_json, include_predicted=args.include_predicted_fi, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    val_neg = sample_negative_pairs(int(graph["pos_pairs"]["val"].shape[0]), len(graph["genes"]), graph["positive"], rng)
    test_neg = sample_negative_pairs(int(graph["pos_pairs"]["test"].shape[0]), len(graph["genes"]), graph["positive"], rng)
    train_probe = sample_negative_pairs(min(1024, int(graph["pos_pairs"]["train"].shape[0])), len(graph["genes"]), graph["positive"], rng)
    assert_message_split(
        graph["edge_index"],
        graph["splits"],
        {"val": val_neg, "test": test_neg, "train_probe": train_probe},
        graph["positive"],
    )
    model = FIPretrainer(len(graph["genes"]), args.hidden_dim, args.num_layers, args.dropout).to(device)
    if model.encoder.convs[0].lin is None:
        raise RuntimeError("GINEConv is not using edge_attr")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    pos_pairs = {name: pairs.to(device) for name, pairs in graph["pos_pairs"].items()}
    semantics = {
        name: (pairs.to(device), targets.to(device)) for name, (pairs, targets) in graph["semantics"].items()
    }
    val_neg = val_neg.to(device)
    test_neg = test_neg.to(device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "graph_json": str(args.graph_json),
        "include_predicted_fi": args.include_predicted_fi,
        "n_input_edges": graph["n_input_edges"],
        "n_kept_edges": graph["n_kept_edges"],
        "n_genes": len(graph["genes"]),
        "n_train_pairs": len(graph["splits"]["train"]),
        "n_val_pairs": len(graph["splits"]["val"]),
        "n_test_pairs": len(graph["splits"]["test"]),
        "n_train_message_edges": int(edge_index.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "fi_semantic_weight": args.fi_semantic_weight,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "edge_attr_names": list(EDGE_ATTR_NAMES),
        "semantic_targets": list(CURATED_SEMANTICS),
        "device": str(device),
    }
    (out / "fi_training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "fi_gene_mapping.json").write_text(
        json.dumps({"genes": graph["genes"], "gene_to_id": graph["gene_to_id"]}, indent=2) + "\n"
    )
    (out / "fi_split.json").write_text(json.dumps(split_symbol_pairs(graph["splits"], graph["genes"])) + "\n")

    metric_names = (
        "link_loss",
        "semantic_loss",
        "loss",
        "link_auroc",
        "link_auprc",
        "semantic_auroc",
        "semantic_auprc",
    )
    history_path = out / "fi_training_metrics.csv"
    with history_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(["epoch", "split", *metric_names])

    best_auprc = float("-inf")
    best_epoch = -1
    n_train = int(pos_pairs["train"].shape[0])
    n_neg = max(1, int(round(n_train * args.negative_ratio)))
    print(
        f"genes={len(graph['genes'])} kept_edges={graph['n_kept_edges']} "
        f"train_pairs={len(graph['splits']['train'])} device={device}"
    )
    for epoch in range(args.epochs):
        model.train()
        train_neg = sample_negative_pairs(n_neg, len(graph["genes"]), graph["positive"], rng).to(device)
        optimizer.zero_grad()
        embeddings = model.encode(edge_index, edge_attr)
        link_loss, _, _ = _link_loss(model, embeddings, pos_pairs["train"], train_neg)
        sem_pairs, sem_targets = semantics["train"]
        sem_loss = F.binary_cross_entropy_with_logits(model.semantic_logits(embeddings, sem_pairs), sem_targets)
        loss = link_loss + args.fi_semantic_weight * sem_loss
        loss.backward()
        optimizer.step()
        val_metrics = evaluate_split(
            model, edge_index, edge_attr, pos_pairs["val"], val_neg, *semantics["val"]
        )
        with history_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([epoch, "val", *[val_metrics[name] for name in metric_names]])
        print(
            f"epoch {epoch} val_link_auprc={val_metrics['link_auprc']:.4f} "
            f"val_link_auroc={val_metrics['link_auroc']:.4f} "
            f"val_semantic_auprc={val_metrics['semantic_auprc']:.4f}"
        )
        if val_metrics["link_auprc"] > best_auprc:
            best_auprc = val_metrics["link_auprc"]
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_link_auprc": best_auprc, "config": config},
                out / "fi_encoder_best.pt",
            )

    checkpoint = torch.load(out / "fi_encoder_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_split(
        model, edge_index, edge_attr, pos_pairs["test"], test_neg, *semantics["test"]
    )
    with history_path.open("a", newline="") as handle:
        csv.writer(handle).writerow(
            [best_epoch, "test", *[test_metrics[name] for name in metric_names]]
        )
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(edge_index, edge_attr).detach().cpu()
    torch.save({"embeddings": embeddings, "genes": graph["genes"]}, out / "fi_gene_embeddings.pt")
    print(
        f"best_epoch={best_epoch} best_val_link_auprc={best_auprc:.4f} "
        f"test_link_auprc={test_metrics['link_auprc']:.4f} "
        f"test_link_auroc={test_metrics['link_auroc']:.4f} "
        f"test_semantic_auprc={test_metrics['semantic_auprc']:.4f} "
        f"test_semantic_auroc={test_metrics['semantic_auroc']:.4f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-predicted-fi", action="store_true")
    parser.add_argument("--fi-semantic-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main() -> None:
    train_fi_encoder(parse_args())


if __name__ == "__main__":
    main()
