"""One-batch check of baseline, fi_topology, and fi_relation_aware.

Run from the package root:

    python -m src.fi.smoke_fi_modes --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from src.data.sampler import CellSampler
from src.fi.sample_edges import build_sample_fi_edges
from src.graph.build_local_graph import GeneUniverse
from src.train.dataset import SampleGraphDataset, collect_records
from src.train.loop import TrainConfig, _fi_edges_for, resolve_device, seed_everything
from src.train.model import SampleGraphClassifier

WORKTREE = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE = WORKTREE / "outputs" / "fi_prior" / "none" / "gene_universe.txt"
DEFAULT_DATA = Path("/media/rokny/DATA6/Jade/sc_training_dataset_latest/sample_h5ad/Tumor/pre")
DEFAULT_OUT = WORKTREE / "outputs" / "fi_modes" / "smoke.json"


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    src, dst = edge_index.tolist()
    return {tuple(sorted((left, right))) for left, right in zip(src, dst)}


def _directed(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def _config(mode: str) -> TrainConfig:
    return TrainConfig(
        hidden_dim=128,
        num_cells=128,
        pooling="mean",
        readout="cell",
        encoder="sage",
        num_gnn_layers=2,
        dropout=0.1,
        gene_strategy="hvg",
        sampling_mode="random",
        batch_size=2,
        n_hvg=5000,
        training_seed=1,
        gene_prior="none",
        fi_mode=mode,
    )


def _dataset(records, universe: GeneUniverse, mode: str) -> SampleGraphDataset:
    config = _config(mode)
    return SampleGraphDataset(
        records,
        sampler=CellSampler(num_cells=config.num_cells, seed=config.training_seed),
        gene_universe=universe,
        gene_strategy="hvg",
        sampling_mode="random",
        training=True,
        seed=config.training_seed,
        n_hvg=5000,
        hvg_names=universe.names,
        fi_edges=_fi_edges_for(config, universe),
    )


def _forward_backward(model, batch, device: torch.device) -> dict[str, float]:
    model.train(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    batch = batch.to(device)
    started = time.perf_counter()
    logit = model(batch)
    loss = F.binary_cross_entropy_with_logits(logit, batch.y.reshape(-1).to(dtype=logit.dtype))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_ms = (time.perf_counter() - started) * 1000.0
    forward_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    backward_started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_ms = (time.perf_counter() - backward_started) * 1000.0
    backward_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "loss": float(loss.detach().cpu()),
        "forward_ms": round(forward_ms, 3),
        "backward_ms": round(backward_ms, 3),
        "forward_peak_bytes": forward_bytes,
        "backward_peak_bytes": backward_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--hvg-file", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    names = tuple(line.strip() for line in args.hvg_file.read_text().splitlines() if line.strip())
    universe = GeneUniverse(names)
    topology = build_sample_fi_edges(names, mode="fi_topology")
    relation = build_sample_fi_edges(names, mode="fi_relation_aware")
    checks = {
        "gene_universe": len(names),
        "fi_pairs": topology.stats["n_fi_pairs"],
        "topology_edges": int(topology.edge_index.shape[1]),
        "relation_edges": int(relation.edge_index.shape[1]),
        "same_pairs": _pairs(topology.edge_index) == _pairs(relation.edge_index),
        "same_directed_edges": _directed(topology.edge_index) == _directed(relation.edge_index),
        "topology_has_no_attr": topology.edge_attr is None,
        "relation_attr_shape": None if relation.edge_attr is None else list(relation.edge_attr.shape),
    }
    if checks["fi_pairs"] != 14656 or checks["topology_edges"] != 29312 or checks["relation_edges"] != 29312:
        raise SystemExit(f"unexpected FI counts: {checks}")
    if not checks["same_pairs"] or not checks["same_directed_edges"] or not checks["topology_has_no_attr"]:
        raise SystemExit(f"FI topology mismatch: {checks}")

    records = collect_records(args.dataset_root)[:2]
    baseline_graph = _dataset(records[:1], universe, "baseline")[0]
    topology_graph = _dataset(records[:1], universe, "fi_topology")[0]
    relation_graph = _dataset(records[:1], universe, "fi_relation_aware")[0]
    if ("gene", "fi", "gene") in baseline_graph.edge_types:
        raise SystemExit("baseline graph contains FI edges")
    for graph, prepared in ((topology_graph, topology), (relation_graph, relation)):
        if int(graph["gene"].num_nodes) != len(names):
            raise SystemExit("FI graph dropped GeneUniverse genes")
        if int(graph["gene", "fi", "gene"].edge_index.shape[1]) != 29312:
            raise SystemExit("sample graph FI edge count is not 29312")
        incident = set(graph["gene", "fi", "gene"].edge_index.flatten().tolist())
        missing = [index for index in range(len(names)) if index not in incident]
        if not missing:
            raise SystemExit("every gene has an FI edge")
        seed_everything(1)
        model = SampleGraphClassifier(
            len(names), hidden_dim=128, encoder="sage", num_layers=2, dropout=0.1, fi_mode=prepared.mode
        )
        gene_x = model.gene_emb(graph["gene"].global_id)
        isolated = missing[0]
        if prepared.mode == "fi_topology":
            delta = model.fi_convs[0](gene_x, graph["gene", "fi", "gene"].edge_index)
        else:
            delta = model.fi_convs[0](
                gene_x,
                graph["gene", "fi", "gene"].edge_index,
                graph["gene", "fi", "gene"].edge_attr,
            )
            held = relation_graph.clone()
            held["gene", "fi", "gene"].edge_attr = torch.zeros_like(held["gene", "fi", "gene"].edge_attr)
            model.eval()
            with torch.no_grad():
                original = model(relation_graph)
                replaced = model(held)
            if torch.allclose(original, replaced):
                raise SystemExit("relation-aware edge attributes did not change the output")
            checks["relation_attr_changes_output"] = True
        if not torch.equal(delta[isolated], torch.zeros_like(delta[isolated])):
            raise SystemExit(f"uncovered gene {names[isolated]} received an FI message")
        checks["uncovered_gene"] = names[isolated]
        checks["uncovered_gene_id"] = isolated
        checks["uncovered_genes"] = len(missing)

    device = resolve_device(args.device)
    runtime = {}
    for mode in ("baseline", "fi_topology", "fi_relation_aware"):
        seed_everything(1)
        dataset = _dataset(records, universe, mode)
        cells = [dataset[index]["cell"].source_index.tolist() for index in range(len(dataset))]
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = SampleGraphClassifier(
            len(names),
            hidden_dim=128,
            encoder="sage",
            num_layers=2,
            dropout=0.1,
            fi_mode=mode if mode != "baseline" else "baseline",
        ).to(device)
        batch = next(iter(loader))
        stats = _forward_backward(model, batch, device)
        if mode == "fi_relation_aware":
            grad = model.fi_convs[0].edge_lin.weight.grad
            if grad is None or float(grad.abs().sum()) == 0.0:
                raise SystemExit("relation edge attributes received no gradient")
            checks["relation_attr_grad"] = float(grad.abs().sum())
        runtime[mode] = {
            **stats,
            "num_genes": int(batch["gene"].num_nodes) if batch["gene"].batch is None else "batched",
            "sampled_cells": cells,
        }
        print(
            f"{mode} forward_ms={stats['forward_ms']} backward_ms={stats['backward_ms']} "
            f"forward_peak_mib={stats['forward_peak_bytes'] / 1024**2:.1f} "
            f"backward_peak_mib={stats['backward_peak_bytes'] / 1024**2:.1f}"
        )
    same_cells = runtime["baseline"]["sampled_cells"] == runtime["fi_topology"]["sampled_cells"] == runtime["fi_relation_aware"]["sampled_cells"]
    if not same_cells:
        raise SystemExit("cell samples differ across modes")
    checks["same_sampled_cells"] = True
    payload = {"device": str(device), "checks": checks, "runtime": runtime}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
