"""CLI: python -m src.train --dataset-root ../data/test --output-dir ../outputs/train."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train.dataset import DEFAULT_MANIFEST, REPO_ROOT, collect_records
from src.train.loop import TrainConfig, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 sample-graph classifier")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "test",
        help="Folder of sample h5ads, or the root that sample_manifest.csv paths are relative to",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "train")
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-cells", type=int, default=256)
    parser.add_argument("--pooling", choices=("mean", "attention"), default="mean")
    parser.add_argument("--gene-strategy", choices=("hvg", "expressed"), default="hvg")
    parser.add_argument("--sampling-mode", choices=("random", "proportional"), default="random")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--hvg-cells-per-sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-cache-samples", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = collect_records(
        args.dataset_root,
        args.manifest,
        tissue=None if args.tissue.lower() == "all" else args.tissue,
        ici_phase=None if args.ici_phase.lower() == "all" else args.ici_phase,
    )
    print(f"samples={len(records)} dataset_root={args.dataset_root}")
    result = train(
        records,
        config=TrainConfig(
            hidden_dim=args.hidden_dim,
            num_cells=args.num_cells,
            pooling=args.pooling,
            gene_strategy=args.gene_strategy,
            sampling_mode=args.sampling_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            n_hvg=args.n_hvg,
            hvg_cells_per_sample=args.hvg_cells_per_sample,
            cache_samples=not args.no_cache_samples,
            use_pos_weight=not args.no_pos_weight,
            seed=args.seed,
            device=args.device,
            threshold=args.threshold,
        ),
        output_dir=args.output_dir,
    )
    print(f"best_epoch={result.best_epoch} output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
