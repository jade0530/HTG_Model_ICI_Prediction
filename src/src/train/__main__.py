"""CLI: python -m src.train ...  |  python -m src.train predict --checkpoint ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.sampler import SAMPLING_MODES
from src.train.dataset import DEFAULT_MANIFEST, REPO_ROOT, collect_records
from src.train.loop import TrainConfig, predict, train


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
    parser.add_argument("--num-cells", type=int, default=256, help="Cell subsample size. Ignored by whole_sample and by_cell_type")
    parser.add_argument("--pooling", choices=("mean", "attention"), default="mean")
    parser.add_argument(
        "--readout",
        choices=("cell", "gene", "both"),
        default="cell",
        help="Pool cell states, gene states, or concatenate both into the prediction head",
    )
    parser.add_argument("--encoder", choices=("placeholder", "sage", "gat"), default="sage")
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-strategy", choices=("hvg", "expressed"), default="hvg")
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default="random",
        help="random cells, type-proportional mix, all cells in the sample, or local graphs per cell type merged into one sample graph",
    )
    parser.add_argument(
        "--cell-type-level",
        choices=("fine", "main"),
        default="fine",
        help="predicted_labels (fine) or predicted_labels_mainCellType (main)",
    )
    parser.add_argument(
        "--cell-type",
        action="append",
        default=None,
        help="Keep only this cell type when aggregating by_cell_type local graphs. Repeat to pass several types",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Run all --epochs; still save and reload the best validation-AUPRC checkpoint",
    )
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience on validation AUPRC")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum validation-AUPRC improvement required to reset patience",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--cancer-proportional-split",
        "-cancer-proportional-split",
        action="store_true",
        help="Stratify the patient-disjoint train/val/test split by cancer type and response",
    )
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed used only for the patient-disjoint train/val/test split",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=0,
        help="Random seed for weights, data order, cell sampling, dropout, and CUDA",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto/cuda use GPU 1; cpu forces CPU",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("max_f1", "youden", "fixed"),
        default="max_f1",
        help="How to pick the R/NR cutoff. max_f1/youden are tuned on val; fixed uses --threshold",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-cache-samples", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record per-graph size and one forward/backward memory+runtime pass before training",
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="Keep a separate SWA weight average; does not replace the best validation checkpoint",
    )
    parser.add_argument(
        "--swa-start",
        type=int,
        default=None,
        help="First 0-based epoch to enter the SWA average. Default: last 25%% of --epochs",
    )
    parser.add_argument(
        "--swa-freq",
        type=int,
        default=1,
        help="Update the SWA average every this many epochs after --swa-start",
    )
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=None,
        help="Constant SWA learning rate after --swa-start. Default: the current --lr",
    )
    return parser.parse_args(argv)


def parse_predict_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score new samples with a saved HTG run")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="best.pt, swa.pt, or a training output directory (loads best.pt)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Folder of new sample h5ads, or a dataset root for the manifest",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "predict")
    parser.add_argument("--tissue", default="all")
    parser.add_argument("--ici-phase", default="all")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override the checkpoint device. auto/cuda use GPU 1; cpu forces CPU",
    )
    return parser.parse_args(argv)


def main_train(argv: list[str] | None = None) -> None:
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
            readout=args.readout,
            encoder=args.encoder,
            num_gnn_layers=args.num_gnn_layers,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            gene_strategy=args.gene_strategy,
            sampling_mode=args.sampling_mode,
            cell_type_level=args.cell_type_level,
            cell_types=tuple(args.cell_type or ()),
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping=not args.no_early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            lr=args.lr,
            optimizer=args.optimizer,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            cancer_proportional_split=args.cancer_proportional_split,
            n_hvg=args.n_hvg,
            cache_samples=not args.no_cache_samples,
            use_pos_weight=not args.no_pos_weight,
            split_seed=args.split_seed,
            training_seed=args.training_seed,
            device=args.device,
            threshold=args.threshold,
            threshold_strategy=args.threshold_strategy,
            profile=args.profile,
            swa=args.swa,
            swa_start=args.swa_start,
            swa_freq=args.swa_freq,
            swa_lr=args.swa_lr,
        ),
        output_dir=args.output_dir,
    )
    print(
        f"best_epoch={result.best_epoch} stop_epoch={result.stop_epoch} "
        f"best_val_auprc={'nan' if result.best_val_auprc != result.best_val_auprc else f'{result.best_val_auprc:.4f}'} "
        f"threshold={result.threshold:.4f} output_dir={result.output_dir}"
        + (
            f" swa_n_averaged={result.swa_n_averaged} swa_threshold={result.swa_threshold:.4f}"
            if result.swa_n_averaged
            else ""
        )
    )


def main_predict(argv: list[str] | None = None) -> None:
    args = parse_predict_args(argv)
    records = collect_records(
        args.dataset_root,
        args.manifest,
        tissue=None if args.tissue.lower() == "all" else args.tissue,
        ici_phase=None if args.ici_phase.lower() == "all" else args.ici_phase,
        require_label=False,
    )
    print(f"samples={len(records)} dataset_root={args.dataset_root}")
    result = predict(
        records,
        args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"wrote {result['output_dir']}/predictions.csv")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "predict":
        main_predict(argv[1:])
        return
    main_train(argv)


if __name__ == "__main__":
    main()
