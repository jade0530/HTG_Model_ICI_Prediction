"""Logistic-regression sample-level ICI baseline on raw h5ad expression.

Each h5ad is one sample. Features are the mean of ``adata.X`` over cells,
standardised on the training fold. The grid is C, penalty, and solver.
``l1`` is fit as ``l1_ratio=1`` and ``l2`` as ``l1_ratio=0``.

``python benchmark/benchmark_logistic_regression.py predict --checkpoint RUN_DIR --dataset-root NEW_H5ADS --output-dir OUT``
scores unseen sample h5ads. Gene names are aligned by name; genes missing
from a new file are filled with 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from benchmark.benchmark_data import (
    add_data_args,
    add_predict_args,
    assign_split,
    build_fold,
    feature_matrix,
    load_records,
    load_saved_run,
    resolve_genes,
    score_metrics,
    write_history_csv,
    write_predict_report,
    write_search_csv,
)
from benchmark.table_grids import logistic_regression_grid
from src.train.loop import EpochResult
from src.train.metrics import format_metrics, select_threshold
from src.train.report import write_run_report

MAX_ITER = 5000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logistic-regression baseline on raw sample h5ads")
    add_data_args(parser)
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--search-grid", choices=("table",), default="table")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Write the hyperparameter table and skip the winning refit",
    )
    return parser.parse_args(argv)


def make_model(params: dict, seed: int) -> LogisticRegression:
    penalty = params["penalty"]
    if penalty == "l1":
        l1_ratio = 1.0
    elif penalty == "l2":
        l1_ratio = 0.0
    else:
        raise ValueError(f"penalty must be l1 or l2, got {penalty!r}")
    return LogisticRegression(
        C=float(params["C"]),
        solver=params["solver"],
        l1_ratio=l1_ratio,
        max_iter=MAX_ITER,
        random_state=seed,
    )


def predict_scores(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def fold_metrics(model: LogisticRegression, X: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    scores = predict_scores(model, X)
    return score_metrics(y, scores, threshold=threshold, loss=log_loss(y, scores, labels=[0, 1]))


def fit_once(params: dict, X_train, y_train, X_val, y_val, seed: int) -> tuple[LogisticRegression, list[EpochResult]]:
    model = make_model(params, seed)
    model.fit(X_train, y_train)
    history = [
        EpochResult(
            epoch=0,
            train=fold_metrics(model, X_train, y_train, threshold=0.5),
            val=fold_metrics(model, X_val, y_val, threshold=0.5),
        )
    ]
    return model, history


def tune(X_train, y_train, X_val, y_val, seed: int) -> tuple[dict, list[dict]]:
    rows = []
    best_params = None
    best_auprc = -1.0
    for params in logistic_regression_grid():
        _model, history = fit_once(params, X_train, y_train, X_val, y_val, seed)
        metrics = history[-1].val
        row = {**params, **{f"val_{key}": value for key, value in metrics.items()}}
        rows.append(row)
        print(
            f"logreg C={params['C']} penalty={params['penalty']} "
            f"solver={params['solver']} val_auprc={metrics['auprc']:.4f}"
        )
        if metrics["auprc"] > best_auprc:
            best_auprc = metrics["auprc"]
            best_params = params
    return best_params, rows


def main_train(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.split_json is None and (args.train_ids is None or args.val_ids is None):
        raise SystemExit("pass --split-json, or both --train-ids and --val-ids")
    if args.search_only and args.params_json is not None:
        raise SystemExit("--search-only does not take --params-json")

    records = load_records(args.dataset_root, args.manifest, args.tissue, args.ici_phase)
    folds = assign_split(
        records,
        split_json=args.split_json,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        test_ids=args.test_ids,
    )
    print(f"samples train={len(folds['train'])} val={len(folds['val'])} test={len(folds['test'])}")

    genes = resolve_genes(folds["train"], n_hvg=args.n_hvg, gene_universe=args.gene_universe)
    train = build_fold(folds["train"], genes)
    val = build_fold(folds["val"], genes)
    test = build_fold(folds["test"], genes) if folds["test"] else None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.X)
    X_val = scaler.transform(val.X)
    X_test = scaler.transform(test.X) if test is not None else None
    print(f"features={len(genes)} matrix={X_train.shape}")

    if args.params_json is None:
        best_params, search_rows = tune(X_train, train.y, X_val, val.y, args.seed)
    else:
        best_params = json.loads(args.params_json.read_text())
        search_rows = []
    if args.search_only:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_search_csv(out / "hyperparam_search.csv", search_rows)
        print(f"search_only configs={len(search_rows)} output_dir={out}")
        return
    print(f"best_params={best_params}")
    model, history = fit_once(best_params, X_train, train.y, X_val, val.y, args.seed)
    for row in history:
        print(f"epoch {row.epoch} {format_metrics('train', row.train)}")
        print(f"epoch {row.epoch} {format_metrics('val', row.val)}")

    train_scores = predict_scores(model, X_train)
    val_scores = predict_scores(model, X_val)
    threshold = select_threshold(
        val.y, val_scores, strategy=args.threshold_strategy, fixed=args.threshold
    )
    predictions = {"train": (train.y, train_scores), "val": (val.y, val_scores)}
    if X_test is not None:
        predictions["test"] = (test.y, predict_scores(model, X_test))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "model": "logistic_regression",
                "best_params": best_params,
                "n_hvg": args.n_hvg,
                "n_features": len(genes),
                "seed": args.seed,
                "split_json": str(args.split_json) if args.split_json else None,
                "gene_universe": str(args.gene_universe) if args.gene_universe else None,
                "threshold_strategy": args.threshold_strategy,
            },
            indent=2,
        )
        + "\n"
    )
    (out / "split.json").write_text(
        json.dumps(
            {
                name: [
                    {"sample_id": record.sample_id, "patient_key": record.patient_key, "label": record.label}
                    for record in folds[name]
                ]
                for name in ("train", "val", "test")
            },
            indent=2,
        )
        + "\n"
    )
    (out / "gene_universe.txt").write_text("\n".join(genes) + "\n")
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
    if search_rows:
        write_search_csv(out / "hyperparam_search.csv", search_rows)
    write_history_csv(out / "history.csv", history)
    dump(
        {
            "model": model,
            "scaler": scaler,
            "genes": genes,
            "threshold": threshold,
            "threshold_strategy": args.threshold_strategy,
        },
        out / "model.joblib",
    )
    write_run_report(
        out,
        history=history,
        predictions=predictions,
        threshold=threshold,
        threshold_strategy=args.threshold_strategy,
    )
    print(f"threshold={threshold:.4f} output_dir={out}")


def parse_predict_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score unseen h5ads with a saved logistic-regression run")
    add_predict_args(parser)
    return parser.parse_args(argv)


def predict(
    checkpoint: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    manifest: Path | None = None,
    tissue: str = "all",
    ici_phase: str = "all",
) -> dict:
    from src.train.dataset import DEFAULT_MANIFEST

    bundle = load_saved_run(checkpoint)
    records = load_records(
        dataset_root,
        manifest or DEFAULT_MANIFEST,
        tissue,
        ici_phase,
        require_label=False,
    )
    X = bundle["scaler"].transform(feature_matrix(records, bundle["genes"]))
    scores = predict_scores(bundle["model"], X)
    print(f"samples={len(records)} threshold={bundle['threshold']:.4f} checkpoint={bundle['path']}")
    metrics = write_predict_report(
        output_dir,
        records,
        scores,
        threshold=float(bundle["threshold"]),
        checkpoint=bundle["path"],
        threshold_strategy=bundle["threshold_strategy"],
    )
    return {"output_dir": Path(output_dir), "metrics": metrics, "scores": scores}


def main_predict(argv: list[str] | None = None) -> None:
    args = parse_predict_args(argv)
    result = predict(
        args.checkpoint,
        args.dataset_root,
        args.output_dir,
        manifest=args.manifest,
        tissue=args.tissue,
        ici_phase=args.ici_phase,
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
