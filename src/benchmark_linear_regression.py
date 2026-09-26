"""Linear-regression sample-level ICI baseline on raw h5ad expression.

Each h5ad is one sample. Features are the mean of ``adata.X`` over cells.
The model is SGD linear regression on labels {0, 1}; predicted values are
the ranking scores. Pass the HTG ``split.json`` (or sample-id lists) so the
split matches the graph model. Hyperparameters are scored on the given val fold.

``python benchmark_linear_regression.py predict --checkpoint RUN_DIR --dataset-root NEW_H5ADS --output-dir OUT``
scores unseen sample h5ads with the saved model. Gene names are aligned by
name; genes missing from a new file are filled with 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.train.benchmark_data import (
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
from src.train.loop import EpochResult
from src.train.metrics import format_metrics, select_threshold
from src.train.report import write_run_report

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ETA0S = (1e-3, 1e-2, 1e-1)
PENALTIES = ("l2", "elasticnet")
L1_RATIOS = (0.15, 0.5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear-regression baseline on raw sample h5ads")
    add_data_args(parser)
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args(argv)


def to_scores(raw: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(raw, dtype=np.float64).reshape(-1), 0.0, 1.0)


def predict_scores(model: SGDRegressor, X: np.ndarray) -> np.ndarray:
    return to_scores(model.predict(X))


def fold_metrics(model: SGDRegressor, X: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    raw = model.predict(X)
    scores = to_scores(raw)
    return score_metrics(y, scores, threshold=threshold, loss=mean_squared_error(y, raw))


def make_model(params: dict, seed: int) -> SGDRegressor:
    return SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        random_state=seed,
        **params,
    )


def run_epochs(model: SGDRegressor, X_train, y_train, X_val, y_val, epochs: int, seed: int) -> list[EpochResult]:
    rng = np.random.default_rng(seed)
    history: list[EpochResult] = []
    for epoch in range(epochs):
        order = rng.permutation(len(y_train))
        model.partial_fit(X_train[order], y_train[order])
        history.append(
            EpochResult(
                epoch=epoch,
                train=fold_metrics(model, X_train, y_train, threshold=0.5),
                val=fold_metrics(model, X_val, y_val, threshold=0.5),
            )
        )
    return history


def tune(X_train, y_train, X_val, y_val, epochs: int, seed: int) -> tuple[dict, list[dict]]:
    rows = []
    best_params = None
    best_auprc = -1.0
    for alpha, eta0, penalty in product(ALPHAS, ETA0S, PENALTIES):
        l1_values = L1_RATIOS if penalty == "elasticnet" else (0.15,)
        for l1_ratio in l1_values:
            params = {"alpha": alpha, "eta0": eta0, "penalty": penalty, "l1_ratio": l1_ratio}
            model = make_model(params, seed)
            history = run_epochs(model, X_train, y_train, X_val, y_val, epochs, seed)
            metrics = history[-1].val
            row = {**params, **{f"val_{key}": value for key, value in metrics.items()}}
            rows.append(row)
            print(
                f"linreg alpha={alpha} eta0={eta0} penalty={penalty} "
                f"l1_ratio={l1_ratio} val_auprc={metrics['auprc']:.4f}"
            )
            if metrics["auprc"] > best_auprc:
                best_auprc = metrics["auprc"]
                best_params = params
    return best_params, rows


def main_train(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.split_json is None and (args.train_ids is None or args.val_ids is None):
        raise SystemExit("pass --split-json, or both --train-ids and --val-ids")

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

    best_params, search_rows = tune(X_train, train.y, X_val, val.y, args.epochs, args.seed)
    print(f"best_params={best_params}")
    model = make_model(best_params, args.seed)
    history = run_epochs(model, X_train, train.y, X_val, val.y, args.epochs, args.seed)
    for row in history:
        print(f"epoch {row.epoch} {format_metrics('train', row.train)}")
        print(f"epoch {row.epoch} {format_metrics('val', row.val)}")

    train_scores = to_scores(model.predict(X_train))
    val_scores = to_scores(model.predict(X_val))
    threshold = select_threshold(
        val.y, val_scores, strategy=args.threshold_strategy, fixed=args.threshold
    )
    predictions = {"train": (train.y, train_scores), "val": (val.y, val_scores)}
    if X_test is not None:
        predictions["test"] = (test.y, to_scores(model.predict(X_test)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "model": "linear_regression_sgd",
                "best_params": best_params,
                "epochs": args.epochs,
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
    parser = argparse.ArgumentParser(description="Score unseen h5ads with a saved linear-regression run")
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
    X = feature_matrix(records, bundle["genes"])
    X = bundle["scaler"].transform(X)
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

