"""Calculate classification metrics from an HTG predictions.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    args = parser.parse_args()

    with args.predictions_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    label_to_int = {"NR": 0, "R": 1}
    y_true = [label_to_int[row["label"]] for row in rows]
    y_pred = [label_to_int[row["pred"]] for row in rows]
    y_score = [float(row["prob_R"]) for row in rows]

    print(f"accuracy: {accuracy_score(y_true, y_pred):.6f}")
    print(f"auroc:    {roc_auc_score(y_true, y_score):.6f}")
    print(f"auprc:    {average_precision_score(y_true, y_score):.6f}")
    print(f"f1:       {f1_score(y_true, y_pred):.6f}")


if __name__ == "__main__":
    main()
