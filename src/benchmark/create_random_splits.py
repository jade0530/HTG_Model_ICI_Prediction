"""Create random patient-disjoint split.json files for benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.train.dataset import DEFAULT_MANIFEST, collect_records, patient_key, split_by_patient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-splits", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tissue", default="Tumor")
    parser.add_argument("--ici-phase", default="pre")
    args = parser.parse_args()

    records = collect_records(
        args.dataset_root,
        args.manifest,
        tissue=None if args.tissue.lower() == "all" else args.tissue,
        ici_phase=None if args.ici_phase.lower() == "all" else args.ici_phase,
    )
    split_seeds = np.random.default_rng().choice(10_000, size=args.num_splits, replace=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def rows(items):
        return [
            {
                "sample_id": item.sample_id,
                "patient_key": patient_key(item),
                "label": item.label,
            }
            for item in items
        ]

    for index, seed in enumerate(split_seeds, start=1):
        folds = split_by_patient(
            records,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=int(seed),
            cancer_proportional=True,
        )
        if args.test_fraction > 0:
            train, val, test = folds
        else:
            train, val = folds
            test = []
        payload = {
            "split_seed": int(seed),
            "train": rows(train),
            "val": rows(val),
            "test": rows(test),
        }
        path = args.output_dir / f"split_{index:03d}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"{path} seed={seed} train={len(train)} val={len(val)} test={len(test)}"
        )


if __name__ == "__main__":
    main()
