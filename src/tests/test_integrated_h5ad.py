from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.train.dataset import collect_records, load_record


def _write_integrated(path: Path) -> None:
    import anndata as ad
    import pandas as pd

    obs = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "C"],
            "patient_id": ["p1", "p1", "p2", "p3"],
            "dataset_id": ["D", "D", "D", "D"],
            "Response": ["R", "R", "NR", "R"],
            "Tissue": ["Tumor", "Tumor", "Tumor", "Normal"],
            "ICI_phase": ["pre", "pre", "pre", "pre"],
            "Cancer type": ["TNBC", "TNBC", "CRC", "TNBC"],
            "predicted_labels": ["T", "T", "B", "T"],
            "conf_score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    matrix = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [0, 5, 0],
            [0, 0, 9],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(matrix, obs=obs)
    adata.var_names = ["G1", "G2", "G3"]
    adata.write_h5ad(path)


def test_dataset_root_accepts_one_integrated_h5ad(tmp_path: Path) -> None:
    path = tmp_path / "integrated.h5ad"
    _write_integrated(path)
    records = collect_records(path, manifest_csv=tmp_path / "missing.csv")
    by_id = {record.sample_id: record for record in records}
    assert set(by_id) == {"A", "B"}
    assert by_id["A"].obs_key == "sample_id"
    assert by_id["A"].h5ad_path == path
    assert by_id["A"].patient_id == "p1"
    assert by_id["A"].label == "R"
    assert by_id["A"].cancer_type == "TNBC"
    assert by_id["B"].label == "NR"

    loaded = load_record(by_id["A"], hvg_names=(), map_gene_symbols=False)
    assert loaded.X.shape == (2, 3)
    assert float(np.asarray(loaded.X)[0, 0]) == pytest.approx(1)
    other = load_record(by_id["B"], hvg_names=(), map_gene_symbols=False)
    assert other.X.shape == (1, 3)
    assert float(np.asarray(other.X)[0, 1]) == pytest.approx(5)


def test_integrated_h5ad_rejects_a_non_h5ad_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not an h5ad")
    with pytest.raises(ValueError, match="h5ad"):
        collect_records(path, manifest_csv=tmp_path / "missing.csv")
