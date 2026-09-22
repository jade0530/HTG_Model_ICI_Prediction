from pathlib import Path
import os
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "htg_numba_cache"))

import anndata as ad
import numpy as np
from scipy import sparse

from src.benchmark.mlp import MLPConfig, train_mlp
from src.train.dataset import SampleRecord


def _write_h5ad(path: Path, matrix: np.ndarray, genes: tuple[str, ...]) -> None:
    adata = ad.AnnData(sparse.csr_matrix(matrix.astype(np.float32)))
    adata.var_names = list(genes)
    adata.obs_names = [f"c{i}" for i in range(matrix.shape[0])]
    adata.write_h5ad(path)


def _record(tmp_path: Path, sample_id: str, patient: str, label: str, matrix: np.ndarray, genes: tuple[str, ...]) -> SampleRecord:
    h5ad_path = tmp_path / f"{sample_id}.h5ad"
    _write_h5ad(h5ad_path, matrix, genes)
    return SampleRecord(
        h5ad_path=h5ad_path,
        metadata_path=None,
        patient_id=patient,
        sample_id=sample_id,
        dataset_id="toy",
        label=label,
        tissue="Tumor",
        ici_phase="pre",
        cancer_type="",
        output_file=h5ad_path.name,
    )


def test_train_mlp_writes_val_metrics(tmp_path: Path):
    genes = ("G1", "G2", "G3")
    rng = np.random.default_rng(0)
    records = [
        _record(tmp_path, "S1", "P1", "R", rng.random((8, 3)) + 1.0, genes),
        _record(tmp_path, "S2", "P2", "R", rng.random((8, 3)) + 1.0, genes),
        _record(tmp_path, "S3", "P3", "NR", rng.random((8, 3)) * 0.2, genes),
        _record(tmp_path, "S4", "P4", "NR", rng.random((8, 3)) * 0.2, genes),
    ]
    out = tmp_path / "mlp"
    result = train_mlp(
        records,
        config=MLPConfig(
            hidden_dims=(8, 4),
            n_hvg=3,
            epochs=5,
            patience=3,
            val_fraction=0.5,
            seed=0,
        ),
        output_dir=out,
        log=False,
    )
    assert (out / "model.pt").is_file()
    assert (out / "split.json").is_file()
    assert (out / "results.csv").is_file()
    assert (out / "val" / "predictions.csv").is_file()
    assert "val" in result["metrics"]
    assert result["n_train"] >= 1 and result["n_val"] >= 1
