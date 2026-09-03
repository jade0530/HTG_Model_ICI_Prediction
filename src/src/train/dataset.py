"""On-the-fly sample-level graphs from a patient-aware sample manifest."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.data.data_loader import DEFAULT_N_HVG, AnnDataSample, SampleData, sample_from_anndata
from src.data.sampler import CellSampler
from src.graph.build_local_graph import GeneStrategy, GeneUniverse, build_local_graph

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "sample_manifest.csv"
LABEL_ALIASES = {
    "R": "R",
    "NR": "NR",
    "RESPONSE": "R",
    "NON-RESPONSE": "NR",
    "NONRESPONSE": "NR",
}


@dataclass(frozen=True)
class SampleRecord:
    """One manifest row: disk location plus patient/sample identity."""

    h5ad_path: Path
    metadata_path: Path | None
    patient_id: str
    sample_id: str
    dataset_id: str
    label: str
    tissue: str
    ici_phase: str
    cancer_type: str
    output_file: str

    @property
    def patient_key(self) -> str:
        """Stable patient identity across a patient's multiple samples."""
        return f"{self.dataset_id}::{self.patient_id}"


def patient_key(item: object) -> str:
    """Return the leakage-safe patient key for a record or loaded sample."""
    key = getattr(item, "patient_key", None)
    if key:
        return str(key)
    dataset_id = getattr(item, "dataset_id", None)
    pid = str(getattr(item, "patient_id"))
    return f"{dataset_id}::{pid}" if dataset_id else pid


def _normalise(value: object) -> str:
    return str(value).strip().lower()


def _canonical_label(value: object) -> str | None:
    return LABEL_ALIASES.get(str(value).strip().upper())


def _resolve_path(dataset_root: Path, relative: str) -> Path:
    rel = Path(str(relative))
    direct = dataset_root / rel
    if direct.exists():
        return direct
    by_name = dataset_root / rel.name
    if by_name.exists():
        return by_name
    return direct


def load_sample_manifest(
    dataset_root: str | Path,
    manifest_csv: str | Path = DEFAULT_MANIFEST,
    *,
    tissue: str | None = "Tumor",
    ici_phase: str | None = "pre",
    skip_missing: bool = True,
) -> list[SampleRecord]:
    """Read ``sample_manifest.csv`` and keep rows that exist under ``dataset_root``.

    ``output_file`` in the CSV is relative to ``dataset_root``, for example::

        /media/rokny/DATA6/Jade/sc_training_dataset_latest
        + sample_h5ad/Tumor/pre/<file>.h5ad

    Default filters match the Tumor/pre training folder. Pass ``tissue=None``
    and ``ici_phase=None`` to keep every row. Labels other than R/NR (and
    obvious aliases) are dropped so the sample contract stays binary.
    """
    root = Path(dataset_root)
    manifest_path = Path(manifest_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sample manifest not found: {manifest_path}")
    tissue_key = None if tissue is None else _normalise(tissue)
    phase_key = None if ici_phase is None else _normalise(ici_phase)
    records: list[SampleRecord] = []
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = _canonical_label(row.get("Response", ""))
            if label is None:
                continue
            if tissue_key is not None and _normalise(row.get("Tissue", "")) != tissue_key:
                continue
            if phase_key is not None and _normalise(row.get("ICI_phase", "")) != phase_key:
                continue
            h5ad_path = _resolve_path(root, row["output_file"])
            if skip_missing and not h5ad_path.is_file():
                continue
            metadata = _resolve_path(root, row["metadata_file"]) if row.get("metadata_file") else None
            records.append(
                SampleRecord(
                    h5ad_path=h5ad_path,
                    metadata_path=metadata if metadata is not None and metadata.is_file() else None,
                    patient_id=str(row["patient_id"]),
                    sample_id=str(row["sample_id"]),
                    dataset_id=str(row["dataset_id"]),
                    label=label,
                    tissue=str(row.get("Tissue", "")),
                    ici_phase=str(row.get("ICI_phase", "")),
                    cancer_type=str(row.get("Cancer type", "")),
                    output_file=str(row["output_file"]),
                )
            )
    if not records:
        raise ValueError(
            f"no usable samples in {manifest_path} under {root} "
            f"(tissue={tissue!r}, ici_phase={ici_phase!r})"
        )
    return records


def load_record(
    record: SampleRecord,
    *,
    hvg_names: Sequence[str] | None = None,
    n_hvg: int = DEFAULT_N_HVG,
    load_normalised_expression: bool = True,
    load_cell_annotation: bool = False,
    load_clinical_metadata: bool = True,
) -> AnnDataSample:
    """Load one h5ad file into the Stage 1 sample contract."""
    import anndata as ad

    adata = ad.read_h5ad(record.h5ad_path)
    return sample_from_anndata(
        adata,
        sample_id=record.sample_id,
        patient_id=record.patient_id,
        label=record.label,
        load_normalised_expression=load_normalised_expression,
        load_cell_annotation=load_cell_annotation,
        load_clinical_metadata=load_clinical_metadata,
        n_hvg=n_hvg,
        hvg_names=hvg_names,
    )


class SampleGraphDataset(Dataset):
    """Each access resamples cells and builds one sample-level graph.

    Items may be already-loaded ``SampleData`` objects or ``SampleRecord`` rows
    from the manifest. Records are read from disk on demand. Training draws a
    new random cell subset every time; validation uses a fixed ``view_index``.
    """

    def __init__(
        self,
        items: Sequence[SampleData | SampleRecord],
        *,
        sampler: CellSampler,
        gene_universe: GeneUniverse,
        gene_strategy: GeneStrategy = "hvg",
        training: bool = True,
        view_index: int = 0,
        seed: int = 0,
        hvg_names: Sequence[str] | None = None,
        n_hvg: int = DEFAULT_N_HVG,
        cache_samples: bool = False,
    ) -> None:
        if not items:
            raise ValueError("SampleGraphDataset requires at least one sample")
        self.items = list(items)
        self.sampler = sampler
        self.gene_universe = gene_universe
        self.gene_strategy = gene_strategy
        self.training = training
        self.view_index = int(view_index)
        self.hvg_names = None if hvg_names is None else tuple(hvg_names)
        self.n_hvg = int(n_hvg)
        self.cache_samples = cache_samples
        self._rng = np.random.default_rng(seed)
        self._cache: dict[int, SampleData] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _sample(self, index: int) -> SampleData:
        item = self.items[index]
        if not isinstance(item, SampleRecord):
            return item
        if self.cache_samples and index in self._cache:
            return self._cache[index]
        sample = load_record(item, hvg_names=self.hvg_names, n_hvg=self.n_hvg)
        if self.cache_samples:
            self._cache[index] = sample
        return sample

    def __getitem__(self, index: int) -> HeteroData:
        sample = self._sample(index)
        cells = self.sampler.sample(
            sample.num_cells,
            training=self.training,
            view_index=self.view_index,
            rng=self._rng if self.training else None,
        )
        graph = build_local_graph(
            sample,
            cells,
            self.gene_universe,
            gene_strategy=self.gene_strategy,
        )
        graph["cell"].x = torch.ones((int(graph["cell"].num_nodes), 1), dtype=torch.float32)
        return graph


def split_by_patient(
    items: Sequence[SampleData | SampleRecord],
    *,
    val_fraction: float = 0.25,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[list, list] | tuple[list, list, list]:
    """Patient-disjoint split. Every sample of a patient stays in one fold.

    The grouping key is ``dataset_id::patient_id`` when both are available, so
    ``A01`` in two studies is not treated as one person, while ``A01_neg`` and
    ``A01_pos`` from the same study cannot leak across train/val/test.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")
    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[patient_key(item)].append(item)
    keys = np.array(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_patients = len(keys)
    if n_patients < 2:
        raise ValueError("need at least two patients to form a train/val split")
    n_test = int(round(n_patients * test_fraction))
    n_val = int(round(n_patients * val_fraction))
    n_test = min(max(n_test, 0), n_patients - 2)
    n_val = min(max(n_val, 1), n_patients - n_test - 1)
    test_keys = set(keys[:n_test].tolist())
    val_keys = set(keys[n_test : n_test + n_val].tolist())
    train_keys = set(keys[n_test + n_val :].tolist())
    if not train_keys or not val_keys:
        raise ValueError("need at least two patients to form a train/val split")
    if not train_keys.isdisjoint(val_keys) or not train_keys.isdisjoint(test_keys) or not val_keys.isdisjoint(test_keys):
        raise RuntimeError("patient split leaked across folds")
    train = [item for key, rows in grouped.items() if key in train_keys for item in rows]
    val = [item for key, rows in grouped.items() if key in val_keys for item in rows]
    if test_fraction <= 0:
        return train, val
    test = [item for key, rows in grouped.items() if key in test_keys for item in rows]
    return train, val, test


def assert_patient_disjoint(*folds: Sequence[object]) -> None:
    """Raise if any patient key appears in more than one fold."""
    seen: dict[str, int] = {}
    for fold_index, fold in enumerate(folds):
        for item in fold:
            key = patient_key(item)
            previous = seen.get(key)
            if previous is not None and previous != fold_index:
                raise ValueError(f"patient {key} appears in more than one split")
            seen[key] = fold_index
