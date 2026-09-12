"""On-the-fly sample-level graphs from a patient-aware sample manifest."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.data.data_loader import (
    CELL_ANNOTATION_KEYS,
    DEFAULT_N_HVG,
    MAIN_CELL_TYPE_KEYS,
    AnnDataSample,
    SampleData,
    sample_from_anndata,
)
from src.data.sampler import SAMPLING_MODES, CellSampler, select_cells
from src.graph.build_global_graph import build_sample_global_graph
from src.graph.build_local_graph import GeneStrategy, GeneUniverse, build_local_graph

SamplingMode = Literal["random", "proportional", "by_cell_type", "whole_sample"]
CellTypeLevel = Literal["fine", "main"]

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


def annotation_keys_for(level: CellTypeLevel = "fine") -> tuple[str, str]:
    """Fine CellTypist labels, or Immune_All_High main types."""
    if level == "main":
        return MAIN_CELL_TYPE_KEYS
    if level != "fine":
        raise ValueError("cell_type_level must be 'fine' or 'main'")
    return CELL_ANNOTATION_KEYS


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
    require_label: bool = True,
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
                if require_label:
                    continue
                label = "UNKNOWN"
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


def records_from_h5ad_dir(
    dataset_root: str | Path,
    *,
    require_label: bool = True,
) -> list[SampleRecord]:
    """Build records from ``*.h5ad`` files and sibling ``*.metadata.csv`` rows."""
    root = Path(dataset_root)
    records: list[SampleRecord] = []
    for h5ad_path in sorted(root.glob("*.h5ad")):
        metadata_path = h5ad_path.with_name(h5ad_path.stem + ".metadata.csv")
        if metadata_path.is_file():
            with metadata_path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
        else:
            row = _obs_identity(h5ad_path)
        label = _canonical_label(row.get("Response", ""))
        if label is None:
            if require_label:
                continue
            label = "UNKNOWN"
        records.append(
            SampleRecord(
                h5ad_path=h5ad_path,
                metadata_path=metadata_path if metadata_path.is_file() else None,
                patient_id=str(row.get("patient_id", h5ad_path.stem)),
                sample_id=str(row.get("sample_id", h5ad_path.stem)),
                dataset_id=str(row.get("dataset_id", "")),
                label=label,
                tissue=str(row.get("Tissue", "")),
                ici_phase=str(row.get("ICI_phase", "")),
                cancer_type=str(row.get("Cancer type", "")),
                output_file=h5ad_path.name,
            )
        )
    if not records:
        raise ValueError(f"no usable h5ad samples in {root}")
    return records


def collect_records(
    dataset_root: str | Path,
    manifest_csv: str | Path = DEFAULT_MANIFEST,
    *,
    tissue: str | None = "Tumor",
    ici_phase: str | None = "pre",
    require_label: bool = True,
) -> list[SampleRecord]:
    """Load the manifest if it matches files under ``dataset_root``, else scan h5ads."""
    root = Path(dataset_root)
    if Path(manifest_csv).is_file():
        try:
            return load_sample_manifest(
                root,
                manifest_csv,
                tissue=tissue,
                ici_phase=ici_phase,
                skip_missing=True,
                require_label=require_label,
            )
        except ValueError:
            pass
    return records_from_h5ad_dir(root, require_label=require_label)


def _obs_identity(h5ad_path: Path) -> dict[str, str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs
        def _first(column: str, default: str) -> str:
            return str(obs[column].iloc[0]) if column in obs.columns else default

        return {
            "patient_id": _first("patient_id", h5ad_path.stem),
            "sample_id": _first("sample_id", h5ad_path.stem),
            "dataset_id": _first("dataset_id", ""),
            "Response": _first("Response", ""),
            "Tissue": _first("Tissue", ""),
            "ICI_phase": _first("ICI_phase", ""),
            "Cancer type": _first("Cancer type", ""),
        }
    finally:
        if adata.isbacked:
            adata.file.close()


def read_gene_names(h5ad_path: Path) -> tuple[str, ...]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        return tuple(str(name) for name in adata.var_names)
    finally:
        if adata.isbacked:
            adata.file.close()


def _item_gene_names(item: SampleData | SampleRecord) -> tuple[str, ...]:
    if isinstance(item, SampleRecord):
        return read_gene_names(item.h5ad_path)
    return tuple(str(name) for name in item.gene_names)


def build_gene_universe(items: Sequence[SampleData | SampleRecord]) -> GeneUniverse:
    """Stable gene IDs for training. Shared order is kept when every sample matches."""
    sequences = [_item_gene_names(item) for item in items]
    first = sequences[0]
    if all(names == first for names in sequences[1:]):
        return GeneUniverse(first)
    seen: dict[str, None] = {}
    for names in sequences:
        for name in names:
            seen.setdefault(name, None)
    return GeneUniverse(tuple(seen))


def load_record(
    record: SampleRecord,
    *,
    hvg_names: Sequence[str] | None = None,
    n_hvg: int = DEFAULT_N_HVG,
    load_normalised_expression: bool = True,
    load_cell_annotation: bool = True,
    load_clinical_metadata: bool = True,
) -> AnnDataSample:
    """Load one h5ad file into the Stage 1 sample contract."""
    import anndata as ad

    adata = ad.read_h5ad(record.h5ad_path)
    has_annotation = all(key in adata.obs.columns for key in CELL_ANNOTATION_KEYS) or all(
        key in adata.obs.columns for key in MAIN_CELL_TYPE_KEYS
    )
    return sample_from_anndata(
        adata,
        sample_id=record.sample_id,
        patient_id=record.patient_id,
        label=record.label,
        load_normalised_expression=load_normalised_expression,
        load_cell_annotation=load_cell_annotation and has_annotation,
        load_clinical_metadata=load_clinical_metadata,
        n_hvg=n_hvg,
        hvg_names=hvg_names,
    )


@dataclass(frozen=True)
class GraphView:
    """One sample-level graph, built from one or more local cell-gene graphs."""

    item_index: int
    cell_type: str | None = None
    cell_types: tuple[str, ...] = ()


def _unique_labels(values: Sequence[object]) -> tuple[str, ...]:
    names = []
    seen: set[str] = set()
    for value in values:
        name = str(value)
        if name in {"", "nan", "None", "NaN"} or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def item_cell_types(
    item: SampleData | SampleRecord,
    annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
) -> tuple[str, ...]:
    """Unique cell-type labels for a loaded sample or an on-disk record."""
    label_key = annotation_keys[0]
    if not isinstance(item, SampleRecord):
        if label_key not in item.cell_annotation:
            raise ValueError(f"by_cell_type requires {label_key!r} on the sample")
        return _unique_labels(item.cell_annotation[label_key])
    import anndata as ad

    adata = ad.read_h5ad(item.h5ad_path, backed="r")
    try:
        if label_key not in adata.obs.columns:
            raise ValueError(f"by_cell_type requires {label_key!r} in {item.h5ad_path}")
        return _unique_labels(adata.obs[label_key].astype(str).tolist())
    finally:
        if adata.isbacked:
            adata.file.close()


class SampleGraphDataset(Dataset):
    """Each access resamples cells, builds local graphs, then merges one sample HTG.

    Items may be already-loaded ``SampleData`` objects or ``SampleRecord`` rows
    from the manifest. Records are read from disk on demand.

    ``random`` uses ``CellSampler.sample``. ``proportional`` uses
    ``cell_type_proportional_sample``. ``whole_sample`` keeps every cell
    (``num_cells`` is ignored). ``by_cell_type`` builds one local graph per
    cell type (every cell of that type; ``num_cells`` is not applied) and
    aggregates them by sharing gene nodes. After cell indices are chosen, every
    mode calls the same ``build_local_graph`` path. Training redraws
    random/proportional cell subsets; validation uses a fixed ``view_index``.
    ``whole_sample`` is deterministic in both splits. Pass ``hvg_names`` to
    apply a frozen training-fold HVG list to every sample.
    """

    def __init__(
        self,
        items: Sequence[SampleData | SampleRecord],
        *,
        sampler: CellSampler,
        gene_universe: GeneUniverse,
        gene_strategy: GeneStrategy = "hvg",
        sampling_mode: SamplingMode = "random",
        annotation_keys: tuple[str, str] = CELL_ANNOTATION_KEYS,
        cell_types: Sequence[str] | None = None,
        training: bool = True,
        view_index: int = 0,
        seed: int = 0,
        n_hvg: int = DEFAULT_N_HVG,
        hvg_names: Sequence[str] | None = None,
        cache_samples: bool = False,
    ) -> None:
        if not items:
            raise ValueError("SampleGraphDataset requires at least one sample")
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(
                "sampling_mode must be 'random', 'proportional', 'by_cell_type', or 'whole_sample'"
            )
        self.items = list(items)
        self.sampler = sampler
        self.gene_universe = gene_universe
        self.gene_strategy = gene_strategy
        self.sampling_mode = sampling_mode
        self.annotation_keys = annotation_keys
        self.cell_types = None if not cell_types else tuple(str(name) for name in cell_types)
        self.training = training
        self.view_index = int(view_index)
        self.n_hvg = int(n_hvg)
        self.hvg_names = None if not hvg_names else tuple(str(name) for name in hvg_names)
        self.cache_samples = cache_samples
        self._rng = np.random.default_rng(seed)
        self._cache: dict[int, SampleData] = {}
        self.views = self._build_views()

    def _build_views(self) -> list[GraphView]:
        if self.sampling_mode != "by_cell_type":
            return [GraphView(index) for index in range(len(self.items))]
        allowed = None if self.cell_types is None else set(self.cell_types)
        views: list[GraphView] = []
        for index, item in enumerate(self.items):
            types = item_cell_types(item, self.annotation_keys)
            if allowed is not None:
                types = tuple(name for name in types if name in allowed)
            if not types:
                continue
            views.append(GraphView(index, cell_types=types))
        if not views:
            raise ValueError("by_cell_type produced no cell-type graphs")
        return views

    def __len__(self) -> int:
        return len(self.views)

    def frozen_hvgs(self) -> dict[str, tuple[str, ...]]:
        """Sample-id to the HVG names used to build that sample's graph."""
        mapping: dict[str, tuple[str, ...]] = {}
        for index in range(len(self.items)):
            sample = self._sample(index)
            mapping[str(sample.sample_id)] = tuple(str(name) for name in sample.hvg_names)
        return mapping

    def _sample(self, index: int) -> SampleData:
        item = self.items[index]
        if not isinstance(item, SampleRecord):
            if self.hvg_names is not None and isinstance(item, AnnDataSample):
                return replace(item, hvg_names=self.hvg_names)
            return item
        if self.cache_samples and index in self._cache:
            return self._cache[index]
        if self.hvg_names is not None:
            hvg_names = self.hvg_names
        elif self.gene_strategy == "hvg":
            hvg_names = None
        else:
            hvg_names = ()
        sample = load_record(item, hvg_names=hvg_names, n_hvg=self.n_hvg)
        if self.cache_samples:
            self._cache[index] = sample
        return sample

    def _cell_indices(self, sample: SampleData, cell_type: str | None) -> np.ndarray:
        return select_cells(
            sample,
            mode=self.sampling_mode,
            sampler=self.sampler,
            training=self.training,
            view_index=self.view_index,
            rng=self._rng if self.training else None,
            cell_type=cell_type,
            annotation_keys=self.annotation_keys,
        )

    def _local_graphs(self, sample: SampleData, view: GraphView) -> list[HeteroData]:
        if self.sampling_mode == "by_cell_type":
            types = view.cell_types or item_cell_types(sample, self.annotation_keys)
            if self.cell_types is not None:
                allowed = set(self.cell_types)
                types = tuple(name for name in types if name in allowed)
            graphs = [self._local_graph(sample, cell_type) for cell_type in types]
            if not graphs:
                raise ValueError("by_cell_type produced no cell-type graphs")
            return graphs
        return [self._local_graph(sample, view.cell_type)]

    def _local_graph(self, sample: SampleData, cell_type: str | None) -> HeteroData:
        cells = self._cell_indices(sample, cell_type)
        graph = build_local_graph(
            sample,
            cells,
            self.gene_universe,
            gene_strategy=self.gene_strategy,
        )
        graph["cell"].x = torch.ones((int(graph["cell"].num_nodes), 1), dtype=torch.float32)
        graph.cell_type = cell_type or ""
        return graph

    def __getitem__(self, index: int) -> HeteroData:
        view = self.views[index]
        sample = self._sample(view.item_index)
        return build_sample_global_graph(self._local_graphs(sample, view))


def _item_response(item: object) -> str:
    return str(getattr(item, "label", "UNKNOWN"))


def _patient_response(rows: Sequence[object]) -> str:
    """Majority R/NR for one patient. Ties become ``MIXED``."""
    counts: dict[str, int] = defaultdict(int)
    for item in rows:
        counts[_item_response(item)] += 1
    if len(counts) == 1:
        return next(iter(counts))
    top = max(counts.values())
    winners = sorted(name for name, count in counts.items() if count == top)
    return winners[0] if len(winners) == 1 else "MIXED"


def _fold_quotas(sizes: np.ndarray, take: int) -> np.ndarray:
    """Largest-remainder allocation of ``take`` slots across groups."""
    take = min(max(int(take), 0), int(sizes.sum()) if len(sizes) else 0)
    quotas = np.zeros(len(sizes), dtype=np.int64)
    if take == 0 or not len(sizes) or int(sizes.sum()) == 0:
        return quotas
    raw = take * sizes / sizes.sum()
    quotas = np.minimum(sizes, np.floor(raw)).astype(np.int64)
    leftover = take - int(quotas.sum())
    remainder = raw - np.floor(raw)
    for index in np.argsort(-remainder, kind="stable"):
        if leftover <= 0:
            break
        if quotas[index] < sizes[index]:
            quotas[index] += 1
            leftover -= 1
    return quotas


def split_by_patient(
    items: Sequence[SampleData | SampleRecord],
    *,
    val_fraction: float = 0.25,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[list, list] | tuple[list, list, list]:
    """Patient-disjoint split, then a light R/NR balance across folds.

    Patients are grouped by ``dataset_id::patient_id``. Fold sizes still follow
    ``val_fraction`` / ``test_fraction``. Within those sizes, R and NR patients
    are allocated in proportion to how often each class appears, so a seed
    shuffle cannot dump every responder into one fold.
    """
    if not 0 < val_fraction < 1 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must split (0, 1)")
    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[patient_key(item)].append(item)
    n_patients = len(grouped)
    if n_patients < 2:
        raise ValueError("need at least two patients to form a train/val split")
    n_test = min(max(int(round(n_patients * test_fraction)), 0), n_patients - 2)
    n_val = min(max(int(round(n_patients * val_fraction)), 1), n_patients - n_test - 1)

    by_label: dict[str, list[str]] = defaultdict(list)
    for key in sorted(grouped):
        by_label[_patient_response(grouped[key])].append(key)
    rng = np.random.default_rng(seed)
    strata = sorted(by_label)
    shuffled: list[list[str]] = []
    for label in strata:
        keys = np.array(by_label[label], dtype=object)
        rng.shuffle(keys)
        shuffled.append(keys.tolist())
    sizes = np.array([len(keys) for keys in shuffled], dtype=np.int64)
    test_take = _fold_quotas(sizes, n_test)
    remain = sizes - test_take
    val_take = _fold_quotas(remain, n_val)

    test_keys: set[str] = set()
    val_keys: set[str] = set()
    train_keys: set[str] = set()
    for keys, n_te, n_va in zip(shuffled, test_take.tolist(), val_take.tolist()):
        test_keys.update(keys[:n_te])
        val_keys.update(keys[n_te : n_te + n_va])
        train_keys.update(keys[n_te + n_va :])

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
