"""Smoke tests against the real sample h5ad files in data/test."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "htg_numba_cache"))

from src.data.data_loader import (
    CELL_ANNOTATION_KEYS,
    MAIN_CELL_TYPE_KEYS,
    annotate_cell_types,
    sample_from_anndata,
    select_sample_hvgs,
)
from src.data.sampler import CellSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPO_ROOT / "data" / "test"
OUTPUT_DIR = REPO_ROOT / "outputs"
N_HVG = 500
CELL_SIZES = (128, 256)


def _real_h5ad_paths() -> list[Path]:
    paths = sorted(REAL_DATA_DIR.glob("*.h5ad"))
    if len(paths) < 4:
        pytest.skip(f"expected 4 h5ad files in {REAL_DATA_DIR}, found {len(paths)}")
    return paths


def _ids_from_adata(adata) -> tuple[str, str, str]:
    return (
        str(adata.obs["sample_id"].iloc[0]),
        str(adata.obs["patient_id"].iloc[0]),
        str(adata.obs["Response"].iloc[0]),
    )


def _has_cell_annotation(adata) -> bool:
    return all(key in adata.obs.columns for key in CELL_ANNOTATION_KEYS)


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _print_table(title: str, columns: list[str], rows: list[list[object]]) -> None:
    widths = [len(column) for column in columns]
    rendered = []
    for row in rows:
        cells = [str(value) for value in row]
        rendered.append(cells)
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))
    print(f"\n{title}")
    print(" | ".join(column.ljust(widths[i]) for i, column in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for cells in rendered:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)))


def _graph_row(summary, graph, extra: list[object]) -> list[object]:
    return extra + [
        int(graph["cell"].num_nodes),
        int(graph["gene"].num_nodes),
        summary["num_expression_edges"],
        f"{summary['expression_density']:.4f}",
        f"{summary['min_edge_weight']:.3f}",
        f"{summary['max_edge_weight']:.3f}",
        summary["tensor_bytes"],
    ]


@pytest.fixture(scope="module")
def real_samples():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    paths = _real_h5ad_paths()
    adatas = [anndata.read_h5ad(path) for path in paths]
    integrated = anndata.concat(adatas, join="inner", index_unique="-")
    hvg_names = select_sample_hvgs(integrated, n_hvg=N_HVG)
    samples = []
    for adata in adatas:
        sample_id, patient_id, label = _ids_from_adata(adata)
        samples.append(
            sample_from_anndata(
                adata,
                sample_id=sample_id,
                patient_id=patient_id,
                label=label,
                load_normalised_expression=True,
                load_cell_annotation=_has_cell_annotation(adata),
                load_clinical_metadata=True,
                n_hvg=N_HVG,
                hvg_names=hvg_names,
            )
        )
    return {
        "paths": paths,
        "adatas": adatas,
        "hvg_names": hvg_names,
        "samples": samples,
    }


def test_real_samples_celltypist_annotations(real_samples):
    columns = [
        "file",
        "sample_id",
        "label",
        "n_cells",
        "has_predicted_labels",
        "has_conf_score",
        "n_cell_types",
    ]
    rows: list[list[object]] = []
    for path, adata, sample in zip(
        real_samples["paths"], real_samples["adatas"], real_samples["samples"]
    ):
        has_labels = "predicted_labels" in adata.obs.columns
        has_scores = "conf_score" in adata.obs.columns
        n_types = 0
        if has_labels:
            n_types = int(np.unique(np.asarray(adata.obs["predicted_labels"]).astype(str)).size)
        rows.append(
            [
                path.name,
                sample.sample_id,
                sample.label,
                sample.num_cells,
                has_labels,
                has_scores,
                n_types,
            ]
        )
    _write_csv(OUTPUT_DIR / "celltypist_annotations.csv", columns, rows)
    _print_table("CellTypist annotations in data/test", columns, rows)
    assert any(row[4] and row[5] for row in rows)


def test_real_samples_load_with_integrated_hvgs(real_samples):
    hvg_names = real_samples["hvg_names"]
    assert 0 < len(hvg_names) <= N_HVG
    labels = {sample.label for sample in real_samples["samples"]}
    assert len(real_samples["samples"]) >= 4
    assert labels <= {"R", "NR"}
    for adata, sample in zip(real_samples["adatas"], real_samples["samples"]):
        assert sample.num_cells == adata.n_obs
        assert sample.hvg_names == hvg_names
        if _has_cell_annotation(adata):
            assert "predicted_labels" in sample.cell_annotation
            labels = np.asarray(sample.cell_annotation["predicted_labels"]).astype(str)
            assert labels.size == sample.num_cells
        else:
            assert sample.cell_annotation == {}
        assert "Response" in sample.clinical_metadata


def test_real_samples_random_graph_sizes_128_and_256(real_samples):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.graph.build_local_graph import GeneUniverse, build_local_graph, summarize_local_graph

    universe = GeneUniverse(real_samples["samples"][0].gene_names)
    columns = [
        "sample",
        "label",
        "n_in_sample",
        "n_requested",
        "n_sampled",
        "gene_strategy",
        "cell_nodes",
        "gene_nodes",
        "edges",
        "density",
        "min_w",
        "max_w",
        "bytes",
    ]
    rows: list[list[object]] = []
    for num_cells in CELL_SIZES:
        sampler = CellSampler(num_cells=num_cells, seed=0)
        for sample in real_samples["samples"]:
            cells = sampler.sample(sample.num_cells, training=False)
            for strategy in ("expressed", "hvg"):
                graph = build_local_graph(sample, cells, universe, gene_strategy=strategy)
                summary = summarize_local_graph(graph)
                rows.append(
                    _graph_row(
                        summary,
                        graph,
                        [
                            sample.sample_id,
                            sample.label,
                            sample.num_cells,
                            num_cells,
                            len(cells),
                            strategy,
                        ],
                    )
                )
    _write_csv(OUTPUT_DIR / "graph_sizes_random.csv", columns, rows)
    _print_table("Random cell sampling: graph size by requested cells", columns, rows)


def test_real_samples_cell_type_graph_sizes(real_samples):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("celltypist")
    from src.graph.build_local_graph import GeneUniverse, build_local_graph, summarize_local_graph

    label_key, score_key = MAIN_CELL_TYPE_KEYS
    universe = GeneUniverse(real_samples["samples"][0].gene_names)
    sampler = CellSampler(seed=0)
    columns = [
        "sample",
        "label",
        "cell_type",
        "n_cells",
        "gene_strategy",
        "cell_nodes",
        "gene_nodes",
        "edges",
        "density",
        "min_w",
        "max_w",
        "bytes",
    ]
    rows: list[list[object]] = []
    count_rows: list[list[object]] = []
    summary_rows: list[list[object]] = []
    for path, adata, sample in zip(
        real_samples["paths"], real_samples["adatas"], real_samples["samples"]
    ):
        annotate_cell_types(adata, source_assembly="hg38")
        annotation = {
            label_key: np.asarray(adata.obs[label_key]),
            score_key: np.asarray(adata.obs[score_key], dtype=np.float64),
        }
        labels = np.asarray(annotation[label_key]).astype(str)
        scores = np.asarray(annotation[score_key], dtype=np.float64)
        counts = {str(name): int((labels == name).sum()) for name in np.unique(labels)}
        top_type = max(counts, key=lambda name: (counts[name], name))
        summary_rows.append(
            [
                path.name,
                sample.sample_id,
                sample.label,
                sample.num_cells,
                len(counts),
                top_type,
                counts[top_type],
                f"{counts[top_type] / sample.num_cells:.4f}",
                f"{float(np.nanmean(scores)):.4f}",
            ]
        )
        by_type = sampler.sample_all_cell_types(
            sample.num_cells,
            training=False,
            cell_annotation=annotation,
            annotation_keys=MAIN_CELL_TYPE_KEYS,
        )
        for cell_type, cells in by_type.items():
            n_cells = int((labels == cell_type).sum())
            count_rows.append(
                [
                    sample.sample_id,
                    sample.label,
                    cell_type,
                    n_cells,
                    f"{n_cells / sample.num_cells:.4f}",
                    f"{float(np.nanmean(scores[cells])):.4f}",
                ]
            )
            for strategy in ("expressed", "hvg"):
                graph = build_local_graph(sample, cells, universe, gene_strategy=strategy)
                summary = summarize_local_graph(graph)
                rows.append(
                    _graph_row(
                        summary,
                        graph,
                        [
                            sample.sample_id,
                            sample.label,
                            cell_type,
                            n_cells,
                            strategy,
                        ],
                    )
                )
    _write_csv(OUTPUT_DIR / "graph_sizes_cell_type.csv", columns, rows)
    _write_csv(
        OUTPUT_DIR / "celltypist_immune_all_high_summary.csv",
        [
            "file",
            "sample_id",
            "label",
            "n_cells",
            "n_high_types",
            "top_high_type",
            "top_high_n",
            "top_high_frac",
            "mean_conf",
        ],
        summary_rows,
    )
    _write_csv(
        OUTPUT_DIR / "celltypist_immune_all_high_counts.csv",
        ["sample", "label", "cell_type", "n_cells", "frac", "mean_conf"],
        count_rows,
    )
    _print_table(
        "Cell-type sampling: one graph per Immune_All_High main cell type",
        columns,
        rows,
    )
    assert rows
