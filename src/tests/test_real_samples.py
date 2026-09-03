"""Smoke tests against the four real Bassez sample h5ad files in data/test."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "htg_numba_cache"))

from src.data.data_loader import sample_from_anndata, select_sample_hvgs
from src.data.sampler import CellSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPO_ROOT / "data" / "test"
N_HVG = 500
N_CELLS = 64


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


@pytest.fixture(scope="module")
def real_bassez_samples():
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
                load_clinical_metadata=True,
                n_hvg=N_HVG,
                hvg_names=hvg_names,
            )
        )
    return {"paths": paths, "adatas": adatas, "hvg_names": hvg_names, "samples": samples}


def test_real_bassez_samples_load_with_integrated_hvgs(real_bassez_samples):
    hvg_names = real_bassez_samples["hvg_names"]
    assert 0 < len(hvg_names) <= N_HVG
    patient_ids = {sample.patient_id for sample in real_bassez_samples["samples"]}
    labels = {sample.label for sample in real_bassez_samples["samples"]}
    assert len(real_bassez_samples["samples"]) == 4
    assert patient_ids == {"BIOKEY_8", "BIOKEY_9", "BIOKEY_10", "BIOKEY_11"}
    assert labels == {"R", "NR"}
    for adata, sample in zip(real_bassez_samples["adatas"], real_bassez_samples["samples"]):
        assert sample.num_cells == adata.n_obs
        assert sample.label in {"R", "NR"}
        assert sample.hvg_names == hvg_names
        assert "predicted_labels" not in sample.cell_annotation
        assert "Response" in sample.clinical_metadata
        assert "sample_id" in sample.clinical_metadata


def test_real_bassez_samples_build_expressed_and_hvg_graphs(real_bassez_samples):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.graph.build_local_graph import GeneUniverse, build_local_graph, summarize_local_graph

    sampler = CellSampler(num_cells=N_CELLS, seed=0)
    universe = GeneUniverse(real_bassez_samples["samples"][0].gene_names)
    for sample in real_bassez_samples["samples"]:
        cells = sampler.sample(sample.num_cells, training=False)
        assert len(cells) == min(N_CELLS, sample.num_cells)
        expressed = build_local_graph(sample, cells, universe, gene_strategy="expressed")
        hvg = build_local_graph(sample, cells, universe, gene_strategy="hvg")
        expressed_summary = summarize_local_graph(expressed)
        hvg_summary = summarize_local_graph(hvg)
        print(
            f"\n{sample.sample_id} ({sample.label}, "
            f"{sample.num_cells} cells in sample, {len(cells)} sampled)"
            f"\n  expressed: cell_nodes={expressed['cell'].num_nodes} "
            f"gene_nodes={expressed['gene'].num_nodes} "
            f"edges={expressed_summary['num_expression_edges']} "
            f"density={expressed_summary['expression_density']:.4f} "
            f"weight=[{expressed_summary['min_edge_weight']:.3f}, "
            f"{expressed_summary['max_edge_weight']:.3f}] "
            f"bytes={expressed_summary['tensor_bytes']}"
            f"\n  hvg:       cell_nodes={hvg['cell'].num_nodes} "
            f"gene_nodes={hvg['gene'].num_nodes} "
            f"edges={hvg_summary['num_expression_edges']} "
            f"density={hvg_summary['expression_density']:.4f} "
            f"weight=[{hvg_summary['min_edge_weight']:.3f}, "
            f"{hvg_summary['max_edge_weight']:.3f}] "
            f"bytes={hvg_summary['tensor_bytes']}"
        )
        assert expressed["cell"].num_nodes == len(cells)
        assert hvg["cell"].num_nodes == len(cells)
        assert expressed_summary["num_genes"] >= hvg_summary["num_genes"]
        assert hvg_summary["num_expression_edges"] > 0
        assert (hvg["cell", "expresses", "gene"].edge_weight > 0).all()
        assert hvg["gene", "expressed_by", "cell"].edge_index.equal(
            hvg["cell", "expresses", "gene"].edge_index.flip(0)
        )
        assert hvg.sample_id == sample.sample_id
        assert hvg.patient_id == sample.patient_id
        assert hvg.y.tolist() == [1.0 if sample.label == "R" else 0.0]
        universe_names = np.asarray(universe.names)
        hvg_gene_names = universe_names[hvg["gene"].global_id.numpy()]
        assert set(hvg_gene_names).issubset(sample.hvg_names)
