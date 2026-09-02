# scRNA-seq ICI heterogeneous graph baseline

This repository currently implements only the first Stage 1 milestone:

`sample-filtered AnnData -> sampled cells -> PyG HeteroData`

The graph contains `cell` and `gene` nodes, weighted non-zero
`cell -> gene` expression edges, and exact reverse edges. Stable global gene IDs
make gene identity consistent between local graphs. No cell-cell or gene-gene
edges are included.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
```

Expression must already be processed/log-normalised. The graph builder does not
normalise data, and it rejects negative expression weights.

## Minimal use

```python
from src.data import CellSampler, sample_from_anndata
from src.graph import GeneUniverse, build_local_graph, summarize_local_graph

sample = sample_from_anndata(
    sample_adata,
    sample_id="sample-1",
    patient_id="patient-1",
    label=1,
    cell_metadata_keys=("cell_type",),
)
cells = CellSampler(num_cells=512, seed=0).sample(sample.num_cells, training=True)
graph = build_local_graph(sample, cells, GeneUniverse(hvg_names))
print(summarize_local_graph(graph))
```

Patient-disjoint train/validation/test splits must be created before graph
generation. Cancer type is not required by the sample contract or graph.
