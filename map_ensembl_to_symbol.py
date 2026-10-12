"""Map sample var_names from Ensembl gene IDs to symbols with the hg38 GFF3.

Usage:
    python map_ensembl_to_symbol.py \
        --input-dir /path/to/samples \
        --output-dir /path/to/symbols \
        --workers 4
"""

from __future__ import annotations

import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

HG38_GFF3 = Path(
    "/media/rokny/DATA3/Jade/scRNA-seq_dataset_collection/support_files/Homo_sapiens.GRCh38.115.gff3"
)


def load_ensembl_to_symbol(annotation_path: Path) -> dict[str, str]:
    """Return unversioned ENSG IDs mapped to gene symbols from gene rows."""
    opener = gzip.open if annotation_path.suffix == ".gz" else open
    mapping: dict[str, str] = {}
    with opener(annotation_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = {}
            for item in fields[8].split(";"):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                attrs[key] = value
            gene_id = attrs.get("gene_id", "").split(".", 1)[0]
            symbol = attrs.get("Name") or attrs.get("gene_name", "")
            if gene_id and symbol and gene_id not in mapping:
                mapping[gene_id] = symbol
    return mapping


def _sum_columns(matrix, columns: list[int]):
    block = matrix[:, columns]
    if sparse.issparse(block):
        return np.asarray(block.sum(axis=1)).ravel()
    return np.asarray(block).sum(axis=1)


def gene_label(name: str, ensembl_to_symbol: dict[str, str]) -> str:
    """Use the gene symbol. An Ensembl ID with no symbol is returned unchanged."""
    symbol = ensembl_to_symbol.get(name.split(".", 1)[0])
    if not symbol:
        return name
    return symbol


def map_var_names(adata: ad.AnnData, ensembl_to_symbol: dict[str, str]) -> ad.AnnData:
    """Rename genes to symbols. Columns that share a symbol are summed."""
    original = [str(name) for name in adata.var_names]
    symbols = [gene_label(name, ensembl_to_symbol) for name in original]

    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, symbol in enumerate(symbols):
        if symbol not in groups:
            groups[symbol] = []
            order.append(symbol)
        groups[symbol].append(index)

    if all(len(columns) == 1 for columns in groups.values()):
        adata.var_names = symbols
        adata.var["ensembl_id"] = original
        return adata

    var = adata.var.iloc[[groups[symbol][0] for symbol in order]].copy()
    var.index = order
    var["ensembl_id"] = [";".join(original[i] for i in groups[symbol]) for symbol in order]

    mapped = ad.AnnData(X=_collapsed(adata.X, order, groups), obs=adata.obs.copy(), var=var)
    for name, layer in adata.layers.items():
        mapped.layers[name] = _collapsed(layer, order, groups)
    mapped.obsm = adata.obsm.copy()
    mapped.obsp = adata.obsp.copy()
    mapped.uns = adata.uns.copy()
    return mapped


def _collapsed(matrix, order: list[str], groups: dict[str, list[int]]):
    columns = [_sum_columns(matrix, groups[symbol]) for symbol in order]
    stacked = np.column_stack(columns)
    if sparse.issparse(matrix):
        return sparse.csr_matrix(stacked)
    return stacked


def convert_sample(sample_path: Path, output_dir: Path, ensembl_to_symbol: dict[str, str]) -> str:
    adata = ad.read_h5ad(sample_path)
    gene_ids = [str(name).split(".", 1)[0] for name in adata.var_names]
    n_mapped = sum(gene_id in ensembl_to_symbol for gene_id in gene_ids)
    mapped = map_var_names(adata, ensembl_to_symbol)
    out_path = output_dir / sample_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.write_h5ad(out_path)
    n_unmapped = adata.n_vars - n_mapped
    percent = 100.0 * n_mapped / adata.n_vars
    return (
        f"{sample_path.name}: {adata.n_vars} genes -> {mapped.n_vars} symbols, "
        f"mapped {n_mapped}/{adata.n_vars} ({percent:.1f}%), "
        f"left as Ensembl ID {n_unmapped}"
    )


def convert_folder(input_dir: Path, output_dir: Path, gtf_path: Path, workers: int) -> None:
    ensembl_to_symbol = load_ensembl_to_symbol(gtf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted(input_dir.glob("*.h5ad"))
    if workers == 1:
        for sample_path in samples:
            print(convert_sample(sample_path, output_dir, ensembl_to_symbol), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(convert_sample, sample_path, output_dir, ensembl_to_symbol)
            for sample_path in samples
        ]
        for job in as_completed(jobs):
            print(job.result(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, default=HG38_GFF3)
    parser.add_argument("--workers", type=int, default=4, help="Samples mapped at the same time")
    args = parser.parse_args()
    convert_folder(args.input_dir, args.output_dir, args.gtf, args.workers)


if __name__ == "__main__":
    main()
