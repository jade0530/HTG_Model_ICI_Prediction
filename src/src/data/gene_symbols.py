"""Map aligned Ensembl gene IDs to symbols once, before cell sampling.

Samples that share one gene panel reuse a single column plan. A panel that is
already gene symbols is left unchanged. An Ensembl ID with no symbol stays as
that exact ID.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

HG38_GFF3 = Path(
    "/media/rokny/DATA3/Jade/scRNA-seq_dataset_collection/support_files/Homo_sapiens.GRCh38.115.gff3"
)

_ensembl_to_symbol: dict[str, str] | None = None
_plans: dict[tuple[str, ...], tuple[tuple[str, ...], sparse.csr_matrix | None, tuple[int, ...] | None]] = {}


def _has_ensg(names: Sequence[str]) -> bool:
    return any(name.split(".", 1)[0].startswith("ENSG") for name in names)


def load_ensembl_to_symbol(annotation_path: Path = HG38_GFF3) -> dict[str, str]:
    global _ensembl_to_symbol
    if _ensembl_to_symbol is not None:
        return _ensembl_to_symbol
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
    _ensembl_to_symbol = mapping
    return mapping


def _symbol(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name.split(".", 1)[0]) or name


def symbol_plan(
    gene_names: Sequence[str],
) -> tuple[tuple[str, ...], sparse.csr_matrix | None, tuple[int, ...] | None]:
    """Symbols for one shared panel.

    The aggregator is present only when several Ensembl IDs share a symbol.
    ``keep`` is the first source column of each output symbol in that case.
    """
    names = tuple(str(name) for name in gene_names)
    cached = _plans.get(names)
    if cached is not None:
        return cached
    if not names or not _has_ensg(names):
        plan = (names, None, None)
        _plans[names] = plan
        return plan

    mapping = load_ensembl_to_symbol()
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, name in enumerate(names):
        symbol = _symbol(name, mapping)
        if symbol not in groups:
            groups[symbol] = [index]
            order.append(symbol)
        else:
            groups[symbol].append(index)
    symbols = tuple(order)
    if len(symbols) == len(names):
        plan = (symbols, None, None)
        _plans[names] = plan
        return plan

    rows: list[int] = []
    cols: list[int] = []
    keep: list[int] = []
    for column, symbol in enumerate(symbols):
        members = groups[symbol]
        keep.append(members[0])
        for row in members:
            rows.append(row)
            cols.append(column)
    aggregator = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(names), len(symbols)),
    )
    plan = (symbols, aggregator, tuple(keep))
    _plans[names] = plan
    return plan


def map_gene_names(gene_names: Sequence[str]) -> tuple[str, ...]:
    return symbol_plan(gene_names)[0]


def apply_expression(matrix, gene_names: Sequence[str]):
    """Return expression with symbol columns and the symbol names."""
    symbols, aggregator, _keep = symbol_plan(gene_names)
    if aggregator is None:
        return matrix, symbols
    if sparse.issparse(matrix):
        return matrix @ aggregator, symbols
    collapsed = sparse.csr_matrix(np.asarray(matrix)) @ aggregator
    return np.asarray(collapsed.toarray()), symbols


def map_adata_gene_symbols(adata):
    """Rename an aligned gene panel to symbols before cells are sampled."""
    names = tuple(str(name) for name in adata.var_names)
    symbols, aggregator, keep = symbol_plan(names)
    if symbols == names:
        return adata
    if aggregator is None:
        adata.var_names = list(symbols)
        return adata

    import anndata as ad

    var = adata.var.iloc[list(keep)].copy()
    var.index = list(symbols)
    mapped = ad.AnnData(
        X=apply_expression(adata.X, names)[0],
        obs=adata.obs.copy(),
        var=var,
    )
    for key, layer in adata.layers.items():
        mapped.layers[key] = apply_expression(layer, names)[0]
    mapped.obsm = adata.obsm.copy()
    mapped.obsp = adata.obsp.copy()
    mapped.uns = adata.uns.copy()
    return mapped
