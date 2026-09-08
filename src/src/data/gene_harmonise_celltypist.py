#!/usr/bin/env python3
"""
Version-aware human gene harmonisation plus CellTypist annotation.

Public entry point
------------------
harmonise_and_annotate(...) -> anndata.AnnData

The returned AnnData uses unversioned GENCODE v49 / Ensembl release 115
GRCh38 ENSG gene IDs as ``var_names``.  Raw counts are retained in
``layers["counts"]`` when a count matrix is available; ``X`` is log1p
normalised to 10,000 counts per cell for CellTypist.

The implementation is deliberately conservative:

* source assembly metadata is not silently replaced by a gene-list guess;
* mappings are attempted in an auditable order;
* one-to-many mappings are dropped rather than splitting expression;
* many-to-one mappings are summed only when raw counts are available;
* ambiguous and unmapped features remain visible in the mapping audit.
"""

from __future__ import annotations

import copy
import gzip
import re
import tempfile
import urllib.parse
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse


PathLike = Union[str, Path]

ENSG_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$")
STANDARD_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}

ID_COLUMN_CANDIDATES = (
    "gene_ids",
    "gene_id",
    "ensembl_id",
    "ensembl_gene_id",
    "Gene ID",
)
SYMBOL_COLUMN_CANDIDATES = (
    "gene_symbols",
    "gene_symbol",
    "gene_name",
    "symbol",
    "Gene Symbol",
)
COUNT_LAYER_CANDIDATES = ("counts", "raw_counts", "count", "raw")


def _open_text(path: PathLike):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def _strip_ensembl_version(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^(?:gene:|Gene:)", "", text)
    return re.sub(r"\.\d+$", "", text)


def _normalise_chrom(value: Any) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom in {"M", "Mt", "mt", "m"}:
        chrom = "MT"
    return chrom


def _parse_gtf_attributes(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in re.findall(r'([A-Za-z0-9_.:-]+)\s+"([^"]*)"\s*;', text):
        attrs[key] = value
    return attrs


def _parse_gff3_attributes(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in text.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        attrs[key] = urllib.parse.unquote(value)
    return attrs


def _first_present(attrs: Mapping[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, "", "NA", "None"):
            return str(value)
    return ""


def read_gene_annotation(
    path: PathLike,
    *,
    standard_chromosomes_only: bool = False,
) -> pd.DataFrame:
    """
    Read gene-level records from a GENCODE GTF or Ensembl/GENCODE GFF3.

    Returns columns:
      gene_id_full, gene_id, gene_symbol, chrom, start, end, strand, biotype
    """
    return _read_gene_annotation_cached(
        str(Path(path).resolve()), standard_chromosomes_only
    ).copy()


@lru_cache(maxsize=8)
def _read_gene_annotation_cached(
    path: str,
    standard_chromosomes_only: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2].lower() != "gene":
                continue

            chrom = _normalise_chrom(fields[0])
            if standard_chromosomes_only and chrom not in STANDARD_CHROMS:
                continue

            attrs_text = fields[8]
            attrs = (
                _parse_gff3_attributes(attrs_text)
                if "=" in attrs_text and not re.search(r'\w+\s+"', attrs_text)
                else _parse_gtf_attributes(attrs_text)
            )
            gene_id_full = _first_present(
                attrs, ("gene_id", "ID", "gene", "stable_id")
            )
            gene_id_full = re.sub(r"^(?:gene:|Gene:)", "", gene_id_full)
            gene_id = _strip_ensembl_version(gene_id_full)
            if not ENSG_RE.match(gene_id):
                continue

            symbol = _first_present(
                attrs, ("gene_name", "Name", "gene_symbol", "display_name")
            )
            biotype = _first_present(
                attrs, ("gene_type", "gene_biotype", "biotype")
            )
            records.append(
                {
                    "gene_id_full": gene_id_full,
                    "gene_id": gene_id,
                    "gene_symbol": symbol,
                    "chrom": chrom,
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "biotype": biotype,
                }
            )

    if not records:
        raise ValueError(f"No gene records were parsed from annotation: {path}")

    result = pd.DataFrame.from_records(records)
    result["annotation_file"] = str(path)
    return result


def _unique_value_map(
    table: pd.DataFrame,
    key_col: str,
    value_col: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    grouped = table.loc[
        table[key_col].astype(str).ne("") & table[value_col].astype(str).ne(""),
        [key_col, value_col],
    ].drop_duplicates()
    for key, group in grouped.groupby(key_col, sort=False):
        values = pd.unique(group[value_col].astype(str))
        if len(values) == 1:
            mapping[str(key)] = str(values[0])
    return mapping


def _first_gene_record_by_id(table: pd.DataFrame) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    ordered = table.assign(
        _is_standard=table["chrom"].isin(STANDARD_CHROMS).astype(int)
    ).sort_values(["_is_standard"], ascending=False)
    for gene_id, group in ordered.groupby("gene_id", sort=False):
        result[str(gene_id)] = group.iloc[0]
    return result


def _fraction_matching_ensg(values: Iterable[Any]) -> float:
    values = [str(x).strip() for x in values]
    if not values:
        return 0.0
    return float(np.mean([bool(ENSG_RE.match(x)) for x in values]))


def _find_column(
    var: pd.DataFrame,
    explicit: Optional[str],
    candidates: Sequence[str],
    *,
    require_ensg: bool,
) -> Optional[str]:
    if explicit is not None:
        if explicit not in var.columns:
            raise KeyError(f"Requested var column {explicit!r} does not exist")
        return explicit
    for column in candidates:
        if column not in var.columns:
            continue
        if not require_ensg or _fraction_matching_ensg(var[column]) >= 0.60:
            return column
    return None


def _detect_feature_namespace(
    var: pd.DataFrame,
    var_names: Sequence[Any],
    *,
    gene_id_col: Optional[str],
    gene_symbol_col: Optional[str],
) -> tuple[str, pd.Series, pd.Series, str]:
    """
    Return feature kind, primary feature values, companion symbols and source.
    """
    id_col = _find_column(
        var, gene_id_col, ID_COLUMN_CANDIDATES, require_ensg=True
    )
    symbol_col = _find_column(
        var, gene_symbol_col, SYMBOL_COLUMN_CANDIDATES, require_ensg=False
    )

    index_values = pd.Series(list(map(str, var_names)), index=var.index, dtype="string")
    if id_col is not None:
        primary = var[id_col].astype("string")
        symbols = (
            var[symbol_col].astype("string")
            if symbol_col is not None
            else pd.Series("", index=var.index, dtype="string")
        )
        return "ensembl_id", primary, symbols, f"var[{id_col!r}]"

    if _fraction_matching_ensg(index_values) >= 0.60:
        symbols = (
            var[symbol_col].astype("string")
            if symbol_col is not None
            else pd.Series("", index=var.index, dtype="string")
        )
        return "ensembl_id", index_values, symbols, "var_names"

    primary = (
        var[symbol_col].astype("string")
        if symbol_col is not None
        else index_values
    )
    return "gene_symbol", primary, primary.copy(), (
        f"var[{symbol_col!r}]" if symbol_col is not None else "var_names"
    )


def _sample_matrix_values(matrix: Any, max_values: int = 100_000) -> np.ndarray:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        values = np.asarray(matrix).ravel()
    if values.size > max_values:
        step = max(1, values.size // max_values)
        values = values[::step][:max_values]
    return values.astype(float, copy=False)


def _is_nonnegative_integer_like(matrix: Any) -> bool:
    values = _sample_matrix_values(matrix)
    if values.size == 0:
        return True
    if not np.all(np.isfinite(values)) or np.min(values) < 0:
        return False
    return bool(np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6))


def _select_expression(
    adata: Any,
    counts_layer: Optional[str],
) -> tuple[Any, pd.DataFrame, Sequence[Any], str, str]:
    """
    Return matrix, matching var, matching var_names, matrix label, expression kind.
    """
    if counts_layer not in (None, "auto"):
        if counts_layer not in adata.layers:
            raise KeyError(f"counts layer {counts_layer!r} is missing")
        matrix = adata.layers[counts_layer]
        if not _is_nonnegative_integer_like(matrix):
            raise ValueError(
                f"adata.layers[{counts_layer!r}] is not non-negative integer-like"
            )
        return matrix, adata.var.copy(), adata.var_names, (
            f"layers[{counts_layer!r}]"
        ), "counts"

    if counts_layer == "auto":
        for layer in COUNT_LAYER_CANDIDATES:
            if layer in adata.layers and _is_nonnegative_integer_like(
                adata.layers[layer]
            ):
                return (
                    adata.layers[layer],
                    adata.var.copy(),
                    adata.var_names,
                    f"layers[{layer!r}]",
                    "counts",
                )
        if getattr(adata, "raw", None) is not None and _is_nonnegative_integer_like(
            adata.raw.X
        ):
            return (
                adata.raw.X,
                adata.raw.var.copy(),
                adata.raw.var_names,
                "raw.X",
                "counts",
            )
        if _is_nonnegative_integer_like(adata.X):
            return adata.X, adata.var.copy(), adata.var_names, "X", "counts"

    return adata.X, adata.var.copy(), adata.var_names, "X", "normalised"


def infer_annotation_fingerprint(
    features: Sequence[Any],
    *,
    feature_kind: str,
    source_hg19: pd.DataFrame,
    target_hg38: pd.DataFrame,
    minimum_informative: int = 20,
    strong_fraction: float = 0.90,
    probable_fraction: float = 0.75,
) -> dict[str, Any]:
    """
    Infer an annotation fingerprint, not a definitive genome assembly.
    """
    values = pd.Index(pd.Series(features, dtype="string").dropna().astype(str).unique())
    if feature_kind == "ensembl_id":
        base = set(map(_strip_ensembl_version, values))
        old_set = set(source_hg19["gene_id"])
        new_set = set(target_hg38["gene_id"])
        old_only = len(base & (old_set - new_set))
        new_only = len(base & (new_set - old_set))
        both = len(base & old_set & new_set)
        neither = len(base - old_set - new_set)

        versioned = {x for x in values if re.search(r"\.\d+$", x)}
        old_full = set(source_hg19["gene_id_full"])
        new_full = set(target_hg38["gene_id_full"])
        exact_old = len(versioned & old_full)
        exact_new = len(versioned & new_full)
        evidence_type = "ensembl_id"
        confidence_cap = "high"
    else:
        base = set(values)
        old_set = set(source_hg19["gene_symbol"].astype(str)) - {""}
        new_set = set(target_hg38["gene_symbol"].astype(str)) - {""}
        old_only = len(base & (old_set - new_set))
        new_only = len(base & (new_set - old_set))
        both = len(base & old_set & new_set)
        neither = len(base - old_set - new_set)
        exact_old = exact_new = 0
        evidence_type = "gene_symbol"
        confidence_cap = "low"

    informative = old_only + new_only
    if informative < minimum_informative:
        label, confidence = "ambiguous", "low"
    else:
        old_fraction = old_only / informative
        new_fraction = new_only / informative
        if old_fraction >= strong_fraction:
            label, confidence = "v19-like", confidence_cap
        elif new_fraction >= strong_fraction:
            label, confidence = "v49-like", confidence_cap
        elif old_fraction >= probable_fraction:
            label, confidence = "probably-v19-like", "medium"
        elif new_fraction >= probable_fraction:
            label, confidence = "probably-v49-like", "medium"
        else:
            label, confidence = "ambiguous", "low"

    return {
        "label": label,
        "confidence": confidence,
        "evidence_type": evidence_type,
        "n_features": len(values),
        "n_v19_only": old_only,
        "n_v49_only": new_only,
        "n_both": both,
        "n_neither": neither,
        "n_versioned_exact_v19": exact_old,
        "n_versioned_exact_v49": exact_new,
    }


def _normalise_assembly(value: str) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"hg19", "grch37", "37", "b37"}:
        return "GRCh37"
    if text in {"hg38", "grch38", "38"}:
        return "GRCh38"
    if text in {"auto", "unknown", "", "none"}:
        return "auto"
    raise ValueError(
        "source_assembly must be hg19/GRCh37, hg38/GRCh38, or auto"
    )


def _load_history_map(path: Optional[PathLike]) -> dict[str, str]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    source_candidates = ("source_gene_id", "old_gene_id", "from", "source")
    target_candidates = ("target_gene_id", "new_gene_id", "to", "target")
    source_col = next((x for x in source_candidates if x in table.columns), None)
    target_col = next((x for x in target_candidates if x in table.columns), None)
    if source_col is None or target_col is None:
        raise ValueError(
            "history_map_csv needs source_gene_id and target_gene_id columns"
        )
    result: dict[str, str] = {}
    for source, target in zip(table[source_col], table[target_col]):
        source = _strip_ensembl_version(source)
        target = _strip_ensembl_version(target)
        if ENSG_RE.match(source) and ENSG_RE.match(target):
            result[source] = target
    return result


def _reciprocal_span_overlap(a: pd.Series, b: pd.Series) -> float:
    if (
        a["chrom"] != b["chrom"]
        or a["strand"] != b["strand"]
        or a["chrom"] not in STANDARD_CHROMS
    ):
        return 0.0
    overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])) + 1)
    if overlap <= 0:
        return 0.0
    len_a = int(a["end"]) - int(a["start"]) + 1
    len_b = int(b["end"]) - int(b["start"]) + 1
    return min(overlap / len_a, overlap / len_b)


def _coordinate_candidate(
    source_record: pd.Series,
    lift37: pd.DataFrame,
    *,
    minimum_reciprocal_overlap: float = 0.80,
) -> tuple[Optional[str], float, int]:
    candidates = lift37.loc[
        (lift37["chrom"] == source_record["chrom"])
        & (lift37["strand"] == source_record["strand"])
        & (lift37["start"] <= int(source_record["end"]))
        & (lift37["end"] >= int(source_record["start"]))
    ]
    if candidates.empty:
        return None, 0.0, 0

    scores: dict[str, float] = {}
    for _, candidate in candidates.iterrows():
        score = _reciprocal_span_overlap(source_record, candidate)
        gene_id = str(candidate["gene_id"])
        scores[gene_id] = max(scores.get(gene_id, 0.0), score)
    passing = {gene_id: score for gene_id, score in scores.items() if score >= minimum_reciprocal_overlap}
    if len(passing) != 1:
        best = max(scores.values(), default=0.0)
        return None, best, len(passing)
    gene_id, score = next(iter(passing.items()))
    return gene_id, score, 1


def _validate_symbol_coordinate(
    source_record: Optional[pd.Series],
    target_gene_id: str,
    lift_by_id: Mapping[str, pd.Series],
) -> tuple[bool, Optional[float]]:
    if source_record is None or target_gene_id not in lift_by_id:
        return True, None
    score = _reciprocal_span_overlap(source_record, lift_by_id[target_gene_id])
    return score > 0.0, score


def build_gene_mapping(
    features: Sequence[Any],
    companion_symbols: Sequence[Any],
    *,
    feature_kind: str,
    source_assembly: str,
    source_hg19: pd.DataFrame,
    target_hg38: pd.DataFrame,
    target_lift37: Optional[pd.DataFrame] = None,
    history_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Build an auditable source-feature to target-v49-ENSG mapping."""
    history_map = dict(history_map or {})
    target_ids = set(target_hg38["gene_id"])
    target_by_id = _first_gene_record_by_id(target_hg38)
    target_symbol_to_id = _unique_value_map(
        target_hg38, "gene_symbol", "gene_id"
    )

    source_by_id = _first_gene_record_by_id(source_hg19)
    source_symbol_to_id = _unique_value_map(
        source_hg19, "gene_symbol", "gene_id"
    )
    source_symbols = set(
        source_hg19.loc[
            source_hg19["gene_symbol"].astype(str).ne(""), "gene_symbol"
        ].astype(str)
    )

    lift_by_id: dict[str, pd.Series] = {}
    usable_lift: Optional[pd.DataFrame] = None
    if target_lift37 is not None:
        usable_lift = target_lift37.loc[
            target_lift37["gene_id"].isin(target_ids)
        ].copy()
        lift_by_id = _first_gene_record_by_id(usable_lift)

    rows: list[dict[str, Any]] = []
    for position, (feature, companion_symbol) in enumerate(
        zip(features, companion_symbols)
    ):
        original_feature = str(feature).strip()
        supplied_symbol = str(companion_symbol).strip()
        if supplied_symbol.lower() in {"nan", "none", "<na>"}:
            supplied_symbol = ""

        source_gene_id = ""
        source_symbol = supplied_symbol
        source_record: Optional[pd.Series] = None
        target_gene_id = ""
        method = "unmapped"
        confidence = "none"
        note = ""
        coordinate_score: Optional[float] = None

        if feature_kind == "ensembl_id":
            source_gene_id = _strip_ensembl_version(original_feature)
            if source_assembly == "GRCh37":
                source_record = source_by_id.get(source_gene_id)
                if source_record is not None and not source_symbol:
                    source_symbol = str(source_record["gene_symbol"])
            elif not source_symbol and source_gene_id in target_by_id:
                source_symbol = str(target_by_id[source_gene_id]["gene_symbol"])
        else:
            source_symbol = original_feature
            if source_assembly == "GRCh37":
                source_gene_id = source_symbol_to_id.get(source_symbol, "")
                source_record = source_by_id.get(source_gene_id)
                if source_symbol in source_symbols and not source_gene_id:
                    note = "symbol is not unique in the GRCh37 source annotation"

        # 1. Stable ID unchanged between releases.
        if source_gene_id and source_gene_id in target_ids:
            target_gene_id = source_gene_id
            method = "stable_id_exact"
            confidence = "high"

        # 2. Explicit reviewed/history crosswalk.
        if not target_gene_id and source_gene_id in history_map:
            candidate = history_map[source_gene_id]
            if candidate in target_ids:
                target_gene_id = candidate
                method = "stable_id_replacement"
                confidence = "high"
            else:
                note = "history replacement absent from target annotation"

        # 3. Symbol is unique in the target; validate location for GRCh37.
        source_symbol_is_usable = not (
            feature_kind == "gene_symbol"
            and source_assembly == "GRCh37"
            and not source_gene_id
        )
        if not target_gene_id and source_symbol and source_symbol_is_usable:
            candidate = target_symbol_to_id.get(source_symbol)
            if candidate is not None:
                valid, score = _validate_symbol_coordinate(
                    source_record, candidate, lift_by_id
                )
                coordinate_score = score
                if valid:
                    target_gene_id = candidate
                    method = (
                        "unique_symbol_coordinate_validated"
                        if score is not None
                        else "unique_symbol"
                    )
                    confidence = "high" if score is not None else "medium"
                else:
                    note = "unique symbol failed GRCh37 coordinate validation"

        # 4. Conservative coordinate rescue on the shared GRCh37 assembly.
        if (
            not target_gene_id
            and source_record is not None
            and usable_lift is not None
        ):
            candidate, score, n_candidates = _coordinate_candidate(
                source_record, usable_lift
            )
            coordinate_score = score
            if candidate is not None:
                target_gene_id = candidate
                method = "coordinate_unique_overlap"
                confidence = "medium"
            elif n_candidates > 1:
                method = "one_to_many_ambiguous"
                confidence = "none"
                note = f"{n_candidates} coordinate candidates passed threshold"

        if target_gene_id:
            target_record = target_by_id[target_gene_id]
            target_symbol = str(target_record["gene_symbol"])
            target_biotype = str(target_record["biotype"])
            mapping_status = "mapped"
        else:
            target_symbol = ""
            target_biotype = ""
            mapping_status = (
                "ambiguous" if method == "one_to_many_ambiguous" else "unmapped"
            )

        rows.append(
            {
                "source_position": position,
                "original_feature": original_feature,
                "source_gene_id": source_gene_id,
                "source_gene_symbol": source_symbol,
                "source_assembly": source_assembly,
                "target_gene_id": target_gene_id,
                "target_gene_symbol": target_symbol,
                "target_biotype": target_biotype,
                "mapping_method": method,
                "mapping_confidence": confidence,
                "mapping_status": mapping_status,
                "coordinate_overlap_score": (
                    np.nan if coordinate_score is None else coordinate_score
                ),
                "mapping_notes": note,
            }
        )

    return pd.DataFrame.from_records(rows)


def _mark_mapping_cardinality(
    mapping: pd.DataFrame,
    *,
    expression_kind: str,
) -> pd.DataFrame:
    mapping = mapping.copy()
    mapping["mapping_cardinality"] = ""
    mapped = mapping["mapping_status"].eq("mapped")
    counts = mapping.loc[mapped, "target_gene_id"].value_counts()
    duplicate_targets = set(counts[counts > 1].index)

    mapping.loc[mapped, "mapping_cardinality"] = "1:1"
    mapping.loc[
        mapped & mapping["target_gene_id"].isin(duplicate_targets),
        "mapping_cardinality",
    ] = "N:1"

    if duplicate_targets and expression_kind != "counts":
        affected = mapped & mapping["target_gene_id"].isin(duplicate_targets)
        mapping.loc[affected, "mapping_status"] = (
            "many_to_one_dropped_noncounts"
        )
        mapping.loc[affected, "mapping_notes"] = (
            "multiple source features map to one target; cannot sum log-normalised values"
        )
    return mapping


def _aggregate_to_target(
    matrix: Any,
    mapping: pd.DataFrame,
    target_hg38: pd.DataFrame,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    accepted = mapping["mapping_status"].eq("mapped")
    accepted_rows = mapping.loc[accepted].copy()
    if accepted_rows.empty:
        raise ValueError("No genes could be mapped to the target annotation")

    # Deterministic order from target annotation, independent of source dataset.
    target_order = (
        target_hg38.drop_duplicates("gene_id")
        .reset_index(drop=True)
        .assign(_target_order=lambda x: np.arange(len(x)))
        .set_index("gene_id")["_target_order"]
        .to_dict()
    )
    target_ids = sorted(
        pd.unique(accepted_rows["target_gene_id"]),
        key=lambda x: target_order.get(x, 10**12),
    )
    target_col = {gene_id: i for i, gene_id in enumerate(target_ids)}

    source_indices = accepted_rows["source_position"].astype(int).to_numpy()
    target_indices = (
        accepted_rows["target_gene_id"].map(target_col).astype(int).to_numpy()
    )
    projection = sparse.csr_matrix(
        (
            np.ones(len(source_indices), dtype=np.float32),
            (source_indices, target_indices),
        ),
        shape=(matrix.shape[1], len(target_ids)),
    )
    source_matrix = (
        matrix.tocsr()
        if sparse.issparse(matrix)
        else sparse.csr_matrix(np.asarray(matrix))
    )
    aggregated = (source_matrix @ projection).tocsr()

    target_records = (
        target_hg38.drop_duplicates("gene_id")
        .set_index("gene_id")
        .loc[target_ids]
    )
    var = pd.DataFrame(index=pd.Index(target_ids, name=None))
    var["gene_symbol"] = target_records["gene_symbol"].astype(str).to_numpy()
    var["gene_biotype"] = target_records["biotype"].astype(str).to_numpy()
    var["chrom"] = target_records["chrom"].astype(str).to_numpy()
    var["start"] = target_records["start"].astype(int).to_numpy()
    var["end"] = target_records["end"].astype(int).to_numpy()
    var["strand"] = target_records["strand"].astype(str).to_numpy()
    source_per_target = accepted_rows["target_gene_id"].value_counts()
    var["n_source_features_aggregated"] = [
        int(source_per_target[x]) for x in target_ids
    ]
    return aggregated, var


def _normalise_log1p(counts: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    counts = counts.astype(np.float32, copy=True)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        n_zero = int(np.sum(totals <= 0))
        raise ValueError(
            f"{n_zero} cells have zero retained counts after gene mapping"
        )
    factors = target_sum / totals
    normalised = sparse.diags(factors.astype(np.float32)) @ counts
    normalised = normalised.tocsr()
    normalised.data = np.log1p(normalised.data)
    return normalised


def _expm1_matrix(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        linear = matrix.tocsr().astype(np.float32, copy=True)
        linear.data = np.expm1(linear.data)
        return linear
    return sparse.csr_matrix(np.expm1(np.asarray(matrix, dtype=np.float32)))


def _validate_celltypist_expression(matrix: Any, target_sum: float) -> None:
    n_rows = min(matrix.shape[0], 1000)
    sample = matrix[:n_rows]
    if sparse.issparse(sample):
        sample = sample.tocsr(copy=True)
        if sample.data.size and (
            np.min(sample.data) < 0 or np.max(sample.data) > np.log1p(target_sum) + 1
        ):
            raise ValueError(
                "X does not look like non-negative log1p-normalised expression"
            )
        restored = sample.copy()
        restored.data = np.expm1(restored.data)
        totals = np.asarray(restored.sum(axis=1)).ravel()
    else:
        array = np.asarray(sample)
        if np.min(array) < 0 or np.max(array) > np.log1p(target_sum) + 1:
            raise ValueError(
                "X does not look like non-negative log1p-normalised expression"
            )
        totals = np.expm1(array).sum(axis=1)
    positive = totals > 0
    if not np.any(positive) or not np.allclose(
        totals[positive], target_sum, rtol=0.05, atol=50
    ):
        raise ValueError(
            "X is not log1p-normalised to approximately "
            f"{target_sum:g} counts per cell"
        )


def _copy_cell_metadata(adata: Any, out: Any) -> None:
    out.uns = copy.deepcopy(dict(adata.uns))
    for key, value in adata.obsm.items():
        out.obsm[key] = value.copy() if hasattr(value, "copy") else copy.deepcopy(value)
    for key, value in adata.obsp.items():
        out.obsp[key] = value.copy() if hasattr(value, "copy") else copy.deepcopy(value)


def _build_celltypist_mapping(target_hg38: pd.DataFrame) -> pd.DataFrame:
    pairs = target_hg38.loc[
        target_hg38["gene_symbol"].astype(str).ne(""),
        ["gene_symbol", "gene_id"],
    ].drop_duplicates()
    symbol_counts = pairs["gene_symbol"].value_counts()
    id_counts = pairs["gene_id"].value_counts()
    pairs = pairs.loc[
        pairs["gene_symbol"].map(symbol_counts).eq(1)
        & pairs["gene_id"].map(id_counts).eq(1)
    ]
    return pairs.reset_index(drop=True)


def _prepare_celltypist_model(
    celltypist_model: Any,
    target_hg38: pd.DataFrame,
    *,
    converted_model_path: Optional[PathLike],
) -> tuple[Any, dict[str, Any]]:
    import celltypist

    model_class = celltypist.models.Model
    cache_path: Optional[Path] = None
    if converted_model_path is not None:
        cache_path = Path(converted_model_path)
        if cache_path.suffix.lower() != ".pkl":
            cache_path = cache_path.with_suffix(".pkl")

    if cache_path is not None and cache_path.exists():
        model = model_class.load(str(cache_path))
        return model, {
            "model_source": str(cache_path),
            "model_conversion": "loaded_existing_converted_model",
            "n_model_features": len(model.features),
        }

    if hasattr(celltypist_model, "features") and hasattr(
        celltypist_model, "classifier"
    ):
        model = copy.deepcopy(celltypist_model)
        model_source = type(celltypist_model).__name__
    else:
        model = model_class.load(str(celltypist_model))
        model_source = str(celltypist_model)

    original_n = len(model.features)
    ensembl_fraction = _fraction_matching_ensg(model.features)
    if ensembl_fraction < 0.60:
        mapping = _build_celltypist_mapping(target_hg38)
        with tempfile.TemporaryDirectory() as temporary_directory:
            map_path = Path(temporary_directory) / "v49_symbol_to_ensg.csv"
            mapping.to_csv(map_path, index=False, header=False)
            model.convert(
                str(map_path),
                convert_from=0,
                convert_to=1,
                unique_only=True,
            )
        conversion = "symbol_to_target_ensg_unique_only"
    else:
        conversion = "model_already_uses_ensembl_ids"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(cache_path))

    return model, {
        "model_source": model_source,
        "model_conversion": conversion,
        "n_original_model_features": original_n,
        "n_model_features": len(model.features),
    }


def _insert_celltypist_results(
    out: Any,
    predictions: Any,
    *,
    majority_voting: bool,
) -> str:
    label_table = predictions.predicted_labels.copy().reindex(out.obs_names)
    if "predicted_labels" not in label_table.columns:
        raise RuntimeError("CellTypist result lacks predicted_labels")

    out.obs["celltypist_predicted_labels"] = label_table[
        "predicted_labels"
    ].astype(str)
    chosen_column = "predicted_labels"
    if majority_voting and "majority_voting" in label_table.columns:
        out.obs["celltypist_majority_voting"] = label_table[
            "majority_voting"
        ].astype(str)
        chosen_column = "majority_voting"

    out.obs["predicted_labels"] = label_table[chosen_column].astype(str)

    probability = predictions.probability_matrix.copy().reindex(out.obs_names)
    chosen_labels = out.obs["predicted_labels"].astype(str)
    confidence = np.full(out.n_obs, np.nan, dtype=float)
    for i, label in enumerate(chosen_labels):
        if label in probability.columns:
            confidence[i] = float(probability.iloc[i][label])
        elif probability.shape[1]:
            confidence[i] = float(probability.iloc[i].max())
    out.obs["conf_score"] = confidence
    return chosen_column


def harmonise_and_annotate(
    adata: Any,
    *,
    source_assembly: str,
    hg19_gtf: PathLike,
    target_hg38_gff: PathLike,
    celltypist_model: Any = "Immune_All_Low.pkl",
    target_v49_lift37_gtf: Optional[PathLike] = None,
    gene_id_col: Optional[str] = None,
    gene_symbol_col: Optional[str] = None,
    counts_layer: Optional[str] = "auto",
    history_map_csv: Optional[PathLike] = None,
    converted_model_path: Optional[PathLike] = None,
    audit_csv: Optional[PathLike] = None,
    standard_chromosomes_only: bool = True,
    strict_assembly_check: bool = False,
    normalise_target_sum: float = 10_000.0,
    check_expression: bool = True,
    majority_voting: bool = True,
    over_clustering: Optional[Any] = None,
    mode: str = "best match",
    p_thres: float = 0.5,
    run_celltypist: bool = True,
) -> Any:
    """
    Harmonise one AnnData to GENCODE v49 ENSG and run converted CellTypist.

    Parameters
    ----------
    adata
        Input AnnData. Cells must be rows and features must be columns.
    source_assembly
        ``"hg19"``/``"GRCh37"``, ``"hg38"``/``"GRCh38"``, or ``"auto"``.
        GEO/author metadata should normally be supplied here. Gene-list
        fingerprinting is used as a cross-check, not as unquestioned truth.
    hg19_gtf
        GENCODE v19 GRCh37 annotation, e.g.
        ``gencode.v19.chr_patch_hapl_scaff.annotation.gtf``.
    target_hg38_gff
        Ensembl release 115 / GENCODE v49 GRCh38 annotation, e.g.
        ``Homo_sapiens.GRCh38.115.gff3``.
    celltypist_model
        CellTypist model name, path, or loaded Model.
    target_v49_lift37_gtf
        Optional but recommended ``gencode.v49lift37.annotation.gtf.gz``.
        It provides conservative coordinate validation/rescue for GRCh37.
    gene_id_col, gene_symbol_col
        Optional explicit input ``adata.var`` columns. If omitted, common
        column names and then ``var_names`` are examined automatically.
    counts_layer
        Count layer name, ``"auto"`` (default), or ``None``. Auto checks common
        count layers, ``raw.X``, then integer-like ``X``. If no count matrix is
        found, ``X`` is treated as already log1p-normalised.
    history_map_csv
        Optional reviewed two-column crosswalk with ``source_gene_id`` and
        ``target_gene_id``.
    converted_model_path
        Optional path used to cache/reuse the converted ENSG CellTypist model.
    audit_csv
        Optional path for the complete feature-level mapping audit.
    standard_chromosomes_only
        Restrict the target vocabulary to chromosomes 1-22, X, Y and MT.
    strict_assembly_check
        Raise if supplied assembly conflicts with a strong ID fingerprint.
    run_celltypist
        Set False to test gene mapping without running CellTypist.

    Returns
    -------
    anndata.AnnData
        New AnnData with target ENSG ``var_names`` and CellTypist results in
        ``obs``. The input object is not modified.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "anndata is required: pip install anndata"
        ) from exc

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData must contain cells and genes")

    source_ref = read_gene_annotation(
        hg19_gtf, standard_chromosomes_only=False
    )
    target_ref = read_gene_annotation(
        target_hg38_gff,
        standard_chromosomes_only=standard_chromosomes_only,
    )
    lift_ref = (
        read_gene_annotation(
            target_v49_lift37_gtf,
            standard_chromosomes_only=True,
        )
        if target_v49_lift37_gtf is not None
        else None
    )

    matrix, feature_var, feature_names, matrix_source, expression_kind = (
        _select_expression(adata, counts_layer)
    )
    if matrix.shape[0] != adata.n_obs:
        raise ValueError("Selected expression matrix does not match adata.n_obs")
    if matrix.shape[1] != len(feature_var):
        raise ValueError("Selected expression matrix does not match feature metadata")

    feature_kind, features, companion_symbols, feature_source = (
        _detect_feature_namespace(
            feature_var,
            feature_names,
            gene_id_col=gene_id_col,
            gene_symbol_col=gene_symbol_col,
        )
    )
    fingerprint = infer_annotation_fingerprint(
        features,
        feature_kind=feature_kind,
        source_hg19=source_ref,
        target_hg38=target_ref,
    )

    resolved_assembly = _normalise_assembly(source_assembly)
    if resolved_assembly == "auto":
        if fingerprint["label"] in {"v19-like", "probably-v19-like"}:
            resolved_assembly = "GRCh37"
        elif fingerprint["label"] in {"v49-like", "probably-v49-like"}:
            resolved_assembly = "GRCh38"
        else:
            raise ValueError(
                "Assembly cannot be resolved from the gene list. Pass "
                "source_assembly='hg19' or source_assembly='hg38' using "
                "GEO/author provenance."
            )
    else:
        conflict = (
            resolved_assembly == "GRCh37"
            and fingerprint["label"] == "v49-like"
        ) or (
            resolved_assembly == "GRCh38"
            and fingerprint["label"] == "v19-like"
        )
        if conflict:
            message = (
                f"Reported assembly {resolved_assembly} conflicts with "
                f"annotation fingerprint {fingerprint['label']}. This can be "
                "legitimate for lift37 annotations; review provenance."
            )
            if strict_assembly_check:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)

    history_map = _load_history_map(history_map_csv)
    mapping = build_gene_mapping(
        features,
        companion_symbols,
        feature_kind=feature_kind,
        source_assembly=resolved_assembly,
        source_hg19=source_ref,
        target_hg38=target_ref,
        target_lift37=lift_ref,
        history_map=history_map,
    )
    mapping = _mark_mapping_cardinality(
        mapping, expression_kind=expression_kind
    )

    if expression_kind == "counts":
        matrix_for_mapping = matrix
    else:
        if check_expression:
            _validate_celltypist_expression(matrix, normalise_target_sum)
        matrix_for_mapping = _expm1_matrix(matrix)

    harmonised_linear, target_var = _aggregate_to_target(
        matrix_for_mapping, mapping, target_ref
    )
    output_x = _normalise_log1p(harmonised_linear, normalise_target_sum)
    counts = harmonised_linear if expression_kind == "counts" else None

    out = ad.AnnData(
        X=output_x,
        obs=adata.obs.copy(),
        var=target_var,
    )
    _copy_cell_metadata(adata, out)
    if counts is not None:
        out.layers["counts"] = counts

    mapped_mask = mapping["mapping_status"].eq("mapped")
    mapping_counts = {
        str(key): int(value)
        for key, value in mapping["mapping_status"].value_counts().items()
    }
    method_counts = {
        str(key): int(value)
        for key, value in mapping.loc[
            mapped_mask, "mapping_method"
        ].value_counts().items()
    }
    input_total = float(matrix_for_mapping.sum())
    retained_total = float(harmonised_linear.sum())
    retained_fraction = (
        retained_total / input_total if input_total > 0 else np.nan
    )

    out.uns["gene_harmonisation"] = {
        "target_annotation": "GENCODE_v49_Ensembl_115_GRCh38",
        "target_annotation_file": str(target_hg38_gff),
        "source_annotation_file": str(hg19_gtf),
        "lift37_annotation_file": (
            "" if target_v49_lift37_gtf is None else str(target_v49_lift37_gtf)
        ),
        "reported_source_assembly": str(source_assembly),
        "resolved_source_assembly": resolved_assembly,
        "feature_namespace": feature_kind,
        "feature_source": feature_source,
        "matrix_source": matrix_source,
        "expression_kind": expression_kind,
        "n_input_features": int(matrix.shape[1]),
        "n_output_genes": int(out.n_vars),
        "retained_matrix_sum_fraction": float(retained_fraction),
        "mapping_status_counts": mapping_counts,
        "mapping_method_counts": method_counts,
        "annotation_fingerprint": {
            str(k): (int(v) if isinstance(v, np.integer) else v)
            for k, v in fingerprint.items()
        },
        "mapping_table": mapping.fillna("").astype(str),
    }

    if audit_csv is not None:
        audit_path = Path(audit_csv)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(audit_path, index=False)

    if run_celltypist:
        try:
            import celltypist
        except ImportError as exc:
            raise ImportError(
                "celltypist is required: pip install celltypist"
            ) from exc

        ct_model, model_info = _prepare_celltypist_model(
            celltypist_model,
            target_ref,
            converted_model_path=converted_model_path,
        )
        overlap = len(set(map(str, out.var_names)) & set(map(str, ct_model.features)))
        model_info["n_query_model_feature_overlap"] = int(overlap)
        model_info["query_model_feature_overlap_fraction"] = float(
            overlap / len(ct_model.features) if len(ct_model.features) else 0.0
        )
        if overlap == 0:
            raise ValueError(
                "No harmonised query genes overlap the converted CellTypist model"
            )
        if overlap < 100:
            warnings.warn(
                f"Only {overlap} query genes overlap the CellTypist model; "
                "inspect the mapping audit before trusting labels.",
                RuntimeWarning,
            )

        predictions = celltypist.annotate(
            out,
            model=ct_model,
            majority_voting=majority_voting,
            over_clustering=over_clustering,
            mode=mode,
            p_thres=p_thres,
        )
        chosen_column = _insert_celltypist_results(
            out,
            predictions,
            majority_voting=majority_voting,
        )
        model_info["label_column_used"] = chosen_column
        model_info["majority_voting"] = bool(majority_voting)
        model_info["mode"] = str(mode)
        out.uns["celltypist"] = model_info

    return out


__all__ = [
    "harmonise_and_annotate",
]
