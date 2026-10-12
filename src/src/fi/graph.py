"""Build a biological Functional Interaction graph and export it as JSON.

This step keeps one FI record per source row. Undirected and bidirectional
relations are not split into PyG message-passing edges.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.fi.parse import DIRECTION_TABLE, SEMANTIC_ATTRIBUTES, parse_annotation, parse_confidence, parse_direction


def exact_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    """Identity of a CSV row. Gene order is part of the identity."""
    return (
        row["gene1"],
        row["gene2"],
        row["annotation"],
        row["direction"],
        row["confidence"].strip(),
    )


def drop_exact_duplicates(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Keep the first copy of each identical row. Conflicting rows stay separate."""
    seen: set[tuple[str, str, str, str, str]] = set()
    kept: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        key = exact_row_key(row)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, removed


def build_node_mapping(rows: list[dict[str, str]]) -> tuple[list[dict], dict[str, int]]:
    """One node per gene symbol present in Gene1 or Gene2, sorted for stable ids."""
    genes = sorted({row["gene1"] for row in rows} | {row["gene2"] for row in rows})
    nodes = [{"id": index, "gene": gene} for index, gene in enumerate(genes)]
    return nodes, {node["gene"]: node["id"] for node in nodes}


def build_fi_records(rows: list[dict[str, str]], gene_to_id: dict[str, int]) -> list[dict]:
    """Turn deduplicated FI rows into biological edge records."""
    records = []
    for row in rows:
        annotation = parse_annotation(row["annotation"])
        direction = parse_direction(row["direction"], row["gene1"], row["gene2"])
        records.append(
            {
                "gene1": row["gene1"],
                "gene2": row["gene2"],
                "node1_id": gene_to_id[row["gene1"]],
                "node2_id": gene_to_id[row["gene2"]],
                "direction_raw": direction["direction_raw"],
                "direction_type": direction["direction_type"],
                "source": direction["source"],
                "target": direction["target"],
                "annotation_raw": annotation["annotation_raw"],
                "annotation_tokens_raw": annotation["annotation_tokens_raw"],
                "unknown_annotation_tokens": annotation["unknown_annotation_tokens"],
                "semantic_attributes": annotation["semantic_attributes"],
                "edge_type_vector": annotation["edge_type_vector"],
                "confidence": parse_confidence(row["confidence"]),
            }
        )
    return records


def pair_statistics(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count unordered gene pairs that repeat after exact duplicates are gone."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(sorted((row["gene1"], row["gene2"])))].append(row)
    repeated = 0
    multiple_annotations = 0
    conflicting_directions = 0
    either = 0
    for group in grouped.values():
        if len(group) < 2:
            continue
        repeated += 1
        annotations = {item["annotation"] for item in group}
        directions = {item["direction"].strip() for item in group}
        multi = len(annotations) > 1
        conflict = len(directions) > 1
        multiple_annotations += int(multi)
        conflicting_directions += int(conflict)
        either += int(multi or conflict)
    return {
        "repeated_unordered_gene_pairs": repeated,
        "pairs_with_multiple_annotation_sets": multiple_annotations,
        "pairs_with_conflicting_directions": conflicting_directions,
        "conflicting_pair_count": either,
    }


def generate_summary(
    n_input_rows: int,
    n_exact_duplicates_removed: int,
    n_nodes: int,
    records: list[dict],
    pair_stats: dict[str, int],
) -> dict:
    """Validation counts for the exported biological graph."""
    direction_raw = {token: 0 for token in (*DIRECTION_TABLE, "unknown")}
    direction_type = {name: 0 for name in ("directed", "undirected", "bidirectional", "unknown")}
    semantic = {name: 0 for name in SEMANTIC_ATTRIBUTES}
    unknown_annotations: Counter[str] = Counter()
    unknown_directions: Counter[str] = Counter()
    confidences: list[float] = []
    missing_confidence = 0
    for record in records:
        raw_direction = record["direction_raw"].strip()
        if raw_direction in DIRECTION_TABLE:
            direction_raw[raw_direction] += 1
        else:
            direction_raw["unknown"] += 1
            unknown_directions[raw_direction] += 1
        direction_type[record["direction_type"]] += 1
        for name in SEMANTIC_ATTRIBUTES:
            semantic[name] += int(record["semantic_attributes"][name])
        unknown_annotations.update(record["unknown_annotation_tokens"])
        if record["confidence"] is None:
            missing_confidence += 1
        else:
            confidences.append(record["confidence"])
    if confidences:
        confidence_stats = {
            "count": len(confidences),
            "missing": missing_confidence,
            "min": min(confidences),
            "max": max(confidences),
            "mean": statistics.fmean(confidences),
            "median": statistics.median(confidences),
        }
    else:
        confidence_stats = {
            "count": 0,
            "missing": missing_confidence,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "num_input_rows": n_input_rows,
        "num_rows_after_exact_dedup": n_input_rows - n_exact_duplicates_removed,
        "num_unique_genes": n_nodes,
        "num_fi_records": len(records),
        "direction_raw_frequencies": direction_raw,
        "direction_type_frequencies": direction_type,
        "semantic_attribute_frequencies": semantic,
        "confidence": confidence_stats,
        "unknown_annotation_tokens": dict(sorted(unknown_annotations.items())),
        "unknown_direction_tokens": dict(sorted(unknown_directions.items())),
        "exact_duplicate_count": n_exact_duplicates_removed,
        "repeated_unordered_gene_pairs": pair_stats["repeated_unordered_gene_pairs"],
        "pairs_with_multiple_annotation_sets": pair_stats["pairs_with_multiple_annotation_sets"],
        "pairs_with_conflicting_directions": pair_stats["pairs_with_conflicting_directions"],
        "conflicting_pair_count": pair_stats["conflicting_pair_count"],
    }


def export_fi_graph_json(path: Path, nodes: list[dict], edges: list[dict]) -> None:
    """Write the biological FI graph. Null source/target stay JSON null."""
    payload = {
        "metadata": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "semantic_vector_order": list(SEMANTIC_ATTRIBUTES),
        },
        "nodes": nodes,
        "edges": edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def export_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
