"""Build the biological FI graph JSON from the Reactome FI table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fi.graph import (
    build_fi_records,
    build_node_mapping,
    drop_exact_duplicates,
    export_fi_graph_json,
    export_summary_json,
    generate_summary,
    pair_statistics,
)
from src.fi.table import load_fi_rows

DEFAULT_INPUT = Path(
    "/media/rokny/DATA3/Jade/HTG_Model_ICI_Prediction/data/functional_interaction/FIsInGene_04142025_with_annotations.txt.zip"
)


def build_fi_graph(input_path: Path, output_dir: Path) -> dict:
    rows = [row for row in load_fi_rows(input_path) if row["gene1"] and row["gene2"]]
    kept, removed = drop_exact_duplicates(rows)
    nodes, gene_to_id = build_node_mapping(kept)
    edges = build_fi_records(kept, gene_to_id)
    summary = generate_summary(len(rows), removed, len(nodes), edges, pair_statistics(kept))
    export_fi_graph_json(output_dir / "fi_graph.json", nodes, edges)
    export_summary_json(output_dir / "fi_graph_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INPUT.parent,
    )
    args = parser.parse_args()
    summary = build_fi_graph(args.input, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"graph={args.output_dir / 'fi_graph.json'}")
    print(f"summary={args.output_dir / 'fi_graph_summary.json'}")


if __name__ == "__main__":
    main()
