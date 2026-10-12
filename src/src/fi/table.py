"""Read an FI table from CSV, TSV, or a zip archive containing one of those."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

CONFIDENCE_COLUMNS = ("Confidence", "Score", "confidence", "score")


def load_fi_rows(path: Path) -> list[dict[str, str]]:
    """Return rows with gene1, gene2, annotation, direction, and confidence text.

    Reactome exports the evidence column as ``Score``. A ``Confidence`` column
    is accepted as the same field.
    """
    text = _read_text(Path(path))
    sample = text.splitlines()[0] if text else ""
    delimiter = "\t" if "\t" in sample else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError(f"FI table has no header: {path}")
    fields = {name.strip(): name for name in reader.fieldnames if name}
    confidence_key = next((fields[name] for name in CONFIDENCE_COLUMNS if name in fields), None)
    missing = [name for name in ("Gene1", "Gene2", "Annotation", "Direction") if name not in fields]
    if missing or confidence_key is None:
        raise ValueError(
            f"FI table must have Gene1, Gene2, Annotation, Direction, and Confidence or Score; got {reader.fieldnames}"
        )
    rows = []
    for record in reader:
        rows.append(
            {
                "gene1": (record.get(fields["Gene1"]) or "").strip(),
                "gene2": (record.get(fields["Gene2"]) or "").strip(),
                "annotation": record.get(fields["Annotation"]) or "",
                "direction": record.get(fields["Direction"]) or "",
                "confidence": record.get(confidence_key) or "",
            }
        )
    return rows


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".txt", ".csv", ".tsv"))
            ]
            if len(names) != 1:
                raise ValueError(f"expected one table inside {path}, found {names}")
            return archive.read(names[0]).decode("utf-8-sig")
    return path.read_text(encoding="utf-8-sig")
