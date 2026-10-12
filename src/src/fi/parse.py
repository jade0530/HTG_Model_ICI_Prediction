"""Parse Functional Interaction annotation, direction, and confidence fields.

Annotation carries semantic meaning. Direction carries orientation. They are
read from separate columns and are not inferred from each other.
"""

from __future__ import annotations

import math
import warnings

SEMANTIC_ATTRIBUTES = (
    "activation",
    "inhibition",
    "catalysis",
    "expression_regulation",
    "complex",
    "co_input",
    "reaction_association",
    "predicted",
)

# Lower-cased annotation token -> semantic attribute.
ANNOTATION_TO_ATTRIBUTE = {
    "activate": "activation",
    "activated by": "activation",
    "inhibit": "inhibition",
    "inhibited by": "inhibition",
    "catalyze": "catalysis",
    "catalyzed by": "catalysis",
    "expression regulates": "expression_regulation",
    "expression regulated by": "expression_regulation",
    "complex": "complex",
    "input": "co_input",
    "reaction": "reaction_association",
    "predicted": "predicted",
}

# Direction token -> (direction_type, orientation).
# orientation is "forward" (Gene1 -> Gene2), "reverse" (Gene2 -> Gene1), or None.
DIRECTION_TABLE = {
    "<-": ("directed", "reverse"),
    "-": ("undirected", None),
    "-|": ("directed", "forward"),
    "->": ("directed", "forward"),
    "|-": ("directed", "reverse"),
    "<->": ("bidirectional", None),
    "|->": ("bidirectional", None),
    "<-|": ("bidirectional", None),
    "|-|": ("bidirectional", None),
}

_warned_directions: set[str] = set()


def parse_annotation(annotation: str | None) -> dict:
    """Split a semicolon-separated annotation into a multi-hot semantic vector.

    Tokens are stripped and matched without case. Unrecognised tokens are kept
    in ``unknown_annotation_tokens`` and do not set a semantic bit.
    """
    raw = "" if annotation is None else str(annotation)
    tokens = [token.strip() for token in raw.split(";") if token.strip()]
    attributes = {name: 0 for name in SEMANTIC_ATTRIBUTES}
    unknown: list[str] = []
    for token in tokens:
        attribute = ANNOTATION_TO_ATTRIBUTE.get(token.lower())
        if attribute is None:
            unknown.append(token)
        else:
            attributes[attribute] = 1
    vector = [attributes[name] for name in SEMANTIC_ATTRIBUTES]
    return {
        "annotation_raw": raw,
        "annotation_tokens_raw": tokens,
        "unknown_annotation_tokens": unknown,
        "semantic_attributes": attributes,
        "edge_type_vector": vector,
    }


def parse_direction(direction: str | None, gene1: str, gene2: str) -> dict:
    """Map a Direction cell onto directed, undirected, or bidirectional.

    Undirected and bidirectional records keep ``source`` and ``target`` empty.
    They are not expanded into a pair of message-passing edges. An unknown
    symbol warns once and is left as ``direction_type='unknown'``.
    """
    raw = "" if direction is None else str(direction)
    key = raw.strip()
    spec = DIRECTION_TABLE.get(key)
    if spec is None:
        if key not in _warned_directions:
            _warned_directions.add(key)
            warnings.warn(
                f"unknown FI direction {key!r}; left uninterpreted",
                UserWarning,
                stacklevel=2,
            )
        return {
            "direction_raw": raw,
            "direction_type": "unknown",
            "source": None,
            "target": None,
        }
    direction_type, orientation = spec
    if orientation == "forward":
        source, target = gene1, gene2
    elif orientation == "reverse":
        source, target = gene2, gene1
    else:
        source, target = None, None
    return {
        "direction_raw": raw,
        "direction_type": direction_type,
        "source": source,
        "target": target,
    }


def parse_confidence(value) -> float | None:
    """Parse FI evidence support. Missing or non-numeric values become None.

    The number is not an activation strength or an effect size.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
