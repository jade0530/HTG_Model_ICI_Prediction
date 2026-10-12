from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.fi.parse import parse_annotation, parse_direction


def test_activate_complex_vector():
    parsed = parse_annotation("activate; complex")
    assert parsed["edge_type_vector"] == [1, 0, 0, 0, 1, 0, 0, 0]


def test_inhibit_reaction_vector():
    parsed = parse_annotation("inhibit; reaction")
    assert parsed["edge_type_vector"] == [0, 1, 0, 0, 0, 0, 1, 0]


def test_predicted_vector():
    parsed = parse_annotation("predicted")
    assert parsed["edge_type_vector"] == [0, 0, 0, 0, 0, 0, 0, 1]


def test_multi_label_example_keeps_every_attribute():
    parsed = parse_annotation(
        "activate; activated by; catalyze; catalyzed by; complex; inhibit; input; reaction"
    )
    assert parsed["edge_type_vector"] == [1, 1, 1, 0, 1, 1, 1, 0]
    assert parsed["semantic_attributes"]["expression_regulation"] == 0
    assert parsed["semantic_attributes"]["predicted"] == 0


def test_unknown_annotation_is_preserved():
    parsed = parse_annotation("activate; binding")
    assert parsed["edge_type_vector"] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert parsed["unknown_annotation_tokens"] == ["binding"]
    assert parsed["annotation_tokens_raw"] == ["activate", "binding"]


@pytest.mark.parametrize(
    ("direction", "source", "target", "direction_type"),
    [
        ("->", "A", "B", "directed"),
        ("<-", "B", "A", "directed"),
        ("-|", "A", "B", "directed"),
        ("|-", "B", "A", "directed"),
        ("-", None, None, "undirected"),
        ("<->", None, None, "bidirectional"),
        ("|->", None, None, "bidirectional"),
        ("<-|", None, None, "bidirectional"),
        ("|-|", None, None, "bidirectional"),
    ],
)
def test_direction_parsing(direction, source, target, direction_type):
    parsed = parse_direction(direction, "A", "B")
    assert parsed["source"] == source
    assert parsed["target"] == target
    assert parsed["direction_type"] == direction_type
    assert parsed["direction_raw"] == direction


def test_unknown_direction_warns_and_is_not_interpreted():
    with pytest.warns(UserWarning, match="unknown FI direction"):
        parsed = parse_direction("??", "A", "B")
    assert parsed["direction_type"] == "unknown"
    assert parsed["source"] is None
    assert parsed["target"] is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = parse_direction("??", "A", "B")
    assert again["direction_type"] == "unknown"
    assert caught == []
