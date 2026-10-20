from __future__ import annotations

import sys
from pathlib import Path

import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.fi.prior import align_fi_prior
from src.train.model import SampleGraphClassifier


def test_missing_fi_genes_are_zero_and_matched_rows_are_copied(tmp_path: Path):
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    path = tmp_path / "fi_gene_embeddings.pt"
    torch.save({"embeddings": embeddings, "genes": ["TP53", "EGFR", "MYC"]}, path)
    table, coverage = align_fi_prior(["EGFR", "MISSING", "TP53"], path)
    assert coverage["n_htg_genes"] == 3
    assert coverage["n_fi_genes"] == 3
    assert coverage["overlap"] == 2
    assert coverage["n_with_fi"] == 2
    assert coverage["n_without_fi"] == 1
    assert torch.equal(table[0], embeddings[1])
    assert torch.equal(table[1], torch.zeros(2))
    assert torch.equal(table[2], embeddings[0])


def test_fi_prior_is_frozen_and_projection_is_trainable():
    table = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    model = SampleGraphClassifier(2, hidden_dim=4, encoder="sage", fi_embeddings=table)
    assert model.fi_proj is not None
    assert model.fi_proj.bias is None
    assert not model.fi_embeddings.requires_grad
    assert model.fi_proj.weight.requires_grad
    assert "fi_embeddings" not in {name for name, _param in model.named_parameters()}
    before = model.fi_embeddings.clone()
    weight = model.fi_proj.weight.detach().clone()
    loss = model.fi_proj(model.fi_embeddings).sum()
    loss.backward()
    model.fi_proj.weight.data -= model.fi_proj.weight.grad
    assert torch.equal(model.fi_embeddings, before)
    assert not torch.equal(model.fi_proj.weight, weight)
    assert torch.equal(model.fi_proj(model.fi_embeddings[1]), torch.zeros(4))


def test_none_prior_leaves_the_base_embedding():
    model = SampleGraphClassifier(2, hidden_dim=4, encoder="sage")
    assert model.fi_proj is None
