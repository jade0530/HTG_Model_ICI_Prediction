import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from src.train.dataset import GraphFolderDataset, load_graph_index


def _graph(sample_id: str, label: str) -> HeteroData:
    graph = HeteroData()
    graph["cell"].x = torch.ones(1, 1)
    graph.sample_id = sample_id
    graph.y = torch.tensor([1.0 if label == "R" else 0.0])
    return graph


def test_graph_folder_loads_the_indexed_samples(tmp_path: Path) -> None:
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    torch.save(_graph("S1", "R"), graphs / "S1.pt")
    torch.save(_graph("S2", "NR"), graphs / "S2.pt")
    rows = [
        {"sample_id": "S1", "patient_id": "P1", "label": "R", "path": str(graphs / "S1.pt")},
        {"sample_id": "S2", "patient_id": "P2", "label": "NR", "path": str(graphs / "S2.pt")},
    ]
    (tmp_path / "index.json").write_text(json.dumps({"graphs": rows}))
    (tmp_path / "gene_universe.txt").write_text("TP53\n")

    loaded = load_graph_index(tmp_path)
    assert [row.sample_id for row in loaded] == ["S1", "S2"]
    assert loaded[0].patient_key == "P1"

    one = GraphFolderDataset(tmp_path, ["S2"])
    assert len(one) == 1
    assert one[0].sample_id == "S2"
    assert float(one[0].y) == 0.0
