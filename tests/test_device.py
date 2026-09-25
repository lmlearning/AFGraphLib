import pytest

from GraphLib.device import send_graph_to_device


class Tensor:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def to(self, device, **kwargs):
        self.calls.append((device, kwargs))
        if self.fail:
            raise RuntimeError("transfer failed")
        return ("converted", device)


class Graph:
    def __init__(self, nodes=None, edges=None):
        self.ndata = nodes or {}
        self.edata = edges or {}

    def node_attr_schemes(self):
        return dict.fromkeys(self.ndata)

    def edge_attr_schemes(self):
        return dict.fromkeys(self.edata)


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_transfers_both_attribute_kinds_to_requested_device(device):
    node, edge = Tensor(), Tensor()
    graph = Graph({"features": node}, {"weights": edge})
    assert send_graph_to_device(graph, device) is graph
    assert graph.ndata == {"features": ("converted", device)}
    assert graph.edata == {"weights": ("converted", device)}
    assert node.calls == edge.calls == [(device, {"non_blocking": True})]


@pytest.mark.parametrize("attribute_kind", ["ndata", "edata"])
def test_failed_transfer_keeps_original_attribute(attribute_kind):
    tensor = Tensor(fail=True)
    graph = Graph()
    getattr(graph, attribute_kind)["features"] = tensor
    with pytest.raises(RuntimeError, match="transfer failed"):
        send_graph_to_device(graph, "cpu")
    assert getattr(graph, attribute_kind)["features"] is tensor


def test_empty_graph_is_unchanged():
    graph = Graph()
    assert send_graph_to_device(g=graph, device="cpu") is graph
    assert graph.ndata == graph.edata == {}
