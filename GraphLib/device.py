"""Tensor-attribute transfer for the legacy DGL graph interface."""


def send_graph_to_device(g, device):
    """Move node/edge attributes in place without removing them before transfer.

    This preserves the legacy helper's graph identity and topology behavior.
    If a tensor transfer fails, its source attribute remains available.
    """
    for name in g.node_attr_schemes():
        g.ndata[name] = g.ndata[name].to(device, non_blocking=True)
    for name in g.edge_attr_schemes():
        g.edata[name] = g.edata[name].to(device, non_blocking=True)
    return g
