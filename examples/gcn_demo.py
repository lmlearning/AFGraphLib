"""Fit the existing GCN to a four-argument illustrative framework on CPU."""
import json
import os
os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import torch
import torch.nn.functional as F

from GraphLib.model import GCN


def run_demo():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # a -> b -> c, plus isolated d. Grounded labels: a, c, d are accepted.
    graph = dgl.add_self_loop(dgl.graph(([0, 1], [1, 2]), num_nodes=4))
    features = torch.eye(4)
    labels = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    model = GCN(graph, in_feats=4, n_hidden=8, n_classes=1,
                n_layers=1, activation=F.relu, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial_loss = F.binary_cross_entropy_with_logits(model(features), labels).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(features), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(features)
        final_loss = F.binary_cross_entropy_with_logits(logits, labels).item()
    return {
        "example": "illustrative training fit, not a held-out benchmark",
        "arguments": 4, "attacks": 2, "logits_shape": list(logits.shape),
        "initial_loss": round(initial_loss, 6), "final_loss": round(final_loss, 6),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
