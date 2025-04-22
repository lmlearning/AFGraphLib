#!/usr/bin/env python
"""
train_pyg_gnn.py
================
PyTorch Geometric pipeline for binary acceptability prediction on TGF
argumentation frameworks.

Key features
------------
* Graph‑level _or_ node‑level train/val/test splits.
* Optional node statistics and/or HOPE embeddings.
* 128‑d input padded with Xavier noise.
* Focal BCE + class weighting + stratified sub‑sampling.
* Cosine LR schedule, early stopping on MCC.
* Detailed epoch metrics (acc, pos/neg acc, precision, NPV, MCC).
"""

from __future__ import annotations

# ───────────────────────── std / 3rd‑party ──────────────────────────
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, LayerNorm
from tqdm import tqdm
from torch_geometric.nn import GraphConv

try:
    from gem.embedding.hope import HOPE as GEM_HOPE
except ImportError:
    GEM_HOPE = None

# ─────────────────────────── utilities ──────────────────────────────


def parse_tgf(file: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    args, atts, seen_hash = [], [], False
    with file.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln == "#":
                seen_hash = True
                continue
            (args if not seen_hash else atts).append(ln)
    atts = [tuple(e.split()) for e in atts]
    return args, atts


def reindex_nodes(g: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, int]]:
    mp = {n: i for i, n in enumerate(g.nodes())}
    return nx.relabel_nodes(g, mp), mp


def hope_embeddings(g: nx.DiGraph, dim: int, cache: Path) -> np.ndarray:
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    if GEM_HOPE is None:
        raise RuntimeError("`gem` not installed but --use_hope_embedding set.")
    d_eff = min(dim, g.number_of_nodes())
    emb = GEM_HOPE(d=d_eff, beta=0.01).learn_embedding(g, no_python=True)
    emb = np.pad(emb, ((0, 0), (0, dim - d_eff)), constant_values=0)
    cache.write_bytes(pickle.dumps(emb))
    return emb


def node_features(g: nx.DiGraph, cache: Path) -> Dict[int, np.ndarray]:
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    colour = nx.algorithms.coloring.greedy_color(g, strategy="largest_first")
    pr = nx.pagerank(g)
    closeness = nx.degree_centrality(g)
    eig = nx.eigenvector_centrality(g, max_iter=10_000)
    indeg, outdeg = dict(g.in_degree()), dict(g.out_degree())

    raw = {
        n: np.array(
            [colour[n], pr[n], closeness[n], eig[n], indeg[n], outdeg[n]],
            dtype=np.float32,
        )
        for n in g.nodes()
    }
    mat = StandardScaler().fit_transform(np.stack(list(raw.values())))
    feats = {n: mat[i] for i, n in enumerate(g.nodes())}
    cache.write_bytes(pickle.dumps(feats))
    return feats


def read_solution(path: Path, mode: str = "any") -> List[str]:
    with path.open() as f:
        first = f.readline().strip()
        line = first if first.startswith("[[") else f.readline().strip()
    line = line[1:-1].replace("]]", "")
    ext_lists = [seg[1:].split(",") for seg in line.split("],")]
    if mode == "all":
        pos = set.intersection(*(set(lst) for lst in ext_lists))
    else:
        pos = {x for sub in ext_lists for x in sub}
    return [x.strip().replace("]", "") for x in pos]


# ─────────────────────────── dataset ────────────────────────────────


class TGFGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        *,
        use_node_features: bool,
        use_hope: bool,
        hope_dim: int = 32,
        label_ext: str = ".EE-PR",
        training_mode: str = "credulous",
        split_on_nodes: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.root = root
        self.use_node_features = use_node_features
        self.use_hope = use_hope
        self.hope_dim = hope_dim
        self.label_ext = label_ext
        self.training_mode = training_mode
        self.split_on_nodes = split_on_nodes
        self.train_ratio, self.val_ratio, self.test_ratio = (
            train_ratio,
            val_ratio,
            test_ratio,
        )
        self.files = [
            f
            for f in root.glob("*.tgf")
            if (root / (f.stem + label_ext)).exists()
        ]
        if not self.files:
            raise RuntimeError(f"No .tgf found in {root}")
        self._mask_cache: Dict[Path, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _masks_for(self, n_nodes: int, path: Path) -> Tuple[torch.Tensor, ...]:
        if path in self._mask_cache:
            return self._mask_cache[path]
        perm = torch.randperm(n_nodes)
        n_tr = int(self.train_ratio * n_nodes)
        n_va = int(self.val_ratio * n_nodes)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[perm[:n_tr]] = True
        val_mask[perm[n_tr : n_tr + n_va]] = True
        test_mask[perm[n_tr + n_va :]] = True
        self._mask_cache[path] = (train_mask, val_mask, test_mask)
        return train_mask, val_mask, test_mask

    def __getitem__(self, idx: int) -> Data:  # type: ignore
        tgf_path = self.files[idx]
        nodes, edges = parse_tgf(tgf_path)
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g, mapping = reindex_nodes(g)

        # labels
        sol_path = tgf_path.with_suffix(self.label_ext)
        mode = "all" if self.training_mode == "sceptical" else "any"
        pos_names = set(read_solution(sol_path, mode))
        y = torch.zeros(g.number_of_nodes(), 1, dtype=torch.float32)
        for name in pos_names:
            if name in mapping:
                y[mapping[name]] = 1.0

        # features parts
        parts: List[np.ndarray] = []
        if self.use_node_features:
            parts.append(
                np.stack(
                    [
                        node_features(
                            g, tgf_path.with_suffix(".features.pkl")
                        )[n]
                        for n in g.nodes()
                    ]
                )
            )
        if self.use_hope:
            parts.append(
                hope_embeddings(g, self.hope_dim, tgf_path.with_suffix(".hope.pkl"))
            )

        FEATURE_DIM = 128
        if parts:
            x_real = np.concatenate(parts, axis=1).astype(np.float32)
            pad = FEATURE_DIM - x_real.shape[1]
            if pad > 0:
                x_pad = torch.empty((x_real.shape[0], pad), dtype=torch.float32)
                nn.init.xavier_uniform_(x_pad)
                x = torch.cat([torch.tensor(x_real), x_pad], dim=1)
            else:
                x = torch.tensor(x_real)
        else:
            x = torch.empty((g.number_of_nodes(), FEATURE_DIM), dtype=torch.float32)
            nn.init.xavier_uniform_(x)

        edge_index = torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()

        train_mask = val_mask = test_mask = None
        if self.split_on_nodes:
            train_mask, val_mask, test_mask = self._masks_for(
                g.number_of_nodes(), tgf_path
            )

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )


# ─────────────────────────── models ────────────────────────────────


class _BlockGNN(nn.Module):
    def _block(self, conv: nn.Module, norm: nn.Module, h, edge_index):
        h = conv(h, edge_index)
        h = norm(h)
        return F.relu(h)
    
class AFGCN(nn.Module):
    """
    AFGCN in PyG: GraphConv + ReLU + Dropout with a residual
    connection that adds the input features after every layer.
    """
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 num_classes: int,
                 num_layers: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be ≥1")

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim))
        for _ in range(1, num_layers):
            self.convs.append(GraphConv(hid_dim, hid_dim))

        self.fc = nn.Linear(hid_dim, num_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, data, *, edge_index=None):
        """
        Accept either:
          • data  – a torch_geometric.data.Data with .x and .edge_index
          • x,edge_index – if you pass them explicitly via kwargs
        """
        x, edge_index = (data.x, data.edge_index) if edge_index is None else (data, edge_index)

        h0 = x                          # for residual
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dp(h)
            h = h + h0                  # additive skip

        return self.fc(h).squeeze(-1)   # (N,) logits


# ─────────────────────────────────────────────────────────────
# 2. RandAlign‑GCN  – PyG version
# ─────────────────────────────────────────────────────────────
class RandAlignGCN(nn.Module):
    """
    RandAlign regularises layer‑to‑layer representations by mixing
    the ℓ‑2‑normalised outputs with a random α each step.
    """
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 num_classes: int,
                 num_layers: int = 3):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be ≥2")

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim))        # first
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hid_dim, hid_dim))   # hidden
        self.convs.append(GraphConv(hid_dim, num_classes))   # output

    # ---------- util ----------
    @staticmethod
    def _randalign(h_prev: torch.Tensor, h_curr: torch.Tensor) -> torch.Tensor:
        α = torch.rand(1, device=h_prev.device).item()
        norm_prev = torch.norm(h_prev, p=2, dim=1, keepdim=True)
        norm_curr = torch.norm(h_curr, p=2, dim=1, keepdim=True)
        scaled_prev = h_prev * (norm_curr / (norm_prev + 1e-9))
        return α * h_curr + (1 - α) * scaled_prev
    # --------------------------

    def forward(self, data, *, edge_index=None):
        x, edge_index = (data.x, data.edge_index) if edge_index is None else (data, edge_index)

        h = x
        for i, conv in enumerate(self.convs[:-1], start=1):
            h_next = F.relu(conv(h, edge_index))
            if i > 1:                               # skip RandAlign after 1st layer
                h = self._randalign(h, h_next)
            else:
                h = h_next

        logits = self.convs[-1](h, edge_index)      # final layer – no RandAlign
        return logits.squeeze(-1)                   # (N,) logits


class GCN(_BlockGNN):
    def __init__(self, in_dim: int, hid: int, out_dim: int, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hid), GCNConv(hid, hid), GCNConv(hid, hid)]
        )
        self.norms = nn.ModuleList([LayerNorm(hid) for _ in range(3)])
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, out_dim)

    def forward(self, data: Data):  # type: ignore
        h, ei = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            h = self._block(conv, norm, h, ei)
            h = self.dp(h)
        return self.fc(h).squeeze(-1)


class GAT(_BlockGNN):
    def __init__(self, in_dim: int, hid: int, out_dim: int, heads=8, dropout=0.5):
        super().__init__()
        self.l1 = GATConv(in_dim, hid, heads)
        self.l2 = GATConv(hid * heads, hid, heads)
        self.n1 = LayerNorm(hid * heads)
        self.n2 = LayerNorm(hid * heads)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hid * heads, out_dim)

    def forward(self, data: Data):  # type: ignore
        h, ei = data.x, data.edge_index
        h = self._block(self.l1, self.n1, h, ei)
        h = self.dp(h)
        h = self._block(self.l2, self.n2, h, ei)
        h = self.dp(h)
        return self.fc(h).squeeze(-1)


class GraphSAGE(_BlockGNN):
    def __init__(self, in_dim: int, hid: int, out_dim: int, dropout=0.5):
        super().__init__()
        self.l1 = SAGEConv(in_dim, hid)
        self.l2 = SAGEConv(hid, hid)
        self.n1 = LayerNorm(hid)
        self.n2 = LayerNorm(hid)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, out_dim)

    def forward(self, data: Data):  # type: ignore
        h, ei = data.x, data.edge_index
        h = self._block(self.l1, self.n1, h, ei)
        h = self.dp(h)
        h = self._block(self.l2, self.n2, h, ei)
        h = self.dp(h)
        return self.fc(h).squeeze(-1)


class GIN(_BlockGNN):
    def __init__(self, in_dim: int, hid: int, out_dim: int, dropout=0.5):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        mlp2 = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.c1 = GINConv(mlp1)
        self.c2 = GINConv(mlp2)
        self.n1 = LayerNorm(hid)
        self.n2 = LayerNorm(hid)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, out_dim)

    def forward(self, data: Data):  # type: ignore
        h, ei = data.x, data.edge_index
        h = self._block(self.c1, self.n1, h, ei)
        h = self.dp(h)
        h = self._block(self.c2, self.n2, h, ei)
        h = self.dp(h)
        return self.fc(h).squeeze(-1)


# ────────────────────────── loss & metrics ──────────────────────────


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits).detach()
        mod = (1 - p) ** self.gamma
        return (mod * bce).mean()


def confusion_counts(pred, true):
    tp = ((pred == 1) & (true == 1)).sum().item()
    tn = ((pred == 0) & (true == 0)).sum().item()
    fp = ((pred == 1) & (true == 0)).sum().item()
    fn = ((pred == 0) & (true == 1)).sum().item()
    return tp, tn, fp, fn


def compute_metrics(logits, targets):
    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob > 0.5).astype(int)
    true = targets.cpu().numpy()

    tp, tn, fp, fn = confusion_counts(pred, true)
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0.0
    pos_acc = tp / (tp + fn) if (tp + fn) else 0.0  # recall
    neg_acc = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    mcc = matthews_corrcoef(true, pred) if tp + fp + fn else 0.0
    return dict(
        acc=acc,
        pos_acc=pos_acc,
        neg_acc=neg_acc,
        precision=precision,
        npv=npv,
        mcc=mcc,
    )


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.count = 0
        self.stop = False

    def step(self, metric):
        if self.best is None or metric > self.best + self.delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True


# ───────────────────── training / evaluation loops ──────────────────


def train_epoch(
    model,
    loader,
    loss_fn,
    optim,
    device,
    *,
    use_subset,
    subset_ratio,
):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optim.zero_grad(set_to_none=True)

        logits = model(batch)
        y = batch.y.squeeze(-1)

        # node masks (node‑level split)
        if batch.train_mask is not None:
            mask = batch.train_mask
            logits, y = logits[mask], y[mask]

        # stratified sub‑sampling
        if use_subset:
            pos_idx = (y == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y == 0).nonzero(as_tuple=True)[0]
            k = max(1, int(subset_ratio * len(y)))
            k_pos = min(len(pos_idx), max(1, int(len(pos_idx) / len(y) * k)))
            k_neg = max(0, k - k_pos)
            sel = torch.cat(
                [
                    pos_idx[torch.randperm(len(pos_idx), device=device)[:k_pos]],
                    neg_idx[torch.randperm(len(neg_idx), device=device)[:k_neg]],
                ]
            )
            logits, y = logits[sel], y[sel]

        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        running_loss += loss.item() * len(y)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, phase="val"):
    model.eval()
    tot_loss = 0.0
    totals = dict(tp=0, tn=0, fp=0, fn=0)

    for batch in tqdm(loader, desc=phase, leave=False):
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.squeeze(-1)
        mask = getattr(batch, f"{phase}_mask") if batch.val_mask is not None else None
        if mask is not None:
            logits, y = logits[mask], y[mask]
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * len(y)

        prob = torch.sigmoid(logits).cpu() > 0.5
        tp, tn, fp, fn = confusion_counts(prob.numpy(), y.cpu().numpy())
        totals["tp"] += tp
        totals["tn"] += tn
        totals["fp"] += fp
        totals["fn"] += fn

    tp, tn, fp, fn = totals.values()
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0.0
    pos_acc = tp / (tp + fn) if (tp + fn) else 0.0
    neg_acc = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    mcc = matthews_corrcoef(
        np.concatenate([np.ones(tp + fn), np.zeros(tn + fp)]),
        np.concatenate([np.ones(tp), np.zeros(fn), np.zeros(tn), np.ones(fp)]),
    ) if tp + fp + fn else 0.0

    metrics = dict(
        loss=tot_loss / tot,
        acc=acc,
        pos_acc=pos_acc,
        neg_acc=neg_acc,
        precision=precision,
        npv=npv,
        mcc=mcc,
    )
    return metrics


# ───────────────────────── CLI & main ───────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    # split‑choice
    p.add_argument("--training_dir")
    p.add_argument("--validation_dir")
    p.add_argument("--data_dir", help="Single dir for node‑level split")
    p.add_argument("--split_on_nodes", action="store_true")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    # model/feature flags
    p.add_argument("--model_type", choices=["AFGCN", "GCN", "GAT", "GraphSAGE", "GIN"], default="AFGCN")
    p.add_argument("--use_node_features", action="store_true")
    p.add_argument("--use_hope_embedding", action="store_true")
    # training hyper‑params
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--label_ext", default=".EE-PR")
    p.add_argument("--training_mode", default="credulous")
    p.add_argument("--checkpoint", default="best_model.pth")
    # subset sampling
    p.add_argument("--use_subset", action="store_true")
    p.add_argument("--subset_ratio", type=float, default=0.5)
    return p.parse_args()


def count_pos_neg(loader, device):
    pos = neg = 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.squeeze(-1)
        mask = batch.train_mask if batch.train_mask is not None else None
        if mask is not None:
            y = y[mask]
        pos += int(y.sum().item())
        neg += int((1 - y).sum().item())
    return pos, neg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.split_on_nodes:
        root = Path(args.data_dir)
        ds = TGFGraphDataset(
            root,
            use_node_features=args.use_node_features,
            use_hope=args.use_hope_embedding,
            label_ext=args.label_ext,
            split_on_nodes=True,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        train_loader = DataLoader(ds, args.batch_size, shuffle=True)
        val_loader = DataLoader(ds, args.batch_size)
    else:
        # graph‑level split
        train_ds = TGFGraphDataset(
            Path(args.training_dir),
            use_node_features=args.use_node_features,
            use_hope=args.use_hope_embedding,
            label_ext=args.label_ext,
            training_mode=args.training_mode
        )
        val_ds = TGFGraphDataset(
            Path(args.validation_dir),
            use_node_features=args.use_node_features,
            use_hope=args.use_hope_embedding,
            label_ext=args.label_ext,
            training_mode=args.training_mode
        )
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, args.batch_size)

    in_dim = 128
    hid = 128
    Model = {"AFGCN": AFGCN, "GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GIN": GIN, "RandAlignGCN" : RandAlignGCN}[args.model_type]
    model = Model(in_dim, hid, 1).to(device)

    # class imbalance weight
    pos, neg = count_pos_neg(train_loader, device)
    pos_weight = torch.tensor([neg / (pos + 1e-8)], device=device)
    loss_fn = FocalBCEWithLogits(args.gamma, pos_weight)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs - args.warmup_epochs)
    )
    stopper = EarlyStopping(args.patience)

    for epoch in tqdm(range(1, args.epochs + 1), desc="epoch"):
        if epoch <= args.warmup_epochs:
            for g in optim.param_groups:
                g["lr"] = args.lr * epoch / args.warmup_epochs

        tr_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optim,
            device,
            use_subset=args.use_subset,
            subset_ratio=args.subset_ratio,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, phase="val")
        sched.step()

        print(
            f"Epoch {epoch:3d} | "
            f"tr_loss {tr_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"acc {val_metrics['acc']:.3f} | "
            f"+acc {val_metrics['pos_acc']:.3f} | "
            f"-acc {val_metrics['neg_acc']:.3f} | "
            f"prec {val_metrics['precision']:.3f} | "
            f"NPV {val_metrics['npv']:.3f} | "
            f"MCC {val_metrics['mcc']:.3f}"
        )

        stopper.step(val_metrics["mcc"])
        if stopper.best == val_metrics["mcc"]:
            torch.save(model.state_dict(), args.checkpoint)
        if stopper.stop:
            print(
                f"Early stopping after {epoch} epochs. "
                f"Best MCC = {stopper.best:.3f}"
            )
            break

    print(f"Training complete. Best model saved to '{args.checkpoint}'")


if __name__ == "__main__":
    main()
