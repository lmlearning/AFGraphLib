import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx
from dgl.nn.pytorch import GraphConv
from model import GCN
import numpy as np
from util import parseTGF,parseAPX, get_features, read_solution_file, get_credulous_labels, get_sceptical_labels, get_masks

def load_graph(file_path, cutoff = 10000000, format="tgf"):
    
    if format== "tgf":
        args, atts = parseTGF(file_path)
    else:
        args, atts = parseAPX(file_path)
    #print(args)
    if (len(args) > cutoff):
        return None, None, None
     #Create graph
    G = nx.DiGraph()
    #G.add_nodes_from(args)
    name_to_idx = {}
    idx_to_name = {}
    for idx, arg in enumerate(args):
        #print(idx,arg)
        G.add_node(idx)
        name_to_idx[arg] = idx
        idx_to_name[idx] = arg
    
    for att in atts:
        G.add_edge(name_to_idx[att[0]], name_to_idx[att[1]])
        
    return G, idx_to_name, name_to_idx

def make_dgl_graph(graph_file, solution_file, label_func = get_credulous_labels, cutoff = 10000000, format="tgf",  error_file_mode=False):
    
    #Load AF into Nx Digraph
    DiGraph, idx_to_name, name_to_idx  = load_graph(graph_file, cutoff, format)
    
    #check cutoff
    if DiGraph == None:
        return None, None, None, None
    
    #Preprocess and Load Features using Node id as feature
    features = get_features(DiGraph.number_of_nodes(), 1)

    #Load labels from solution files
    labels   = label_func(DiGraph, idx_to_name, solution_file, name_to_idx,  error_file_mode)

    #All masks set as in synthetic dataset
    #Set training mask
    #Set validation mask
    #Set test mask
    train_mask, val_mask, test_mask = get_masks(DiGraph.number_of_nodes(), 1, .2, .2)

    #Create DGL graph
    g = dgl.DGLGraph()
    g.from_networkx(DiGraph)
    return g, features, labels, train_mask

def merge_graphs(graphs, feature_arrays, label_arrays, training_masks):
    if th.cuda.is_available():
        for gf in graphs:
            gf = send_graph_to_device(gf, "cuda:0")
    g = dgl.batch(graphs)
    features = np.concatenate(feature_arrays)
    labels   = np.concatenate(label_arrays)
    training_masks = np.concatenate(training_masks)
    return g, features, labels, training_masks

def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(DEVICE, non_blocking=True)
    
    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(DEVICE, non_blocking=True)
    return g
