import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx
from dglutil import make_dgl_graph, merge_graphs, send_graph_to_device, load_graph
from util import parseTGF,get_features, read_solution_file, get_credulous_labels, get_masks,getRandomBatch, load_ckp
from model import GCN
import argparse
import pickle
import numpy as np


BLANK  = 0
IN     = 1
OUT    = 2

def solve(adj_matrix):
    
    #set up labelling
    labelling = np.zeros((adj_matrix.shape[0]), np.int8)

    #find all unattacked arguments
    a = np.sum(adj_matrix, axis=0) == 0
    unattacked_args = np.nonzero(a)[0]
    #print(unattacked_args  )
    #label them in
    labelling[unattacked_args] = IN
    cascade = True
    while cascade:
        #find all outgoing attacks
        #print(np.nonzero(adj_matrix[unattacked_args,:]))
        new_attacks = np.unique(np.nonzero(adj_matrix[unattacked_args,:])[1])
        #print("NEW",new_attacks)
        new_attacks_l = np.array([i for i in new_attacks if labelling[i] != OUT])
        
        #label those out
        #print(new_attacks_l)
        if len(new_attacks_l) > 0:
            labelling[new_attacks_l] = OUT
            affected_idx = np.unique(np.nonzero(adj_matrix[new_attacks_l,:])[1])
        else:
            affected_idx = np.zeros((0), dtype='int64')
        #find any arguments that have all attackers labelled out
        
        all_outs = []
        for idx in affected_idx:
            incoming_attacks = np.nonzero(adj_matrix[:,idx])[0]
            if(np.sum(labelling[incoming_attacks] == OUT) == len(incoming_attacks)):
                all_outs.append(idx)

        #label those in
        if len(all_outs) > 0:
            labelling[np.array(all_outs)] = IN
            unattacked_args = np.array(all_outs)
        else:
            cascade = False
    
    #print grounded extension     
    in_nodes = np.nonzero(labelling == IN)[0]
    return in_nodes

def solve_all(adj_matrix, in_nodes = None):
    
    #print grounded extension
    if in_nodes is None:
        in_nodes = solve(adj_matrix)
    preds = np.zeros((adj_matrix.shape[0]), np.int8)
    preds[in_nodes] = 1.0
    return preds

def solve_and_predict_all(graph, task, gr_ext = None, threshold_in = None):
        
        if gr_ext is None:
            gr_ext = solve(nx.to_numpy_array(graph))
        
        preds = predict_all(graph, task, gr_ext, threshold_in)
        
        preds[gr_ext] = 1.0
    
        return preds
    
def solve_and_predict_heuristic_all(graph, task, threshold_in = None):
        
        if detect_admbuster(graph):
            return solve_all(nx.to_numpy_array(graph))
        
        gr_ext = solve(nx.to_numpy_array(graph))
        preds = predict_all(graph, task, gr_ext, threshold_in)
        
        preds[gr_ext] = 1.0
    
        return preds

def predict_all(graph, task, gr_ext = None, threshold_in = None):
    with th.no_grad():
        
        g = dgl.DGLGraph()
        g.from_networkx(graph)
        
        if gr_ext is None:
            gr_ext = solve(nx.to_numpy_array(graph))
       
        thresholds = dict()
        thresholds['DS-CO'] = 0.5
        thresholds['DC-CO'] = 0.5
        thresholds['DS-PR'] = 0.5
        thresholds['DC-PR'] = 0.5
        thresholds['DS-ST'] = 0.5
        thresholds['DC-ST'] = 0.5
        thresholds['DS-SST'] = 0.5
        thresholds['DC-SST'] = 0.5
        thresholds['DS-SST'] = 0.5
        thresholds['DC-SST'] = 0.5
        thresholds['DS-ID'] = 0.85

        features = np.zeros((graph.number_of_nodes(), 2) , dtype=np.float32)

        net = GCN(g, 2, 128, 1, 4, F.relu, 0.5)
        net.load_state_dict(th.load("/notebooks/" + task + ".dict"))
        
        if threshold_in is None:
            threshold = thresholds[task]
        else:
            threshold = threshold_in
            
        features  = th.FloatTensor(features)
        #nn.init.xavier_uniform_(features)
        features[:,1] = -1
        features[gr_ext,1] = 1

        logits    = net(features)
        logp      = th.sigmoid(logits)
        preds     = logp.view(-1) > threshold
        
        return preds.numpy()
    
    
def detect_admbuster(nx_graph):
    #has an even number of nodes
    if nx_graph.number_of_nodes() % 2 != 0 or nx_graph.number_of_nodes() < 200:
        return False
    
    #has in_degrees = 2 and out_degrees = 1 or the reverse
    #excepting the endpoints
    found_A = False
    found_D = False
    found_equal_node = False 
    num_even = 0
    num_odd  = 0
    
    for node in nx_graph.nodes:
        #print(nx_graph.in_degree(node), nx_graph.out_degree(node))
        if nx_graph.in_degree(node) == nx_graph.out_degree(node):
            continue
            #if found_equal_node == False:
            #    found_equal_node = True
            #    continue
            #else:
            #    return False
            
        if nx_graph.in_degree(node) == 0:
            if found_A:
                return False
            else:
                found_A = True
                continue
        elif nx_graph.out_degree(node)== 0:
            if found_D:
                return False
            else:
                found_D = True
                continue
        
        if nx_graph.in_degree(node) == 2 and nx_graph.out_degree(node) == 1:
            num_even += 1
            continue
        elif nx_graph.in_degree(node) == 1 and nx_graph.out_degree(node) == 2:
            num_odd += 1
            continue
            
        return False
    
    if abs(num_even - num_odd) > 2:
        return False
    else:
        return True
            
        
            