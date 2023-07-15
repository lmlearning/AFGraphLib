import dgl
import torch 
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import argparse
from dgl.nn import GraphConv
from sklearn.preprocessing import StandardScaler
import json
import os
import sys

BLANK  = 0
IN     = 1
OUT    = 2

def solve(adj_matrix):
    
    #set up labelling
    labelling = np.zeros((adj_matrix.shape[0]), np.int8)

    #find all unattacked arguments
    a = np.sum(adj_matrix, axis=0) == 0
    unattacked_args = np.nonzero(a)[0]

    #label them in
    labelling[unattacked_args] = IN
    cascade = True
    while cascade:
        #find all outgoing attacks)
        new_attacks = np.unique(np.nonzero(adj_matrix[unattacked_args,:])[1])
        new_attacks_l = np.array([i for i in new_attacks if labelling[i] != OUT])
        
        #label those out
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

class AFGCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, dropout=0.5):
        super(AFGCNModel, self).__init__()
        self.layer1 = GraphConv(in_features, hidden_features)
        self.layer2 = GraphConv(hidden_features, hidden_features)
        self.layer3 = GraphConv(hidden_features, hidden_features)
        self.layer4 = GraphConv(hidden_features, fc_features)
        self.fc = nn.Linear(fc_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer3(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer4(g, h + inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

def read_af_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    attacks = []
    args = []

    for line in lines:
        # Ignore comment lines
        if line.startswith('#'):
            continue

        # Split the line into parts
        parts = line.split()

        # If it's a p-line, extract the number of arguments and create args array
        if parts[0] == 'p' and parts[1] == 'af':
            num_args = int(parts[2])
            args = list([str(s) for s in range(1, num_args + 1)])

        # If it's an attack line, add the attack to the list of attacks
        elif len(parts) == 2:
            i, j = parts[0], parts[1]
            attacks.append([i, j])

    return args, attacks

def graph_coloring(nx_G):
    coloring = nx.algorithms.coloring.greedy_color(nx_G, strategy='largest_first')
    return coloring

def calculate_node_features(nx_G):
    coloring = graph_coloring(nx_G)
    page_rank = nx.pagerank(nx_G)
    closeness_centrality = nx.degree_centrality(nx_G)
    eigenvector_centrality = nx.eigenvector_centrality(nx_G, max_iter=10000)
    in_degrees = nx_G.in_degree()
    out_degrees = nx_G.out_degree()

    raw_features = {}
    for node in nx_G.nodes():
        raw_features[node] = [
            coloring[node],
            page_rank[node],
            closeness_centrality[node],
            eigenvector_centrality[node],
            in_degrees[node],
            out_degrees[node],
        ]

    # Normalize the features
    scaler = StandardScaler()
    nodes = list(nx_G.nodes())
    feature_matrix = scaler.fit_transform([raw_features[node] for node in nodes])

    # Create the normalized features dictionary
    normalized_features = {node: feature_matrix[i] for i, node in enumerate(nodes)}

    return normalized_features

def reindex_nodes(graph):
    mapping = {node.strip(): index for index, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping), mapping

def load_thresholds(file_path):
    with open(file_path, 'r') as f:
        thresholds = json.load(f)
    return thresholds


def main(cmd_args):
    
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) 
    
    args, atts = read_af_input(cmd_args.filepath)
    nxg = nx.DiGraph()
    nxg.add_nodes_from(args)
    nxg.add_edges_from(atts)
    nxg, mapping = reindex_nodes(nxg)
    graph = dgl.from_networkx(nxg)
    graph = dgl.add_self_loop(graph)

    with torch.no_grad():
        
        gr_ext = solve(nx.to_numpy_array(nxg))
        arg_id = mapping[cmd_args.argument]

        if arg_id in gr_ext:
            print( "YES" )
            return

        else:
            # Load thresholds from the JSON file
            threshold_path = os.path.join(__location__, cmd_args.thresholds_file)
            thresholds = load_thresholds(threshold_path)
            threshold = thresholds[cmd_args.task]
            
            if threshold == 1:
                print("NO")
                return

            inputs = torch.randn(graph.number_of_nodes(), cmd_args.in_features , dtype=torch.float)
            
            net = AFGCNModel(cmd_args.in_features, 128, 128, 1)
            checkpoint_path = os.path.join(__location__, cmd_args.task + ".pth")
            load_checkpoint(net, checkpoint_path)
            
            
            features  = calculate_node_features(nxg)
            features_tensor = torch.tensor(np.array([features[node] for node in nxg.nodes()]), dtype=torch.float)
            num_rows_to_overwrite = features_tensor.size(0)
            num_columns_in_features = features_tensor.size(1)
            inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_features)
            inputs_to_overwrite.copy_(features_tensor)
            outputs = net(graph, inputs)
            
            predicted = (torch.sigmoid(outputs.squeeze()) > threshold).float()

            if predicted[arg_id] == True:
                print("YES")
            else:
                print("NO")


# Define a function to load the model checkpoint and retrieve the associated loss
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=128 , help='number of input features')
    parser.add_argument('--filepath', type=str, default='' , help='file')
    parser.add_argument('--task', type=str, default='' , help='task')
    parser.add_argument('--argument', type=str, default='' , help='argument')
    parser.add_argument('--thresholds_file', type=str, default='thresholds.json', help='path to the thresholds JSON file')

    args = parser.parse_args()
    main(args)