from sklearn.metrics import matthews_corrcoef
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import networkx as nx
import dgl
from torch.utils.data import Dataset, DataLoader
from dgl.nn import GraphConv, GCN2Conv
import dgl.nn as dglnn
import random
import sys
import pickle
import argparse
from gem.embedding.hope import HOPE as GEM_HOPE
import numpy as np
from sklearn.preprocessing import StandardScaler


def save_class_weights(weights, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)

def load_class_weights(filepath):
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)
    return weights

def reindex_nodes(graph):
    mapping = {node.strip(): index for index, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping), mapping

def graph_coloring(nx_G):
    coloring = nx.algorithms.coloring.greedy_color(nx_G, strategy='largest_first')
    return coloring

def train_hope_embedding(nx_G, dimensions=32):
    if nx_G.number_of_nodes() <= dimensions:
        dimensions_re = nx_G.number_of_nodes() 
    else:
        dimensions_re = dimensions
        
    # Use the HOPE class from the GEM library
    model = GEM_HOPE(d=dimensions_re, beta=0.01)  # beta is a hyperparameter for the HOPE model
    embeddings = model.learn_embedding(graph=nx_G, no_python=True)

    # Pad embeddings if their length is less than dimensions
    padded_embeddings = []
    for embedding in embeddings:
        if len(embedding) < dimensions:
            padding = dimensions - len(embedding)
            padded_embedding = np.pad(embedding, (0, padding), mode='constant', constant_values=0)
        else:
            padded_embedding = embedding
        padded_embeddings.append(padded_embedding)

    padded_embeddings = np.array(padded_embeddings)
    
    return padded_embeddings

def calculate_and_cache_hope_embedding(nx_G, cache_file, dimensions=32):

    embeddings = train_hope_embedding(nx_G, dimensions)

    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

def load_hope_embedding_cache(nx_G, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = calculate_and_cache_hope_embedding(nx_G, cache_file)

    return embeddings

def calculate_and_cache_node_features(nx_G, cache_file):
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

    # Save the normalized features to a file
    with open(cache_file, 'wb') as f:
        pickle.dump(normalized_features, f)

    return normalized_features

def load_graph_features_cache(nx_G, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
    else:
        features = calculate_and_cache_node_features(nx_G, cache_file)

    return features

#Utility function for parsing TGF
def parseTGF(file):
    f = open(file, 'r')
    
    args = []
    atts = []
    hash_seen = False
    for _, line in enumerate(f):
        #print('id={},val={}'.format(idx,line))
        line = line.strip()
        if line == '#':
            hash_seen = True
            continue
        if not hash_seen:
            args.append(line)
        else:
            atts.append(line.split(' '))
    
    return args, atts

class TGFGraphDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_node_features=True, use_hope_embedding=False, label_ext=".EE-PR", training_mode="credulous"):
        self.root_dir = root_dir
        self.transform = transform
        self.use_node_features = use_node_features
        self.use_hope_embedding = use_hope_embedding
        self.label_ext = label_ext
        self.hope_embedding_cache = {}
        self.graph_cache = {}
        self.training_mode = training_mode

        # List all files ending with '.tgf'
        all_tgf_files = [f for f in os.listdir(root_dir) if f.endswith('.tgf')]

        # Filter graph_files based on the existence of a matching solution file
        self.graph_files = []
        for tgf_file in all_tgf_files:
            file_name_without_ext = os.path.splitext(tgf_file)[0]
            solution_file = os.path.join(root_dir, file_name_without_ext + self.label_ext)
            if os.path.exists(solution_file):
                self.graph_files.append(tgf_file)

        self.graph_features_cache = {}

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        

        if torch.is_tensor(idx):
            idx = idx.tolist()

        graph_path = os.path.join(self.root_dir, self.graph_files[idx])
        cache_key = os.path.splitext(self.graph_files[idx])[0] 
        
        if cache_key in self.graph_cache:
            dgl_G, labels_tensor, features_tensor = self.graph_cache[cache_key]
            return dgl_G, labels_tensor, features_tensor

        args, atts = parseTGF(graph_path)
        nx_G = nx.DiGraph()
        nx_G.add_nodes_from(args)
        nx_G.add_edges_from(atts)
        nx_G, mapping = reindex_nodes(nx_G)

        # Read labels
        labels_file = os.path.splitext(graph_path)[0] + self.label_ext
        if self.training_mode == "sceptical":
            lmode = "all"
        else:
            lmode = "any"
        labels_names = read_solution_file(labels_file, label_mode=lmode)
        
        # Convert node names to binary labels
        labels_tensor = torch.zeros(nx_G.number_of_nodes(), dtype=torch.long)
        for idx, node in enumerate(labels_names):
            if len(node) > 0:
                labels_tensor[mapping[node]] = 1 if node in labels_names else 0

            

        if self.use_node_features:
            # Load or calculate node-level features
            
            features_cache_file = os.path.join(self.root_dir, cache_key + "_features.pkl")
            if cache_key in self.graph_features_cache:
                features = self.graph_features_cache[cache_key]
            elif os.path.exists(features_cache_file):
                features = load_graph_features_cache(nx_G, features_cache_file)
                self.graph_features_cache[cache_key] = features
            else:
                features = calculate_and_cache_node_features(nx_G, features_cache_file)
            
        if self.use_hope_embedding:
            hope_cache_file = os.path.join(self.root_dir, cache_key + "_hope.pkl")
            
            if cache_key in self.hope_embedding_cache:
                hope_embeddings = self.hope_embedding_cache[cache_key]
            elif os.path.exists(hope_cache_file):
                hope_embeddings = load_hope_embedding_cache(nx_G, hope_cache_file)
                self.hope_embedding_cache[cache_key] = hope_embeddings
            else:
                hope_embeddings = calculate_and_cache_hope_embedding(nx_G, hope_cache_file)
                self.hope_embedding_cache[cache_key] = hope_embeddings
            scaler = StandardScaler()
            hope_embeddings = scaler.fit_transform(hope_embeddings)
        
            
        dgl_G = dgl.from_networkx(nx_G)
        features_tensor = None
        if self.use_node_features and self.use_hope_embedding:
            combined_features = []
            for node in nx_G.nodes():
                node_features = features[node]
                hope_embedding = hope_embeddings[node]
                combined_feature = np.concatenate((node_features, hope_embedding), axis=-1)
                combined_features.append(combined_feature)
            features_tensor = torch.tensor(np.array(combined_features), dtype=torch.float)
        elif self.use_node_features:
            features_tensor = torch.tensor(np.array([features[node] for node in nx_G.nodes()]), dtype=torch.float)
        elif self.use_hope_embedding:
            features_tensor = torch.tensor(np.array([hope_embeddings[node] for node in nx_G.nodes()]), dtype=torch.float)
        else:
            features_tensor = torch.randn(nx_G.number_of_nodes(), 6)
        if self.transform:
            dgl_G = self.transform(dgl_G)

        dgl_G = dgl.add_self_loop(dgl_G)

        self.graph_cache[cache_key] = (dgl_G, labels_tensor, features_tensor)

        return dgl_G, labels_tensor, features_tensor
    
def read_solution_file(path_to_framework, strip_first_char=True, error_file_mode=False, label_mode="any"):
    f = open(path_to_framework, 'r')
    if error_file_mode:
        f.readline()
    input = f.readline()
    if strip_first_char == True:
        input = input[1:-1]
    input = input.replace("]]", "")
    in_arr = input.split('],')
    sol_arr = [s[1:].split(',') for s in in_arr]

    if label_mode == "all":
        flattened_sol_arr = list(set.intersection(*(set(sublist) for sublist in sol_arr)))
        flattened_sol_arr = [item.strip().replace("]","") for item in flattened_sol_arr]
    else:  # Default to "any" mode
        flattened_sol_arr = list(set(item.strip().replace("]","") for sublist in sol_arr for item in sublist))

    return flattened_sol_arr
    
def evaluate(model, loader, loss_function, device):
    model.eval()
    total = 0
    correct = 0
    predictions = []
    true_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batched_graph, label_list, input_list in loader:
            inputs = torch.randn(batched_graph.number_of_nodes(), in_features).to(device)
            batched_labels = torch.cat(label_list).to(device).float()
            batched_inputs = torch.cat(input_list).to(device).float()

            num_rows_to_overwrite = batched_inputs.size(0)
            num_columns_in_batched_inputs = batched_inputs.size(1)
            inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_batched_inputs)
            inputs_to_overwrite.copy_(batched_inputs)

            outputs = model(batched_graph.to(device), inputs)
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            batch_loss = loss_function(outputs.squeeze(), batched_labels)
            total_loss += batch_loss.item()

            total += batched_labels.size(0)
            correct += (predicted == batched_labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(batched_labels.cpu().numpy())

    accuracy = correct / total
    mcc = matthews_corrcoef(true_labels, predictions)
    avg_val_loss = total_loss / len(loader)  # Calculate the average validation loss
    return accuracy, mcc, avg_val_loss

def collate(samples):
    graphs, labels, inputs = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels, inputs  # Return labels as a list of tensors

class AFGCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, num_layers=4, dropout=0.5):
        super(AFGCNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Ensure there is at least one layer
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        
        # First graph convolution layer from input features to hidden features
        self.layers.append(GraphConv(in_features, hidden_features))
        
        # Additional hidden layers
        for _ in range(1, num_layers):
            self.layers.append(GraphConv(hidden_features, hidden_features))
        
        # Final fully connected layer that outputs to the number of classes
        self.fc = nn.Linear(hidden_features, num_classes)

    def forward(self, g, inputs):
        h = inputs
        for layer in self.layers:
            # Apply graph convolution
            h = layer(g, h)
            # Activation function
            h = F.relu(h)
            # Dropout
            h = self.dropout(h)
            # Skip connection: add the input features to the output of each layer
            h = h + inputs
        
        # Apply final fully connected layer
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

class RandAlignGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, num_layers):
        super(RandAlignGCN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Initial layer from input features to the first hidden layer
        self.layers.append(GraphConv(in_feats, hidden_feats))
        
        # Hidden layers
        for _ in range(num_layers - 2):  # num_layers includes the output layer
            self.layers.append(GraphConv(hidden_feats, hidden_feats))
        
        # Output layer
        self.layers.append(GraphConv(hidden_feats, num_classes))

    def apply_randalign(self, h_prev, h_curr):
        """
        Apply RandAlign regularization between two consecutive layers using a uniform alpha.
        """
        alpha = torch.rand(1).item()  # Scalar alpha, uniformly sampled
        norm_prev = torch.norm(h_prev, p=2, dim=1, keepdim=True)
        norm_curr = torch.norm(h_curr, p=2, dim=1, keepdim=True)
        scaled_h_prev = h_prev * (norm_curr / norm_prev)
        return alpha * h_curr + (1 - alpha) * scaled_h_prev

    def forward(self, g, inputs):
        h = inputs
        for i in range(1, len(self.layers) - 1):
            h_next = self.layers[i](g, h)
            h_next = F.relu(h_next)
            if i == 1:  # Skip RandAlign for the first layer's output
                h = h_next
            else:
                h = self.apply_randalign(h, h_next)  # Apply RandAlign for subsequent layers

        # Output layer does not use RandAlign
        logits = self.layers[-1](g, h)
        return logits
    
class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, dropout=0.5):
        super(GCNModel, self).__init__()
        self.layer1 = GraphConv(in_features, hidden_features)
        self.layer2 = GraphConv(hidden_features, fc_features)
        self.fc = nn.Linear(fc_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

class GCN2Model(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, num_blocks, dropout=0.9, alpha=0.9, lambda_=1):
        super(GCN2Model, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.layers.append(GCN2Conv(in_feats=in_features, layer=1, alpha=alpha, lambda_=lambda_))
        self.dropout_layers.append(nn.Dropout(dropout))

        for l in range(2, num_blocks + 1):
            self.layers.append(GCN2Conv(in_feats=hidden_features, layer=l, alpha=alpha, lambda_=lambda_))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.final_layer = GCN2Conv(in_feats=hidden_features, layer=num_blocks + 1, alpha=alpha, lambda_=lambda_)
        self.fc = nn.Linear(fc_features, num_classes)

    def forward(self, g, inputs):
        self.orig_features = inputs
        h = inputs
        
        for layer, dropout in zip(self.layers, self.dropout_layers):
            h = layer(g, h, self.orig_features)
            h = F.relu(h)
            h = dropout(h)

        h = self.final_layer(g, h, self.orig_features)
        h = F.relu(h)
        h = self.fc(h)
        return h.squeeze()  # Remove the last dimension

class GATModel(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, num_heads, dropout=0.5):
        super(GATModel, self).__init__()
        self.layer1 = dglnn.GATConv(in_features, hidden_features, num_heads=num_heads)
        self.layer2 = dglnn.GATConv(hidden_features * num_heads, fc_features, num_heads=num_heads)
        self.fc = nn.Linear(fc_features * num_heads, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = F.elu(h)
        h = h.view(h.size(0), -1)  # Reshape to concatenate all heads
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = F.elu(h)
        h = h.view(h.size(0), -1)  # Reshape to concatenate all heads
        h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()

class GraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, aggregator_type='mean', dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        self.layer1 = dglnn.SAGEConv(in_features, hidden_features, aggregator_type)
        self.layer2 = dglnn.SAGEConv(hidden_features, fc_features, aggregator_type)
        self.fc = nn.Linear(fc_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h.squeeze()

class GINModel(nn.Module):
    def __init__(self, in_features, hidden_features, fc_features, num_classes, num_layers, dropout=0.5):
        super(GINModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.layers.append(dglnn.GINConv(nn.Linear(in_features, hidden_features), 'sum'))
        self.dropout_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GINConv(nn.Linear(hidden_features, hidden_features), 'sum'))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.final_layer = dglnn.GINConv(nn.Linear(hidden_features, fc_features), 'sum')
        self.fc = nn.Linear(fc_features, num_classes)

    def forward(self, g, inputs):
        h = inputs
        for layer, dropout in zip(self.layers, self.dropout_layers):
            h = layer(g, h)
            h = F.relu(h)
            h = dropout(h)

        h = self.final_layer(g, h)
        h = F.relu(h)
        h = self.fc(h)
        return h.squeeze()

# Define a function to save the model checkpoint along with the associated loss
def save_checkpoint(model, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)

# Define a function to load the model checkpoint and retrieve the associated loss
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def train(model, loader, loss_function, optimizer, device, subset_ratio=0.5, use_subset=True):
    model.to(device)
    model.train()
    for batch_idx, (batched_graph, label_list, input_list) in enumerate(loader):
        inputs = torch.randn(batched_graph.number_of_nodes(), in_features).to(device)

        #overwrite random features with real
        batched_inputs = torch.cat(input_list).to(device).float()
        num_rows_to_overwrite = batched_inputs.size(0)
        num_columns_in_batched_inputs = batched_inputs.size(1)
        inputs_to_overwrite = inputs.narrow(0, 0, num_rows_to_overwrite).narrow(1, 0, num_columns_in_batched_inputs)
        inputs_to_overwrite.copy_(batched_inputs)
        
        batched_labels = torch.cat(label_list).to(device).float()
        outputs = model(batched_graph.to(device), inputs)

        if use_subset:
            # Select a random subset of indices based on the subset_ratio
            subset_size = int(subset_ratio * len(batched_labels))
            subset_indices = random.sample(range(len(batched_labels)), subset_size)
            subset_outputs = outputs.squeeze()[subset_indices]
            subset_labels = batched_labels[subset_indices]
        else:
            # Use the full batch
            subset_outputs = outputs.squeeze()
            subset_labels = batched_labels

        # Calculate loss using the subset of predictions and labels
        loss = loss_function(subset_outputs, subset_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy and MCC
        predicted = (torch.sigmoid(subset_outputs) > 0.5).float()
        correct = (predicted == subset_labels).sum().item()
        total = subset_labels.size(0)
        batch_accuracy = correct / total
        batch_mcc = matthews_corrcoef(subset_labels.cpu().numpy(), predicted.cpu().numpy())

        # Print loss, batch accuracy, and MCC for the current batch
        print(f'Batch: {batch_idx+1}, Loss: {loss.item()}, Batch Accuracy: {batch_accuracy}, Batch MCC: {batch_mcc}')

def calculate_class_weights(dataset):
    num_samples = 0
    pos_count = 0
    print("here")
    for _, labels, _ in dataset:        
            num_samples += len(labels)
            pos_count += torch.sum(labels)
    print(num_samples)
    print(pos_count)
    class_weights = [(num_samples - pos_count) / pos_count] 
    return class_weights


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a GCN model on TGF graphs.')
    parser.add_argument('--use_node_features', action='store_true', default=False,
                        help='Use node-level features instead of Xavier-initialized random features.')
    parser.add_argument('--use_hope_embedding', action='store_true', default=False,
                        help='Use HOPE embeddings as an input feature for the GCN.')
    parser.add_argument('--model_type', type=str, choices=['GCNModel', 'GCN2Model', 'AFGCNModel', 'GATModel', 'GraphSAGEModel', 'GINModel', 'RandAlignGCN'], default='AFGCNModel',
                    help='Choose the model type: GCNModel, GCN2Model, AFGCNModel, GATModel, GraphSAGEModel, or GINModel.')
    parser.add_argument('--validation_dir', type=str, default='validation_data',
                        help='Directory containing validation data.')
    parser.add_argument('--training_dir', type=str, default='training_data',
                        help='Directory containing training data.')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers for GCN2Model.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory where checkpoints are stored.')
    parser.add_argument('--checkpoint_prefix', type=str, default='checkpoint',
                        help='Prefix for checkpoint files.')
    parser.add_argument('--solution_ext', type=str, default='EE-PR',
                        help='Postfix for solution files.')
    parser.add_argument('--use_subset', action='store_true', default=False,
                    help='Use a random subset of data for each training batch. Set to False to use the full batch.')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                    help='Apply class weights in the loss calculation to handle class imbalance.')

    args = parser.parse_args()

    # Set necessary parameters
    root_dir = args.training_dir
    validation_dir = args.validation_dir
    num_layers = args.num_layers
    checkpoint_dir = args.checkpoint_dir
    checkpoint_prefix = args.checkpoint_prefix
    solution_extension = args.solution_ext
    use_subset = args.use_subset

    if checkpoint_prefix[:2] == "DS":
        training_mode = "sceptical"
    else:
        training_mode = "credulous"

    in_features = 128
    hidden_features = 128
    fc_features = 128
    num_classes = 1
    batch_size = 300
    learning_rate = 0.01
    epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA Available:{torch.cuda.is_available()}")
    
    # Create dataset and DataLoader
    tgf_dataset = TGFGraphDataset(root_dir, use_node_features=args.use_node_features, use_hope_embedding=args.use_hope_embedding, label_ext=solution_extension, training_mode=training_mode)
    tgf_loader = DataLoader(tgf_dataset, batch_size=batch_size, shuffle=use_subset, collate_fn=collate)
    
    # Create a validation DataLoader
    validation_dataset = TGFGraphDataset(validation_dir, use_node_features=args.use_node_features, use_hope_embedding=args.use_hope_embedding, label_ext=solution_extension, training_mode=training_mode)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    print(len(tgf_loader))
    print(len(validation_loader))
    
    # Initialize the model, loss, and optimizer
    if args.model_type == 'GCNModel':
        model = GCNModel(in_features, hidden_features, fc_features, num_classes)
    elif args.model_type == 'AFGCNModel':
        model = AFGCNModel(in_features, hidden_features, num_classes,num_layers)
    elif args.model_type == 'GCN2Model':
        model = GCN2Model(in_features, hidden_features, fc_features, num_classes, num_layers)
    elif args.model_type == 'GATModel':
        model = GATModel(in_features, hidden_features, fc_features, num_classes, num_heads=8)
    elif args.model_type == 'GraphSAGEModel':
        model = GraphSAGEModel(in_features, hidden_features, fc_features, num_classes, aggregator_type='mean')
    elif args.model_type == 'GINModel':
        model = GINModel(in_features, hidden_features, fc_features, num_classes, num_layers=3)
    elif args.model_type == 'RandAlignGCN':
        model = RandAlignGCN(in_features, hidden_features, num_classes, num_layers=3)

    if args.use_class_weights:
        # Calculate or load class weights for weighted loss
        class_weights_filename = f'class_weights_{solution_extension}.pkl'
        class_weights_filepath = os.path.join(root_dir, class_weights_filename)
        if os.path.exists(class_weights_filepath):
            class_weights = load_class_weights(class_weights_filepath)
        else:
            class_weights = calculate_class_weights(tgf_dataset)
            save_class_weights(class_weights, class_weights_filepath)
        print("Calculated class weights:", class_weights)
        weighted_loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights, device=device))
    else:
        # Use standard BCE loss without weights
        weighted_loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define paths for checkpoints
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_prefix + '.pth')
    best_model_path = os.path.join(checkpoint_dir, checkpoint_prefix + '_best.pth')

    # Load checkpoint if it exists
    start_epoch = 0
    best_val_loss = sys.float_info.max
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(model, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}, best validation loss {best_val_loss}")

    # Train the model and evaluate every epoch
    for epoch in range(epochs):
        print(f"Epoch:{epoch}/{epochs}")

        # Call the train function with a subset_ratio (e.g., 0.5 for 50% of the data)
        train(model, tgf_loader, weighted_loss_function, optimizer, device, subset_ratio=0.5, use_subset=use_subset)

        # Evaluate on validation set and get the average validation loss
        accuracy, mcc, avg_val_loss = evaluate(model, validation_loader, weighted_loss_function, device)
        print(f'Epoch: {epoch+1}, Loss: {avg_val_loss}, Accuracy: {accuracy}, MCC: {mcc}')

        # Save checkpoint
        save_checkpoint(model, epoch, avg_val_loss, checkpoint_path)

        # Update the best model checkpoint if a lower validation loss is achieved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, epoch, best_val_loss, best_model_path)
            print(f"Best model checkpoint updated (epoch {epoch+1}, loss {best_val_loss})")



