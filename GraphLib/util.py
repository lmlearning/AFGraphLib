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
import shutil


#Utility function for parsing TGF
def parseTGF(file):
    f = open(file, 'r')
    
    args = []
    atts = []
    hash_seen = False
    for idx, line in enumerate(f):
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

#Utility function for parsing TGF
def parseAPX(file):
    f = open(file, 'r')
    
    args = []
    atts = []
    
    for idx, line in enumerate(f):
        #print('id={},val={}'.format(idx,line))
        line = line.strip()
        if line.startswith("arg"):
            args.append(line[4:-2])
        elif line.startswith("att"):
            atts.append(line[4:-2].split(','))
    
    return args, atts


def get_features(num_nodes, num_feats):
    return np.random.rand(num_nodes, num_feats)

def read_solution_file(path_to_framework, strip_first_char = True, error_file_mode = False):
    
    f = open(path_to_framework, 'r')
    if error_file_mode:
        f.readline()
    input = f.readline()
    if strip_first_char == True:
        input = input[1:-1]
    #print(input)    
    input = input.replace("]]", "")
    in_arr = input.split('],')
    #print(in_arr)
    sol_arr = [s[1:].split(',') for s in in_arr]
    for arr in sol_arr:
        arr.sort() 
    return sol_arr

def get_coadmissible_labels(G, idx_to_name, solution_file, name_to_idx):
    out_arr = np.zeros((G.number_of_nodes(),G.number_of_nodes()), dtype=np.float32)
    sols = read_solution_file(solution_file)
    for n in G.nodes():
        for arr in sols:
            if idx_to_name[n] in arr:
                for arg in arr:
                    if ']' in arg:
                        arg = arg.replace(']','')
                    out_arr[n, name_to_idx[arg]] = 1.0 
                #out_arr[n] = [1.0 if idx_to_name[idx] in arr else 0.0 for idx in range(G.number_of_nodes()) ]
    
    return out_arr

def get_credulous_labels(G, idx_to_name, solution_file, name_to_idx = None,  error_file_mode=False):
    out_arr = np.zeros((G.number_of_nodes(),), dtype=np.float32)
    sols = read_solution_file(solution_file, True, error_file_mode)
    for n in G.nodes():
        for arr in sols:
            if idx_to_name[n] in arr:
                out_arr[n] = 1.0
    
    return out_arr

def get_SE_labels(G, idx_to_name, solution_file, name_to_idx = None,  error_file_mode=False):
    out_arr = np.zeros((G.number_of_nodes(),), dtype=np.float32)
    sols = read_solution_file(solution_file, False,  error_file_mode)
    for n in G.nodes():
        for arr in sols:
            if idx_to_name[n] in arr:
                out_arr[n] = 1.0
    
    return out_arr


def get_sceptical_labels(G, idx_to_name, solution_file, name_to_idx = None,  error_file_mode=False):
    out_arr = np.zeros((G.number_of_nodes(),), dtype=np.float32)
    sols = read_solution_file(solution_file, True, error_file_mode)
    
    for n in G.nodes():
        sceptically_accepted = True
        for arr in sols:
            if not idx_to_name[n] in arr:
                sceptically_accepted = False
        if sceptically_accepted == True:
            out_arr[n] = 1.0
    
    return out_arr

def print_extension(labels, idx_to_name):
    name_arr = []
    for idx, val in enumerate(labels):
        if val == 1.0:
            name_arr.append(idx_to_name[idx])
        
    print("[" + ','.join(name_arr) + "]")

def get_masks(num_nodes, train_ratio, val_ratio, test_ratio):
    rng = np.random.RandomState(42)
    ntrain = int(num_nodes * train_ratio)
    nval = int(num_nodes * val_ratio)
    ntest = int(num_nodes * test_ratio)
    mask_array = np.zeros((num_nodes,), dtype=np.int32)
    mask_array[0:ntrain] = 1
    mask_array[ntrain:ntrain+nval] = 2
    mask_array[ntrain+nval:ntrain+nval+ntest] = 3
    rng.shuffle(mask_array)
    train_mask = (mask_array == 1).astype(np.int32)
    val_mask = (mask_array == 2).astype(np.int32)
    test_mask = (mask_array == 3).astype(np.int32)
    return train_mask, val_mask, test_mask


def getRandomBatch(sample_size, g_arr, balance = False, label_arr = None, exclude_empty_AFs = False, sel_idx = []):
    g_arr = g_arr
    if label_arr == None:
        label_arr = []
        half_len = len(g_arr.nodes) // 2
        if len(g_arr.nodes) % 2 == 0:
            label_arr.extend([1 for i in range(0,half_len + 1)])
        else:
            label_arr.extend([1 for i in range(0,half_len)])
        
        label_arr.extend([0 for i in range(0,half_len)])
        
        
    sample = np.random.choice(len(g_arr), sample_size, False)
    #print(sample)
    training_mask = []
    valid_mask = []
    i = 0
    cur_idx = 0
    
    for graph in g_arr:
        breakpoint = int(len(graph.nodes) * 0.8)
        
        if len(sel_idx) > 0:
            num_yes_train = int(0.8 * sum(label_arr[i][sel_idx[i]]))
            num_yes_valid = int(0.2 * sum(label_arr[i][sel_idx[i]]))
            num_no_train  = breakpoint - num_yes_train
            num_no_valid  = (len(graph.nodes) - breakpoint) - num_yes_valid
        else:
            num_yes_train = int(0.8 * sum(label_arr[i]))
            num_yes_valid = int(0.2 * sum(label_arr[i]))
            num_no_train  = breakpoint - num_yes_train
            num_no_valid  = (len(graph.nodes) - breakpoint) - num_yes_valid
        
        yes_or_no_is_highest = "y"
        balance_number_train = abs(num_yes_train - num_no_train)
        balance_number_valid = abs(num_yes_valid - num_no_valid)
        if num_yes_train < num_no_train:
            yes_or_no_is_highest = "n"
        
        #print(len(graph.nodes), num_yes_train, num_yes_valid, num_no_train, num_no_valid)
        
        cur_t_mask = []
        cur_v_mask = []
        if (not i in sample) or (exclude_empty_AFs == True and num_yes_train == 0):
            cur_t_mask.extend([0 for q in range (0,len(graph.nodes))])
            cur_v_mask.extend([0 for q in range (0,len(graph.nodes))])        
        else:
            for q in range (0,len(graph.nodes)):
                if np.random.rand() < 0.2:
                    cur_v_mask.append(1)
                    cur_t_mask.append(0)
                else:
                    cur_v_mask.append(0)
                    cur_t_mask.append(1)
            #cur_t_mask.extend([1 for q in range (0,breakpoint)])
            #cur_t_mask.extend([0 for q in range (breakpoint,len(graph.nodes))])            
            #cur_v_mask.extend([0 for q in range (0,breakpoint)])
            #cur_v_mask.extend([1 for q in range (breakpoint,len(graph.nodes))])
            
        
        if len(sel_idx) > 0:
            if balance == True:
                balancing_factor = 0
                for j in range(0,breakpoint):
                    #print(label_arr[i][j])
                    if yes_or_no_is_highest == "y":
                        if label_arr[i][sel_idx[i]][j] == 1 and balancing_factor < balance_number_train:
                            balancing_factor += 1
                            cur_t_mask[j] = 0
                    elif yes_or_no_is_highest == "n":
                        if label_arr[i][sel_idx[i]][j] == 0 and balancing_factor < balance_number_train:
                            balancing_factor += 1
                            cur_t_mask[j] = 0

                balancing_factor = 0
                for j in range(breakpoint,len(graph.nodes)):
                    if yes_or_no_is_highest == "y":
                        if label_arr[i][sel_idx[i]][j] == 1 and balancing_factor < balance_number_valid:
                            balancing_factor += 1
                            cur_v_mask[j] = 0
                    elif yes_or_no_is_highest == "n":
                        if label_arr[i][sel_idx[i]][j] == 0 and balancing_factor < balance_number_valid:
                            balancing_factor += 1
                            cur_v_mask[j] = 0
        else:
            if balance == True:
                balancing_factor = 0
                for j in range(0,breakpoint):
                    #print(label_arr[i][j])
                    if yes_or_no_is_highest == "y":
                        if label_arr[i][j] == 1 and balancing_factor < balance_number_train:
                            balancing_factor += 1
                            cur_t_mask[j] = 0
                    elif yes_or_no_is_highest == "n":
                        if label_arr[i][j] == 0 and balancing_factor < balance_number_train:
                            balancing_factor += 1
                            cur_t_mask[j] = 0

                balancing_factor = 0
                for j in range(breakpoint,len(graph.nodes)):
                    if yes_or_no_is_highest == "y":
                        if label_arr[i][j] == 1 and balancing_factor < balance_number_valid:
                            balancing_factor += 1
                            cur_v_mask[j] = 0
                    elif yes_or_no_is_highest == "n":
                        if label_arr[i][j] == 0 and balancing_factor < balance_number_valid:
                            balancing_factor += 1
                            cur_v_mask[j] = 0


        training_mask.extend(cur_t_mask)
        valid_mask.extend(cur_v_mask)
        
        i += 1
        
    return training_mask, valid_mask

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    th.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = th.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
        
