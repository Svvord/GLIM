import glim
import numpy as np
import os
import glim.utils as utils
import argparse

parser = argparse.ArgumentParser(
  description='GLIM',
  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--relationship-file', type=str, default='./data/relationship_table.txt')
parser.add_argument('--node-feature-file', type=str, default='./data/node_feature.npy')
parser.add_argument('--embedding-save-file', type=str, default='./results/hmln_feature.npy')

args = parser.parse_args()


def load_graph_network(adj_path, feature_path):
    X, A, Y = [], None, []
    n_node = 0

    # Acquire Edges
    edge_list = []
    node_list = []
    node_type = {}

    with open(adj_path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f.readlines():
            node1, node2, *_ = line.strip().split('\t')
            edge_list.append((node1, node2))
            node_list.extend([node1, node2])
                
    node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
    n_node = len(node_map)
    A = np.zeros((n_node, n_node))
    for node1, node2 in edge_list:
        A[node_map[node1], node_map[node2]] = 1
        A[node_map[node2], node_map[node1]] = 1
    A = np.float32(A)
    
    
    ####################################################
    #            Acquire Features                      #
    ####################################################

    if os.path.exists(feature_path):
        X = np.load(feature_path)
    else:
        X = np.float32(utils.N2V(A, 512, 4, 1))
        np.save(feature_path, X)
    
    return X, A


if __name__ == '__main__': 
    features, adj = load_graph_network(args.relationship_file, args.node_feature_file)
    adj = adj + np.eye(adj.shape[0])
    glim.train.fit_transform(features, adj, args.embedding_save_file, device='cuda')
