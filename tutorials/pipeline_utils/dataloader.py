import numpy as np
import os


####################################################
#                  Cora Dataset                    #
####################################################
def load_cora(filepath='../data/public-data/cora/'):
    X, A, Y = [], None, []
    n_node = 0
    
    ####################################################
    #            Acquire Features                      #
    ####################################################
    node_map = {}
    feature_path = filepath + 'cora.content'
    with open(feature_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            node, *vector, label = line.strip().split()
            node = int(node)
            vector = [int(item) for item in vector]
            
            node_map[node] = n_node
            n_node += 1
            Y.append(label)
            X.append(vector)
    X = np.float32(np.array(X))
    
    ####################################################
    #            Acquire Edges                         #
    ####################################################
    A = np.zeros((n_node, n_node))
    adj_path = filepath + 'cora.cites'
    with open(adj_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            node1, node2 = [int(item) for item in line.strip().split()]
            A[node_map[node1], node_map[node2]] = 1
            A[node_map[node2], node_map[node1]] = 1
    A = np.float32(A)
    
    ####################################################
    #            Acquire Labels                        #
    ####################################################
    label_map = {item: i for i, item in enumerate(sorted(set(Y)))}
    Y = np.array([label_map[item] for item in Y])
    
    return X, A, Y


####################################################
#              Citeseer Dataset                    #
####################################################
def load_citeseer(filepath='../data/public-data/citeseer/'):
    X, A, Y = [], None, []
    n_node = 0
    
    ####################################################
    #            Acquire Features                      #
    ####################################################
    node_map = {}
    feature_path = filepath + 'citeseer.content'
    with open(feature_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            node, *vector, label = line.strip().split()
            node = str(node)
            vector = [int(item) for item in vector]
            
            node_map[node] = n_node
            n_node += 1
            Y.append(label)
            X.append(vector)
    X = np.float32(np.array(X))
    
    ####################################################
    #            Acquire Edges                         #
    ####################################################
    A = np.zeros((n_node, n_node))
    adj_path = filepath + 'citeseer.cites'
    with open(adj_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            node1, node2 = [str(item) for item in line.strip().split()]
            if node1 not in node_map or node2 not in node_map:
                continue
            A[node_map[node1], node_map[node2]] = 1
            A[node_map[node2], node_map[node1]] = 1
    A = np.float32(A)
    
    ####################################################
    #            Acquire Labels                        #
    ####################################################
    label_map = {item: i for i, item in enumerate(sorted(set(Y)))}
    Y = np.array([label_map[item] for item in Y])
    
    return X, A, Y


####################################################
#                  PubMed Dataset                  #
####################################################
def load_pubmed(filepath='../data/public-data/Pubmed-Diabetes/data/'):
    X, A, Y = [], None, []
    n_node = 0
    
    ####################################################
    #            Acquire Features                      #
    ####################################################
    node_map = {}
    feature_path = filepath + 'Pubmed-Diabetes.NODE.paper.tab'
    with open(feature_path, 'rt', encoding='utf-8') as f:
        next(f)
        line = next(f)
        feature_map = {item.split(':')[1]:i for i, item in enumerate(line.strip().split()[1:-1])}
        i=1
        for line in f.readlines():
            node, label, *vector = line.strip().split()
            node = str(node)
            row_v = np.zeros(500)
            for item in vector[:-1]:
                key, value = item.split('=')
                value = float(value)
                row_v[feature_map[key]] = value
            node_map[node] = n_node
            n_node += 1
            Y.append(label)
            X.append(row_v)
    X = np.float32(np.array(X))
    
    ####################################################
    #            Acquire Edges                         #
    ####################################################
    A = np.zeros((n_node, n_node))
    adj_path = filepath + 'Pubmed-Diabetes.DIRECTED.cites.tab'
    with open(adj_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines()[2:]:
            _, node1, _, node2 = line.strip().split()
            node1 = node1.split(':')[-1]
            node2 = node2.split(':')[-1]
            A[node_map[node1], node_map[node2]] = 1
            A[node_map[node2], node_map[node1]] = 1
    A = np.float32(A)
    
    ####################################################
    #            Acquire Labels                        #
    ####################################################
    label_map = {item: i for i, item in enumerate(sorted(set(Y)))}
    Y = np.array([label_map[item] for item in Y])
    
    return X, A, Y

if __name__ == '__main__':
    pass