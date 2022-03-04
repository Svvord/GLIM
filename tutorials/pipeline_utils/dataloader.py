import numpy as np
import os

####################################################
#                   PPI Dataset                    #
####################################################
def load_ppi(filepath='/home/mld20/Project/LDGI/public-data/PPI/'):
    import json
    X, A, Y = [], None, []
    n_node = 0
    
    with open(filepath + 'ppi-id_map.json', encoding='utf-8') as f:
        node_map = json.load(f)
    n_node = len(node_map)
    
    feature_path = filepath + 'ppi-feats.npy'
    X = np.load(feature_path)
    X = np.float32(X)
    
    g_path = filepath + 'ppi-G.json'
    A = np.zeros((n_node, n_node), np.float32)
    with open(g_path, encoding='utf-8') as f:
        ppi_G = json.load(f)
    for item in ppi_G['links']:
        A[item['source'], item['target']] = 1
    
    label_path = filepath + 'ppi-class_map.json'
    with open(label_path, encoding='utf-8') as f:
        ppi_label = json.load(f)
    for i in range(n_node):
        Y.append(ppi_label[str(i)])
    Y = np.float32(np.array(Y))
    
    return X, A, Y

####################################################
#            MultiLayer Dataset                    #
####################################################
def load_multilayer(filepath='/home/mld20/Project/LDGI/data/'):
    X, A, Y = [], None, []
    n_node = 0

    # Acquire Edges
    edge_list = []
    node_list = []
    node_type = {}
    adj_path = filepath + 'relationship_table.txt'
    with open(adj_path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f.readlines():
            node1, node2, type1, type2, _ = line.strip().split('\t')
            edge_list.append((node1, node2))
            node_list.extend([node1, node2])
            if len(type1) != 1:
                type1 = 'c'
            if len(type2) != 1:
                type2 = 'c'
            node_type[node1] = type1
            node_type[node2] = type2
                
    node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
    n_node = len(node_map)
    A = np.zeros((n_node, n_node))
    for node1, node2 in edge_list:
        A[node_map[node1], node_map[node2]] = 1
        A[node_map[node2], node_map[node1]] = 1
    A = np.float32(A)
    
    ####################################################
    #            Acquire Labels                        #
    ####################################################
    Y = [node_type[item] for item in sorted(list(set(node_list)))]
    label_map = {item: i for i, item in enumerate(sorted(set(Y)))}
    Y = np.array([label_map[item] for item in Y])
    
    
    ####################################################
    #            Acquire Features                      #
    ####################################################
    from ldgi import utils
    feature_path = filepath + 'relationship_n2v.npy'
    if os.path.exists(feature_path):
        X = np.load(feature_path)
    else:
        X = np.float32(utils.N2V(A, 512, 4, 1))
        np.save(feature_path, X)
    
    return X, A, Y


####################################################
#                  Cora Dataset                    #
####################################################
def load_cora(filepath='/home/mld20/Project/LDGI/public-data/cora/'):
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
def load_citeseer(filepath='/home/mld20/Project/LDGI/public-data/citeseer/'):
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
def load_pubmed(filepath='/home/mld20/Project/LDGI/public-data/Pubmed-Diabetes/data/'):
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