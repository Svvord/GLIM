import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

import networkx as nx
from node2vec import Node2Vec

def data2tsne(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    tsne = TSNE()
    tsne.fit_transform(embedding)
    return tsne.embedding_

def data2umap(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    embedding_ = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        n_components = 2,
        learning_rate = 1.0,
        spread = 1.0,
        set_op_mix_ratio = 1.0,
        local_connectivity = 1,
        repulsion_strength = 1,
        negative_sample_rate = 5,
        angular_rp_forest = False,
        verbose = False
    ).fit_transform(embedding)
    return embedding_

def umap_plot(data, save_path):
    import seaborn as sns
    plt.figure(figsize=(10,10))
    fig = sns.scatterplot(
        x = 'UMAP_1',
        y = 'UMAP_2',
        data = data,
        hue = 'hue',
        palette="deep"
    )
    fig = plt.gcf()
    fig.savefig(save_path)
    plt.close()
    
def gplot(embedding_, type_info, filename):
    test = pd.DataFrame(embedding_, columns=['UMAP_1', 'UMAP_2'])
    test['hue'] = type_info
    save_path = './pic/'+filename + '.png'
    umap_plot(test, save_path)

def create_plot(features, labels, save_path, style='tsne', n_pca=None):
    if style=='tsne':
        if not n_pca:
            n_pca = 0
        embedding_ = data2tsne(features, n_pca)
    elif style=='umap':
        if not n_pca:
            n_pca = 30
        embedding_ = data2umap(features, n_pca)
    else:
        print(f'No style:{style}!')
        return
    gplot(embedding_, labels, save_path)
    
    
def N2V(adj, hid_units, p=1, q=1, walk_length=20, num_walks=40):
    edge_index = np.where(adj>0)
    edge_index = np.r_[[edge_index[0]], [edge_index[1]]].T
    
    def create_net(elist):
        import networkx as nx
        g = nx.Graph()
        elist = np.array(elist)
        g.add_edges_from(elist)
        for edge in g.edges():
            g[edge[0]][edge[1]]['weight'] = 1
        return g
    
    graph = create_net(edge_index)
    node2vec = Node2Vec(graph, dimensions=hid_units, walk_length=walk_length, num_walks=num_walks, p=p,q=q)
    model = node2vec.fit()
    outputs = np.array([model.wv[str(item)] for item in range(len(adj))])
    return outputs
    
if __name__ == '__main__':
    pass