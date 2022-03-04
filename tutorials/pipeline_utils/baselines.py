import pipeline_utils.dataloader as dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#############################################
#           Packages for GLIM               #
#############################################
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
import math
EPS = 1e-15

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


#############################################
#           Loading Dataset                 #
############################################# 
def load_data(dataset, idx):
    if dataset == 'cora':
        features, adj, labels = dataloader.load_cora()
        nb_classes = 7
    elif dataset == 'citeseer':
        features, adj, labels = dataloader.load_citeseer()
        nb_classes = 6
    elif dataset == 'pubmed':
        features, adj, labels = dataloader.load_pubmed()
        nb_classes = 3
    elif dataset == 'multilayer':
        features, adj, labels = dataloader.load_multilayer()
        nb_classes = 3
    else:
        print(f'{dataset} does not exist!')
    
    if idx:
        idx_train, idx_val, idx_test = idx
    else:
        torch.manual_seed(0)
        random_index = torch.randperm(len(features))
        if dataset == 'cora':
            split1, split2 = 140, 1708
        elif dataset == 'citeseer':
            split1, split2 = 120, 2312
        elif dataset == 'pubmed':
            split1, split2 = 60, 18717
        elif dataset == 'multilayer':
            split1, split2 = 18000, 19291
        elif dataset == 'ppi':
            split1, split2 = 44906, 51420
            
        idx_train = random_index[:split1]
        idx_val = random_index[split1:split2]
        idx_test = random_index[split2:]
        
    return features, adj, labels, idx_train, idx_val, idx_test, nb_classes


#############################################
#             Random Split                  #
#############################################
def random_split(n_sample, dataset):
    random_index = torch.randperm(n_sample)
    if dataset == 'cora':
        split1, split2 = 140, 1708
    elif dataset == 'citeseer':
        split1, split2 = 120, 2312
    elif dataset == 'pubmed':
        split1, split2 = 60, 18717
    elif dataset == 'multilayer':
        split1, split2 = 18000, 19291
    elif dataset == 'ppi':
        split1, split2 = 44906, 51420

    idx_train = random_index[:split1]
    idx_val = random_index[split1:split2]
    idx_test = random_index[split2:]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test





#############################################
#               Test Pipeline               #
#############################################
def glim_pipeline(dataset, idx=None, log_=False):
    features, adj, labels, idx_train, idx_val, idx_test, nb_classes = load_data(dataset, idx)
    
    adj = adj + np.eye(adj.shape[0])
    hid_units = 512
    #edge_index = torch.LongTensor(np.where(adj>0))
    #features = torch.FloatTensor(features)
    
    class Encoder(nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(Encoder, self).__init__()
            self.conv = GCNConv(in_channels, hidden_channels, cached=True)
            self.prelu = nn.PReLU(hidden_channels)

        def forward(self, x, edge_index):
            x = self.conv(x, edge_index)
            out = self.prelu(x)
            return out

    class Summary(MessagePassing):
        def __init__(self):
            super().__init__(aggr='max')

        def forward(self, x, edge_index):
            return self.propagate(edge_index, x=x)

        def message(self, x_j):
            return x_j

    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def uniform(size, tensor):
        if tensor is not None:
            bound = 1.0 / math.sqrt(size)
            tensor.data.uniform_(-bound, bound)

    def reset(nn):
        def _reset(item):
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

        if nn is not None:
            if hasattr(nn, 'children') and len(list(nn.children())) > 0:
                for item in nn.children():
                    _reset(item)
            else:
                _reset(nn)

    class GraphLocalInfomax(torch.nn.Module):
        def __init__(self, hidden_channels, encoder, summary, corruption):
            super(GraphLocalInfomax, self).__init__()
            self.hidden_channels = hidden_channels
            self.encoder = encoder
            self.summary = summary
            self.corruption = corruption
            self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
            self.reset_parameters()

        def reset_parameters(self):
            reset(self.encoder)
            reset(self.summary)
            uniform(self.hidden_channels, self.weight)

        def forward(self, x, edge_index):
            pos_z = self.encoder(x, edge_index)
            cor = self.corruption(x, edge_index)
            cor = cor if isinstance(cor, tuple) else (cor, )
            neg_z = self.encoder(*cor)
            summary = self.summary(pos_z, edge_index)
            return pos_z, neg_z, summary

        def discriminate(self, z, summary, sigmoid=True):
            value = torch.sum(torch.mul(z, torch.matmul(summary, self.weight)), dim=1)
            return value

        def loss(self, pos_z, neg_z, summary):
            pos_loss = self.discriminate(pos_z, summary)
            neg_loss = self.discriminate(neg_z, summary)
            return -torch.log(1/(1 + torch.exp(neg_loss-pos_loss))).mean()
            


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = torch.LongTensor(np.where(adj>0)).to(device)
    features = torch.FloatTensor(features).to(device)
    
    ####       ####
    #     FAST    #
    ####       ####
    feature_size = features.shape[1]
    summary = Summary().to(device)
    encoder = Encoder(feature_size, hid_units).to(device)
    model = GraphLocalInfomax(hid_units, encoder, summary, corruption).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def train():
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(features, edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return loss.item()

    best = 1e9
    patience = 20
    for epoch in range(1, 301):
        loss = train()
        
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
        
        if log_:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))
    with torch.no_grad():
        embeds, _, _ = model(features, edge_index)
        embeds = embeds.unsqueeze(0).detach()
        
        
    labels = torch.LongTensor(labels[np.newaxis]).to(device)
    xent = nn.CrossEntropyLoss()
    
    tot = torch.zeros(1)
    tot = tot.to(device)
    accs = []

    for _ in range(50):
        idx_train, idx_val, idx_test = random_split(len(features), dataset)
        
        train_embs = embeds[0, idx_train]
        val_embs = embeds[0, idx_val]
        test_embs = embeds[0, idx_test]

        train_lbls = labels[0, idx_train]
        val_lbls = labels[0, idx_val]
        test_lbls = labels[0, idx_test]
        
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        if log_:
            print(acc)
        tot += acc
    if log_:
        print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    return accs.mean().item(), accs.std().item()




if __name__ == '__main__':
    pass