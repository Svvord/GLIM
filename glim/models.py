import os 
import numpy as np
import math
import random

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv

from . import utils


class Encoder(nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(Encoder, self).__init__()
            self.conv = GCNConv(in_channels, hidden_channels, cached=True)
            self.prelu = nn.PReLU(hidden_channels)

        def forward(self, x, edge_index):
            out = self.conv(x, edge_index)
            out = self.prelu(out)
            return out

class Summary(MessagePassing):
    # aggregation type: 1.mean, 2.max, 3.sum
    def __init__(self, aggr='max'):
        super().__init__(aggr=aggr)

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
        return -torch.log(1/(1 + torch.exp(torch.clamp(neg_loss-pos_loss, max=10)))).mean()
