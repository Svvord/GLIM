from . import models
import torch
import torch.nn as nn
import numpy as np

def fit_transform(features, adj, save_path, n_hid=512, device='cpu', print_log=True, patience=20, n_epoch=300, tmp_path='best_dgi.pkl'):
    device = torch.device(device)
    
    # Self loop is required.
    edge_index = torch.LongTensor(np.where(adj>0)).to(device)
    # Features_shape = n_sample * n_dim
    features = torch.FloatTensor(features).to(device)
    
    hid_units = n_hid
    feature_size = features.shape[1]
    
    summary = models.Summary('max').to(device)
    encoder = models.Encoder(feature_size, hid_units).to(device)
    corruption = models.corruption
    model = models.GraphLocalInfomax(hid_units, encoder, summary, corruption).to(device)
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
    for epoch in range(1, n_epoch+1):
        loss = train()
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), tmp_path)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        if print_log:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
            
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(tmp_path))
    
    with torch.no_grad():
        embeds, _, _ = model(features, edge_index)
        embeds = embeds.detach().cpu().numpy()
    np.save(save_path, embeds)
    
    return embeds