import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from tqdm import tqdm

from util import loss_func

device = "cpu"

def training_loop(X,W, autoencoder, lambda_sparse = 0, lambda_orth = 0, epochs=100):

    X = X.to(device)
    W = W.to(device)

    lr = 1e-3
    min_lr = 5e-4
    plateau_len = 100
    factor = 0.5
    threshold = 1e-6
    threshold_mode = 'rel'

    optimizer = torch.optim.Adam(autoencoder.parameters(),lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=plateau_len, min_lr=min_lr,threshold=threshold,threshold_mode=threshold_mode)

    losses = torch.zeros(epochs+1)
    autoencoder.eval()
    
    encoded_X, decoded_X = autoencoder(X)
    V_weight = autoencoder.V.weight

    before_loss = loss_func(X=X,W=W, encoded_X=encoded_X, decoded_X=decoded_X,V_weight=V_weight, lambda_sparse=lambda_sparse, lambda_orth=lambda_orth)

    losses[0] = before_loss

    autoencoder.train()

    for t in tqdm(range(epochs),position=0,leave=True):

        optimizer.zero_grad()

        encoded_X, decoded_X = autoencoder(X)
        V_weight = autoencoder.V.weight

        loss_t = loss_func(X=X,W=W, encoded_X=encoded_X, decoded_X=decoded_X,V_weight=V_weight, lambda_sparse=lambda_sparse, lambda_orth=lambda_orth)
        losses[t+1] = loss_t.item()
        
        loss_t.backward()
        optimizer.step()
        lr_scheduler.step(loss_t.item())
    
    return losses
