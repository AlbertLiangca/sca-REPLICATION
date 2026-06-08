import torch
import torch.nn as nn
import torch.optim as optim
import itertools

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def loss_func(X,W, encoded_X, decoded_X, V_weight, lambda_sparse = 0, lambda_orth = 0):

    # Calculate reconstruction loss
    
    reconstruction_loss = torch.sum((W@(decoded_X-X))**2)

    # Calculate sparsity penalty
    sparsity_loss = lambda_sparse * torch.sum(torch.abs(encoded_X))

    # Calculate orthogonality penalty
    
    VTV = V_weight.T@V_weight
    I = torch.eye(V_weight.shape[1],device=device)
    orth_loss = lambda_orth * torch.norm((VTV - I))**2

    # Sum to calculate loss
    loss = reconstruction_loss + sparsity_loss + orth_loss

    # Backpropagate

    return loss

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

    for t in range(epochs):

        optimizer.zero_grad()

        encoded_X, decoded_X = autoencoder(X)
        V_weight = autoencoder.V.weight

        loss_t = loss_func(X=X,W=W, encoded_X=encoded_X, decoded_X=decoded_X,V_weight=V_weight, lambda_sparse=lambda_sparse, lambda_orth=lambda_orth)
        losses[t+1] = loss_t.item()
        if t % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"loss: {loss_t:>7f}  [{t:>5d}/{epochs:>5d}]")
        if t % 1000 == 0:
            current_lr = lr_scheduler.get_last_lr()[-1]
            print(f"learning rate: {current_lr:>7f}")
        
        loss_t.backward()
        optimizer.step()
        lr_scheduler.step(loss_t.item())
    
    return losses
