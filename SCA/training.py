import torch
import torch.nn as nn
import torch.optim as optim
import itertools

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def train(X,W, encoder, decoder, optimizer, lambda_sparse = 0, lambda_orth = 0):

    X = X.to(device)
    W = W.to(device)

    # Calculate reconstruction loss
    encoded_X = encoder(X)
    decoded_X = decoder(encoded_X)
    reconstruction_loss = torch.sum((W@(X - decoded_X))**2)

    # Calculate sparsity penalty
    sparsity_loss = lambda_sparse * torch.sum(torch.abs(encoded_X))

    # Calculate orthogonality penalty
    V_weight = decoder.V.weight.data
    VTV = torch.matmul(V_weight.transpose(1,0),V_weight)
    I = torch.eye(VTV.size(dim=0),device=device)
    orth_loss = lambda_orth * torch.norm((VTV - I),p='fro')**2

    # Sum to calculate loss
    loss = reconstruction_loss + sparsity_loss + orth_loss

    # Backpropagate

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def training_loop(X,W, encoder, decoder, lambda_sparse = 0, lambda_orth = 0, epochs=100):

    lr = 1e-3
    min_lr = 5e-4
    plateau_len = 100
    factor = 0.5
    threshold = 1e-6
    threshold_mode = 'rel'

    all_params = itertools.chain(encoder.parameters(), decoder.parameters())
    optimizer = torch.optim.Adam(all_params,lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=plateau_len, min_lr=min_lr,threshold=threshold,threshold_mode=threshold_mode)
    
    losses = torch.zeros(epochs)
    for t in range(epochs):
        loss_t = train(X=X,W=W, encoder=encoder, decoder=decoder, optimizer=optimizer, lambda_sparse=lambda_sparse, lambda_orth=lambda_orth)
        losses[t] = loss_t
        if t % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"loss: {loss_t:>7f}  [{t:>5d}/{epochs:>5d}]")
        if t % 1000 == 0:
            current_lr = lr_scheduler.get_last_lr()[-1]
            print(f"learning rate: {current_lr:>7f}")
        lr_scheduler.step(loss_t)
    
    return losses
