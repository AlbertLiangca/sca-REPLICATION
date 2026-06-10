import torch

device = "cpu"

#---Sub-functions---#
def create_W(X):
    '''
    Input: T x N matrix X of T timesteps of N neurons' neural activity
    Output: Diagonal T x T matrix W, where each entry contains the inverse sum-square at that timestep
    '''
    sum_sq_activity = torch.sum(X,axis=1)**2
    inverse = 1/ torch.sqrt(sum_sq_activity + 0.1)
    inverse = inverse / torch.mean(inverse)
    W = torch.diag(inverse)
    return W

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

    '''print(f"recon: {reconstruction_loss}")
    print(f"sparse: {sparsity_loss}")
    print(f"orth: {orth_loss}")'''

    return loss