import torch
import torch.nn as nn
import sys

from architecture import SCA_autoencoder
from training import training_loop

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def SCA(X,K,epochs = 3000):

    #---Sub-functions---#
    def create_W(X):
        '''
        Input: T x N matrix X of T timesteps of N neurons' neural activity
        Output: Diagonal T x T matrix W, where each entry contains the inverse sum-square at that timestep
        '''
        sum_sq_activity = torch.sum(X,axis=1)**2
        inverse = 1/ sum_sq_activity
        W = torch.diag(inverse)
        return W
    
    # Extract T and N from data tensor
    T,N = X.shape
    
    #---Determining Initialization Parameters and Hyperparameters---#

    # Sample_weight
    W = create_W(X)

    # Find Vh
    U,S,Vh = torch.linalg.svd(W@X)

    # Find Q
    QT = Vh[0:K]
    Q = QT.T

    # UV = QQT
    QQT = Q@QT

    # Calculate initialized reconstruction cost
    init_reconstruction = torch.sum((X - (X@QQT))**2)

    # Calculate initialized orthogonality cost if off-diagonal entries of VVT = 0.1
    # Since V is orthonormal, VVT - I is 0.1 everywhere except for diagonal, which is 0
    init_orth = torch.tensor(0.1).expand(K,K)-torch.eye(K)*0.1
    init_orth = torch.norm(init_orth)**2

    # Calculate initialized sparsity cost
    init_sparse = torch.sum(torch.abs(X@Q))

    # Algebraically rearrange to determine initalization lambdas
    lambda_init_orth = 0.1*init_reconstruction / init_orth
    lambda_init_sparse = 0.1*init_reconstruction / init_sparse

    #---Initializing the autoencoders---#
    
    #QT = torch.randn(QT.shape)
    autoencoder = SCA_autoencoder(N=N,K=K,Q=QT).to(device)

    #---Train the autoencoders---#

    autoencoder.train()
    losses = training_loop(X=X,W=W,autoencoder = autoencoder,lambda_sparse = lambda_init_sparse,lambda_orth=lambda_init_orth,epochs=epochs)

    #---Extract trained encoder-decoder state_dict---#
    autoencoder_state_dict = autoencoder.state_dict()

    #print(torch.allclose(before_dict['U.weight'],encoder_state_dict['U.weight']))
    latent, out = autoencoder(X.to(device))

    return {
        'autoencoder_state_dict':autoencoder_state_dict,
        'losses':losses,
        'latent':latent
    }


def main():
    X0 = torch.rand((15,10))
    out = SCA(X0,2)
    return out

if __name__ == "__main__":
    sys.exit(main())