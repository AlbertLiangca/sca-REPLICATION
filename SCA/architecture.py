import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

class Sphere(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self,X):
        return X / X.norm(dim=self.dim,keepdim=True)
    def right_inverse(self,X):
        return X / X.norm(dim=self.dim,keepdim=True)

class SCA_encoder(nn.Module):
    '''
    Input: a (T * N) matrix X, of N neurons recorded over T timesteps.
    Output: an affine mapping of X to K-dimensional space.
    '''
    def __init__(
        self,
        N: 100, # number of neurons
        K: 3, # dimensionality
        Q: None # Initalization params
    ):
        super().__init__()
        self.U = nn.Linear(N,K)
        if Q != None:
            Q = nn.Parameter(Q)
            self.U.weight = Q
        nn.init.constant_(self.U.bias,0)

    def forward(self,x):
        encoded_x = self.U(x)
        return encoded_x

class SCA_decoder(nn.Module):
    '''
    Input: affine mapped X in K-dimensional space
    Output: X_hat, or the reconstructed X back in neural activity space
    '''
    def __init__(
        self,
        N: 100, # number of neurons
        K: 3, # dimensionality
    ):
        super().__init__()
        self.V = P.register_parametrization(nn.Linear(K,N),"weight",Sphere(dim=0))
        nn.init.constant_(self.V.bias,0)

    def forward(self, encoded_x):
        x_hat = self.V(encoded_x)
        return x_hat
        
        