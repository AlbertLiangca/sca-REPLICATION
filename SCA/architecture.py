import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
import scipy.io as sio
import itertools

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
        Q: None # Initialization params
    ):
        super().__init__()
        self.V = nn.Linear(K,N)
        if Q != None:
            QT = nn.Parameter(Q.transpose(1,0))
            self.V.weight = QT
        nn.init.constant_(self.V.bias,0)

    def forward(self, encoded_x):
        normalized_V_weight = nn.functional.normalize(self.V.weight.data,dim=1)
        self.V.weight.data = normalized_V_weight
        x_hat = self.V(encoded_x)
        return x_hat
        