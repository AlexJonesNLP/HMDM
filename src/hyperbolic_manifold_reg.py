# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy

def squash(u):
  u = u.cpu().detach().numpy() # Detach from GPU for computations
  if u.all() == 0:
    return u
  res = u * ( np.linalg.norm(u, ord=np.inf) / np.linalg.norm(u) )
  return res.to('cuda:0')

class HyperbolicRegression(nn.Module):
    '''
    Implementation of http://proceedings.mlr.press/v108/marconi20a/marconi20a.pdf 
    '''
    def __init__(self,
                 num_hidden_layers,
                 hidden_size,
                 output_dim,
                 keep_prob):
        super(NeuralNetwork, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.dropout = nn.Dropout(1-self.keep_prob) # we'll apply dropout after each of the hidden layers
        # 1 x 768 * 768 x 20 = 1 x 20
        # 1 x 20 * 20 x 20 = 1 x 20
        # 1 x 20 * 20 x 20 = 1 x 20
        # ...
        # 1 x 20 * 20 x 768 = 1 x 768
        self.Wi = nn.Linear(self.output_dim, self.hidden_size) # input weights
        self.Wo = nn.Linear(self.hidden_size, self.output_dim) # output weights
        self.Wh = nn.ModuleList() # hidden layer weights
        for _ in range(self.num_hidden_layers):
            self.Wh.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x):
        curr = self.Wi(x)
        '''
        Alternate nonlinearity from Marconi et al.; keeps transformed vectors within
        (unit) Poincaré ball
        '''
        curr = squash(np.tanh(curr)) # x' = SQUASH(tanh(LINEAR(x)))
        for i in range(1, len(self.Wh)):
            hidden_layer = self.Wh[i]
            curr = hidden_layer(curr)
            curr = self.dropout(curr)
            curr = squash(np.tanh(curr))
        pred = self.Wo(curr)
        pred = squash(np.tanh(pred))
        return pred
    
    def save(self, model):
        torch.save(model, 'HMR') # reload model with torch.load('HMR')
