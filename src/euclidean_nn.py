#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn

class NeuralNetwork(nn.Module):
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
        '''
        Now dealing with nonlinearities:
        
        We need a nonlinearity after the input layer and a nonlinearity after
        each hidden layer, including after the penultimate layer
        
        We'll use the ReLU activation function as our nonlinearity
        '''
        self.nonlinearities = [nn.ReLU() for _ in range(num_hidden_layers+1)]
    
    def forward(self, x):
        curr = self.Wi(x)
        curr = self.nonlinearities[0](curr)
        for i in range(1, len(self.Wh)):
            hidden_layer = self.Wh[i]
            curr = hidden_layer(curr)
            curr = self.dropout(curr)
            curr = self.nonlinearities[i](curr)
        pred = self.Wo(curr)
        return pred
    
    def save(self, model):
        torch.save(model, 'NN') # reload model with torch.load('NN')
