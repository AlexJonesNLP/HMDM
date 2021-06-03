# -*- coding: utf-8 -*-

import torch.nn
import numpy as np
from geomloss import SamplesLoss
from geoopt.optim import RiemannianAdam
import math
from tqdm import tqdm
from hnn_utils import *

class HyperbolicLayer(torch.nn.Module):
  '''
  Custom hyperbolic (Möbius) layer
  '''
  def __init__(self, size_in, size_out):
      super().__init__()
      self.size_in, self.size_out = size_in, size_out
      weights = torch.Tensor(size_out, size_in)
      self.weights = torch.nn.Parameter(weights)
      bias = torch.Tensor(size_out)
      self.bias = torch.nn.Parameter(bias)

      # initialize weights and biases
      torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
      fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
      bound = 1 / math.sqrt(fan_in)
      torch.nn.init.uniform_(self.bias, -bound, bound)  # bias init

  def forward(self, X):
    '''
    f_HypL := exp(0, W.dot(log(0, X))) (+) b
    See https://arxiv.org/pdf/1911.02536.pdf, https://arxiv.org/pdf/1805.09112.pdf
    '''
    self.logx = log_map(X[0, :], torch.zeros(X.shape[1]).to('cuda:0'))
    self.exp_Wx = exponential_map(torch.mul(self.weights.t()[0, :], self.logx.to('cuda:0')), torch.zeros(X.shape[1]).to('cuda:0'))
    self.z = mobius_add(self.exp_Wx, self.bias)
    return torch.reshape(self.z, (1, self.z.shape[0]))

class HyperbolicNN(torch.nn.Module):
  def __init__(self,
               num_hidden_layers, 
               hidden_size,
               output_dim):
    super(HyperbolicNN, self).__init__()
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.output_dim = output_dim
    self.Wi = HyperbolicLayer(1, self.hidden_size) # input weights
    self.Wo = HyperbolicLayer(self.hidden_size, self.output_dim) # output weights
    self.Wh = torch.nn.ModuleList() # hidden layer weights
    for k in range(self.num_hidden_layers):
      self.Wh.append(HyperbolicLayer(self.hidden_size, self.hidden_size))
  
  def save(self, model):
    torch.save(model, 'HyperNN') # reload model with torch.load('HyperNN')

  def predict(self, x):
    self.x = x
    curr = self.Wi(self.x)
    for hidden_layer in self.Wh:
      curr = hidden_layer(curr)
    pred = self.Wo(curr)
    return pred

'''
Utils for hyperbolic arithmetic operations
'''

def mobius_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    '''
    Perform Möbius addition between vectors u and v on the Poincaré ball
    See https://arxiv.org/pdf/1911.02536.pdf for reference
    '''
    num_comp1 = ( 1 + 2 * u.dot(v) + np.linalg.norm(v.cpu().detach().numpy())**2 ) * u
    num_comp2 = ( 1 - np.linalg.norm(u.cpu().detach().numpy())**2) * v
    num = num_comp1 + num_comp2
    denom = 1 + 2 * u.dot(v) + np.linalg.norm(u.cpu().detach().numpy())**2 * np.linalg.norm(v.cpu().detach().numpy())**2
    res = num / denom
    return res

def exponential_map(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    '''
    Exponential map between the manifold X containing u and the tangent space of X at
    point p
    See https://arxiv.org/pdf/1911.02536.pdf for reference
    '''
    EPS = 1e-5
    p += EPS
    p_conformal_factor = 1 / (1 - np.linalg.norm(p.cpu())**2)
    sum_term = np.tanh(0.5 * p_conformal_factor * np.linalg.norm(u.cpu().detach().numpy()))
    sum_term = sum_term * (u / np.linalg.norm(u.cpu().detach().numpy()))
    res = mobius_add(p, sum_term)
    return res

def log_map(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    '''
    Logarithmic map between the manifold X containing u and the tangent space of X at
    point p
    See https://arxiv.org/pdf/1911.02536.pdf for reference
    '''
    EPS = 5e-1
    p += EPS
    p_conformal_factor = 1 / (1 - np.linalg.norm(p.cpu().detach().numpy())**2)
    p = p.to('cuda:0')
    u = u.to('cuda:0')
    mobius_term = mobius_add(-p, u)
    res_term1 = np.arctanh(np.linalg.norm(mobius_term.cpu().detach().numpy())-EPS)
    res_term2 = mobius_term / np.linalg.norm(mobius_term.cpu().detach().numpy())
    res = (2 / p_conformal_factor) * res_term1 * res_term2
    return res

