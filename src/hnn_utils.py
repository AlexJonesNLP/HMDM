# -*- coding: utf-8 -*-

import numpy as np

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
