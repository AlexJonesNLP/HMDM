import numpy as np
from typing import List, Tuple

def poincare_dist(u: np.ndarray, v: np.ndarray) -> float:
    '''
    Compute the distance between two vectors u and v on the Poincaré ball
    '''
    num = 2 * (np.linalg.norm(u-v))**2
    denom = (1-np.linalg.norm(u)**2) * (1-np.linalg.norm(v)**2)
    res = np.arccosh(1 + num/denom)
    return res

def poincare_norm(u: np.ndarray) -> float:
    '''
    Compute the Poincaré norm of n-dimensional vector u
    '''
    return 2 * np.arctanh(np.linalg.norm(u))

def project(u: np.ndarray) -> np.ndarray:
    '''
    Project embedding to guarantee norm is strictly less than 1 (and thus
    within the Poincaré ball)
    '''
    eps = 1e-5
    if np.linalg.norm(u) >= 1:
        return u / (np.linalg.norm(u)-eps)
    return u

def poincare_softrank_loss(relation_vecs: List[Tuple[np.ndarray]]) -> float:
    '''
    Compute the soft-ranking loss defined in https://arxiv.org/pdf/1705.08039.pdf
    over embeddings representing hypernymy relations
    '''
    # Compute numerator term in log-sum
    num = 1
    for relation_pair in relation_vecs:
        num *= np.exp(-poincare_dist(relation_pair[0], relation_pair[1]))
    num = np.log(num)
    # Compute denominator term (nested log-sum)
    denom = 1
    negative_dict = {}
    for i in range(len(relation_vecs)):
        for j in range(len(relation_vecs)):
            hyponym, hypernym = relation_vecs[i][0], relation_vecs[j][1]
        if hyponym not in negative_dict:
            if (hyponym, hypernym) not in relation_vecs:
                negative_dict[hyponym] = [hypernym]
        else:
            if (hyponym, hypernym) not in relation_vecs:
                negative_dict[hyponym].append(hypernym)
    for hyponym in negative_dict:
        sum_ = sum[np.exp(-poincare_dist(hyponym, hypernym))
                   or hypernym in negative_dict[hyponym]]
        denom *= sum_
    denom = np.log(denom)
    loss = num - denom
    return loss

def mobius_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    '''
    Perform Möbius addition between vectors u and v on the Poincaré ball
    See https://arxiv.org/pdf/1911.02536.pdf for reference
    '''
    num_comp1 = ( 1 + 2 * u.dot(v) + np.linalg.norm(v)**2 ) * u
    num_comp2 = ( 1 - np.linalg.norm(u)**2) * v
    num = num_comp1 + num_comp2
    denom = 1 + 2 * u.dot(v) + np.linalg.norm(u)**2 * np.linalg.norm(v)**2
    res = num / denom
    return res

def exponential_map(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    '''
    Exponential map between the manifold X containing u and the tangent space of X at
    point p
    See https://arxiv.org/pdf/1911.02536.pdf for reference
    '''
    p_conformal_factor = 1 / (1 - np.linalg.norm(p)**2)
    sum_term = np.tanh(0.5 * p_conformal_factor * np.linalg.norm(u))
    sum_term *= u / np.linalg.norm(u)
    res = mobius_add(p, sum_term)
    return res

def log_map(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    '''
    Logarithmic map between the manifold X containing u and the tangent space of X at
    point p
    See https://arxiv.org/pdf/1911.02536.pdf for reference
    '''
    p_conformal_factor = 1 / (1 - np.linalg.norm(p)**2)
    mobius_term = mobius_add(-p, u)
    res_term1 = np.arctanh(np.linalg.norm(mobius_term))
    res_term2 = mobius_term / np.linalg.norm(mobius_term)
    res = (2 / p_conformal_factor) * res_term1 * res_term2
    return res


