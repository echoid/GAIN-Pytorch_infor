import torch
import numpy as np
# from tqdm import tqdm
from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import sys


from scipy import stats
from scipy.stats import norm
import numpy as np

def introduce_mising(X):
    N, D = X.shape
    Xnan = X.copy()

    mask = np.ones(X.shape)
    # ---- MNAR in D/2 dimensions
    q2 = np.median(Xnan[:, :int(D)], axis=0)
    print(q2)
    
    ix_larger_than_mean = 1- (Xnan[:, :int(D)] > q2)

    mask[:, :int(D)] = ix_larger_than_mean

    return mask


# create the data distribution

#compute KL Divergence
"""KL Divergence(P|Q)"""
def KL_div(p_probs, q_probs):    
    KL_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(KL_div)
def JS_Div(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (KL_div(p, m) + KL_div(q, m)) / 2



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)
    
# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C



def calculate_linfo(fake_data, real_data):
    # Calculate column means and standard deviations for fake and real data
    # Calculate column means and standard deviations for fake and real data
    fake_means = torch.mean(fake_data, dim=0)
    fake_stds = torch.std(fake_data, dim=0)
    
    real_means = torch.mean(real_data, dim=0)
    real_stds = torch.std(real_data, dim=0)
    
    # Calculate L2 norms for column means and column stds
    l2_norm_means = torch.norm(fake_means - real_means, p=2)
    l2_norm_stds = torch.norm(fake_stds - real_stds, p=2)
    
    # Calculate Linfo score
    Lmean = torch.norm(fake_means - real_means, float('inf'))
    Lsd = torch.norm(fake_stds - real_stds, float('inf'))
    
    Linfo = max(0, Lmean - 0) + max(0, Lsd - 0)
    
    return Linfo




#%% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])        

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx
