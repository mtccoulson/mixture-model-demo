 

import numpy as np
import pandas as pd
from sklearn import cluster, datasets, mixture
from plotnine import *


class gaussian_distribution():
    def __init__(self, p_dict):
        self.covariance = p_dict['covariance']
        self.mean = p_dict['mean']
        self.dim = p_dict['dim']
    
    def evaluate_prob(self, x):
        x_m = x - self.mean
        norm = (1 / np.sqrt((2 * np.pi)) ** self.dim * np.linalg.det(self.covariance))
        
        if len(x.shape) == 1 | x.shape[0] == 1:
            out = np.empty(1)
        else:
            out = np.empty(x.shape[0])
        
        for i in range(len(out)):
            exponent = - 0.5 * np.linalg.solve(self.covariance, x_m[i]).T.dot(x_m[i])
            out[i] = norm * np.exp(exponent)
        return out
    
    def update_params(self, p_update):
        self.covariance = p_update['covariance']
        self.mean = p_update['mean']



class gaussian_mixture_model():
    def __init__(self, n_mixture, g_init = None, dim = None):
        self.n_mixture = n_mixture
        self.mixtures = self.gaussian_distribution_init(g_init, dim)
        self.dim = self.dim_init(g_init, dim)
        self.mixing_param = np.ones(self.n_mixture) / self.n_mixture    
    
    def gaussian_distribution_init(self, g_init, dim):
        mixtures = []
        if g_init:
            for _, p_init in enumerate(g_init):
                mixtures.append(gaussian_distribution(p_init))
        else:
            for _ in range(self.n_mixture):
                mixtures.append(gaussian_distribution({'mean' : np.random.uniform(0, 1, dim),
                                                       'covariance' : np.identity(dim),
                                                       'dim' : dim}))
        return mixtures
    
    def dim_init(self, g_init, dim):
        if g_init:
            dim = g_init['dim']
        else:
            dim = dim
        return dim
        
    def e_step(self, data):
        w = []
        for gaussian in self.mixtures:
            w_j = gaussian.evaluate_prob(x = data)
            w.append(w_j)
        w = w / np.sum(w, axis = 0)
        return w
    
    def m_step(self, data, w):
        for i in range(self.n_mixture):
            #Mixing parameter update
            self.mixing_param[i] = np.sum(w[:, i]) / len(data)
            #Gaussian parameter updates
            self.mixtures[i].mean = np.sum(w[:, i] * data, axis = 1) / np.sum(w[:, i])
            self.mixtures[i].covariance =  np.matmul((w[:, i] * (data - self.mixtures[i].mean)).T, (data - self.mixtures[i].mean)) / np.sum(w[:, i])

# =============================================================================
#     def fit(self, data, tol = 0.01):
#         
#         diff_ll = tol + 1
#         prev_ll = 0
#         while diff_ll > tol:
#             #Step paramaters
#             w = self.e_step(self, data)
#             self.m_step(self, data, w)
#             
#             #Calculate log likelihood
#             ll = self.log_likelihood_eval(self, data)
#             print(ll)
#             
#             #Check if convergence criteria has been reached
#             diff_ll = abs(ll - prev_ll)
#             prev_ll = ll
# =============================================================================
        
    def fit(self, data, n_runs = 10):
        for i in range(n_runs):
            print(i)
            w = self.e_step(data = data)
            self.m_step(data = data, w = w)
    
    
if __name__ == '__main__':
    dim = 2
    distrib_test = gaussian_distribution({'mean' : np.random.uniform(0, 1, dim),
                                          'covariance' : np.identity(dim),
                                          'dim' : dim})
    test_point = np.array((0.5, 0.5))
    distrib_test.evaluate_prob(test_point)

    
    #Test the mixture model fitting to blobs
    n_samples = 1500
    blobs = datasets.make_blobs(n_samples, cluster_std=[1.0, 2.5, 0.5])[0]
    
    GMM = gaussian_mixture_model(n_mixture = 3, dim = 2)
    GMM.fit(data = blobs)
    