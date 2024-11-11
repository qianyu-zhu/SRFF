#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to calculate the upper bound of the expected squared error of the SR-OMC kernel approximation.
"""

import sys
import numpy as np
from tqdm import tqdm
from math import gamma
import matplotlib.pyplot as plt
from tools.quadrature_tools import *
from tools.dataset_tools import get_synthetic_uniform_dataset

directory = 'real_approx/'

L = 1

def mc_bound(x_y, d, M_R, M_S, sigma):

    c2 = x_y**2/(2*sigma**2) #choose sigma s.t. c=1
    bound = (L * c2/np.sqrt(gamma(d/2)) * (c2/(2*M_R-1))**(2*M_R-1))\
            + 8/M_S * ((4*M_R+d)/(d-1) * c2)**2 * np.exp(4*(4*M_R+d)*c2/(d-1))
    return 2*bound

def omc_bound(x_y, d, M_R, M_S, sigma):
    c2 = x_y**2/(2*sigma**2)
    bound = (L * c2/np.sqrt(gamma(d/2)) * (c2/(2*M_R-1))**(2*M_R-1))\
            + 2/M_S * ((4*M_R+d)/(d-1) * c2)**4 * np.exp(4*(4*M_R+d)*c2/(d-1))
    return 2*bound



def main(dataset, d, N_R):

    approx_type = 'SR-OMC'
    nsamples = 5000
    repeated = 20
    N_list = np.arange(1, 6, 1)

    print('start dataset {}'.format(name))
    if name == 'Synthetic':
        X = get_synthetic_uniform_dataset(nsamples, d, 1)
        Y = -deepcopy(X)
    else:
        pass
    print('dataset size: ', X.shape)
    d = X.shape[1]
    sigma = np.sqrt(2)
    print('sigma: ', sigma)

    print('start to calculate exact kernel: ...')
    K_exact = kernel(X, Y, sigma, 'exact', None, N_R)
    samples = K

    print('calculate error: ...')
    err = np.zeros((len(N_list), repeated))
    for j, nn in enumerate(N_list):
        for r in range(repeated):
            K = kernel(X, Y, sigma, approx_type, nn, N_R)
            err[j, r] = np.norm(np.diag(K) - np.diag(K_exact))**2/nsamples # expected squared error
    
    upper_bound = [omc_bound(2, d, M_R, d*N, sigma) for N in N_list]
    err_m = np.average(err, axis=1)
    std = np.sqrt(np.var(err, axis=1))
    bins = N_R * d * N_list
    plt.plot(bins, upper_bound,  color = 'black', linewidth = 2, label = 'bound')
    plt.plot(bins, err_m,  color = color_list[approx_type], marker=marker_list[approx_type], markersize=10, linewidth = 2, label = approx_type)
    plt.fill_between(bins, err_m-std, err_m+std, color = color_list[approx_type], alpha = .2)
    plt.ylabel('Expected squared error', fontsize=24)
    plt.xlabel('Number of features', fontsize=24)
    plt.title('Practical error and upper bound', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plottings/upper_bound.png', format='png', dpi=200)
    plt.show()


    print('Finished!')

if __name__ == "__main__":
    dataset = sys.argv[1]
    d = sys.argv[2]
    N_R = sys.argv[3]
    main(dataset, int(d), int(N_R))