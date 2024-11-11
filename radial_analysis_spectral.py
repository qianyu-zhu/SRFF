#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:36:17 2024

@author: juliezhu
"""
import sys
import numpy as np
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from tools.kernel_tools import *
from tools.dataset_tools import *
from tools.quadrature_tools import *
from tools.visualization_tools import *
from spherical_comparison import color_list
from tools.dataset_tools import make_dataset, sample_dataset

compute_K = True
compute_err = True
directory = 'real_approx/'

marker_list = ['P','^','v','*','s','o', 'D']

threshold_dic = { 4: 1200,
                 16: 3200,
                256: 4200,
                784: 15200
                }


def relative_errors(X, Y, K_exact_inv_sqrt, approx_types, sigma, N_S_list, N_R_list, repeated=3):
    # need an exact kernel
    errs = {}
    d = X.shape[1]
    threshold = threshold_dic[d]
    for N_R in N_R_list:
        N_S_list = [int(i) for i in N_S_list]
        for approx_type in tqdm(approx_types):
            errs[approx_type+str(N_R)] = np.zeros((len(N_S_list), repeated))
            for j, nn in enumerate(tqdm(N_S_list)):
                for r in range(repeated):
                    if nn * N_R * d > threshold:
                        errs[approx_type+str(N_R)][j, r] = None
                    else:
                        K = kernel(X, Y, sigma, approx_type, nn, N_R)
                        # errs[approx_type+str(N_R)][j, r] = kernel_approximation_err(K_exact, K, 'rel f')
                        errs[approx_type+str(N_R)][j, r] = np.linalg.norm(K_exact_inv_sqrt @ K @ K_exact_inv_sqrt - np.identity(len(K)), 2)
                # print('N_R: ', N_R, 'N_S', nn, approx_type)
    return errs


def main(dataset, d, size):
    approx_types = ['SR-OMC']

    name = dataset

    # start_deg, max_deg, repeated, shift, step, nsamples, delimiter
    sample_params = [0, 5, 100, 0, 1, 500, 8500]
    start_deg, max_deg, repeated, shift, step, nsamples, _ = sample_params
    N_S_list = 2**np.arange(0, 6)
    N_R_list = [1, 2, 3, 5, 7]


    print('start dataset {}'.format(name))

    #### use real dataset
    dataset, params = make_dataset(name, sample_params, 'datasets/')
    X = sample_dataset(nsamples, dataset)

    #### use synthetic dataset
    # N = 1000
    # X = get_synthetic_Gaussian_dataset(N,d)
    print('dataset size: ', X.shape)

    #### Define the Gaussian kernel (be aware of the bandwidth)
    d = X.shape[1]
    if size=="small":
        sigma = 2*d**(1/4)/10 # theoretical gaurantee: >=2*d**(1/4)
    elif size=='medium':
        sigma = 2*d**(1/4)
    elif size=='large':
        sigma = 2*d**(1/4)*10
    print('sigma: ', sigma)

    sigma_str = '_sigma={:.2f}'.format(sigma)
    dim_str = '_d={:d}'.format(d)
    if compute_K:
        K_exact = kernel(X, X, sigma, 'exact', None, None)
        K_exact_inv_sqrt = inv_sqrt_m(K_exact)
        with open(directory + 'dataset_matrix/spectral_' + name + dim_str + sigma_str, 'wb') as f:
            pkl.dump(K_exact, f)
            pkl.dump(K_exact_inv_sqrt, f)
            print('saved kernel')
    else:
        with open(directory + 'dataset_matrix/spectral_' + name + dim_str + sigma_str, 'rb') as f:
            K_exact = pkl.load(f)[:nsamples,:nsamples]
            K_exact_inv_sqrt = pkl.load(f)
        print('load kernel/inverse square root done!')

    print('calculate error')
    if compute_err:
        errs = relative_errors(X, X, K_exact_inv_sqrt, approx_types, sigma, N_S_list, N_R_list, repeated)
        
        with open(directory + 'radial/spectral_' + name + str(d) + sigma_str, 'wb') as f:
            pkl.dump(errs, f)
        print('saved errs')
    else:
        with open(directory + 'radial/spectral_' + name + str(d) + sigma_str, 'rb') as f:
            errs = pkl.load(f)
        print('load errs')

    for i, N_R in enumerate(N_R_list):
        for approx_type in ['SR-OMC']:
            err = errs[approx_type+str(N_R)]
            err_m = np.average(err, axis=1)
            bins = N_R * d * N_S_list
            print(approx_type+str(N_R), err_m)
            err_m[err_m < 10**(-10)] = None
            plt.loglog(bins, err_m, marker=marker_list[i], markersize=10, \
                       linewidth = 2, label = r'$M_R=$'+str(N_R))
    plt.xlim([N_R_list[0]*d*N_S_list[0], min(N_R_list[-1]*d*N_S_list[-1], threshold_dic[d])])
    plt.ylabel('Relative error', fontsize=22)
    plt.xlabel(r'Total features, $M_S\times M_R$', fontsize=22)
    plt.legend(loc='lower left', fontsize=18)
    plt.title(r'{}, $d=${:d}, $\sigma=${:.2f}'.format(name, d, sigma), fontsize=22)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plottings/radial_analysis_spectral_' + name + dim_str + sigma_str + '.png', format='png', dpi=200)
    plt.show()

    print('Finished!')


if __name__ == "__main__":
    dataset = sys.argv[1]
    d = int(sys.argv[2])
    size = sys.argv[3]
    main(dataset, d, size)