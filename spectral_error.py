#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script is used to compare the relative error of different kernel approximation methods on synthetic datasets.
"""


import sys
import numpy as np
import pickle as pkl
from scipy.linalg import logm
from tools.kernel_tools import *
from tools.dataset_tools import *
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.visualization_tools import *
from tools.dataset_tools import make_dataset, sample_dataset

compute_err = True

marker_list = {'SR-MC': 'P',
              'SR-SOMC': 'P',
              'SR-OMC': '^',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SOMC': 'v',
              'SR-OKQ-OMC': 'cornflowerblue',
              'RFF': '*',
              'ORF': 's', 
              'QMC': 'o', 
              'GQ': 'P',
              'SSR': 'D'}

color_list = {'SR-MC': 'y',
              'SR-SOMC': 'red',
              'SR-OMC': 'purple',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SOMC': 'blue',
              'SR-OKQ-OMC': 'cornflowerblue',
              'RFF': 'green',
              'ORF': 'gold', 
              'QMC': 'darkorange', 
              'GQ': 'turquoise',
              'SSR': 'lightcoral'}

directory = 'spectral_err/'


def relative_errors(nsamples, dataset, approx_types, sigma, N_list1, N_list2, N_R, repeated=3):
    # need an exact kernel
    errs = {}
    for approx_type in approx_types:
        errs[approx_type] = np.zeros((len(N_list1), repeated))
    for r in range(repeated):
        X = sample_dataset(nsamples, dataset, random=True)
        K_exact = kernel(X, X, sigma, 'exact', None, None)
        # K_exact_inv_sqrt = inv_sqrt_m(K_exact)
        K_exact_inv = inv_m(K_exact)
        for approx_type in approx_types:
            if approx_type == 'SSR':
                N_list = N_list2
            else:
                N_list = N_list1
            for j, nn in enumerate(N_list):
                K = kernel(X, X, sigma, approx_type, nn, N_R)
                # errs[approx_type][j, r] = np.linalg.norm(K_exact_inv_sqrt @ K @ K_exact_inv_sqrt - np.identity(nsamples), 2)
                errs[approx_type][j, r] = np.linalg.norm(np.log(np.linalg.eigvals(K_exact_inv @ K)), 2)
    return errs


def main(dataset, N_R, sig_frac, nsamples):
    approx_types = ['ORF', 'QMC', 'SSR', 'SR-OMC'] #'RFF', 

    datasets = [dataset] #['Powerplant' 'LETTER', 'USPS', 'MNIST', 'CIFAR100', 'LEUKEMIA']

    # start_deg, max_deg, runs, shift, step, nsamples, delimiter
    sample_params = [0, 5, 10, 0, 1, 500, 8500]
    repeated = 20
    N_list1 = np.arange(1, 10*int(2/N_R)+1, int(2/N_R))
    # N_list1 = np.arange(1, 10, 2) * int(2/N_R)
    N_list2 = np.arange(1, 20, 2)
    for name in datasets:

        print('start dataset {}'.format(name))
        dataset, params = make_dataset(name, sample_params, 'datasets/')
        
        X = sample_dataset(nsamples, dataset)
        print('dataset size: ', X.shape)

        #### Define the Gaussian kernel (be aware of the bandwidth)
        d = X.shape[1]
        sigma = 2*d**(1/4) * sig_frac
        #sigma = 2*d**(1/4) # theoretical gaurantee: >=2*d**(1/4)
        print('sigma: ', sigma)

        sigma_str = '_sigma={:.2f}'.format(sigma)
        N_R_str = '_N_R=' + str(N_R)
        n_str = '_n='+str(nsamples)

        print('calculate error: ...')
        if compute_err:
            errs = relative_errors(nsamples, dataset, approx_types, sigma, N_list1, N_list2, N_R, repeated)
            # with open(directory + 'err_bin/' + name + n_str + sigma_str + N_R_str, 'wb') as f:
            with open(directory + 'err_bin/intrinsic_dist_' + name + n_str + sigma_str + N_R_str, 'wb') as f:
                pkl.dump(errs, f)
            print('saved errs')
        else:
            # with open(directory + 'err_bin/' + name + n_str + sigma_str + N_R_str, 'rb') as f:
            with open(directory + 'err_bin/intrinsic_dist_' + name + n_str + sigma_str + N_R_str, 'wb') as f:
                errs = pkl.load(f)
                print('load errs')
        



        for approx_type in approx_types:
            err = errs[approx_type]
            err_m = np.average(err, axis=1)
            std = np.sqrt(np.var(err, axis=1))
            bins1 = N_R * d * N_list1
            bins2 = ((d+1) * 2 ) * N_list2
            print(approx_type, err_m)
            if approx_type == 'SSR':
                bins = bins2
            else:
                bins = bins1
            plt.plot(bins, err_m,  color = color_list[approx_type], marker=marker_list[approx_type], markersize=10, linewidth = 2, label = approx_type)
            # plt.fill_between(bins, err_m-std, err_m+std, color = color_list[approx_type], alpha = .2)
            # plt.plot(bins, err_m+std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
            # plt.plot(bins, err_m-std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
        plt.ylabel('Spectral error', fontsize=20)
        plt.xlabel('Number of features', fontsize=20)
        plt.xlim([min(bins1[0], bins2[0]), min(bins1[-1], bins2[-1])])
        plt.title(r'Spectral error of {}'.format(name), fontsize=20)
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig('plottings/intrinsic_dist_' + name + n_str + sigma_str + N_R_str + '.png', format='png', dpi=200)
        plt.show()


        print('Finished!')


if __name__ == "__main__":
    dataset = sys.argv[1]
    N_R = sys.argv[2]
    sig_frac = sys.argv[3]
    nsamples = sys.argv[4]
    main(dataset, int(N_R), float(sig_frac), int(nsamples))


    # how to run:
    # python spectral_error.py 'Powerplant' 2 0.5 100
    # python spectral_error.py 'Powerplant' 2 1 100
    # python spectral_error.py 'Powerplant' 2 1.5 100