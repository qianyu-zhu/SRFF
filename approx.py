#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script is used to compare the relative error of different kernel approximation methods on approximating real-life kernel gram matrices.
the matrix is 5000*5000, and the kernel is Gaussian kernel with bandwidth 2*d^(1/4) * sig_frac, where sig_frac is a customized scaling parameter.
repeated 10 times to compute the sd.
"""

import sys
import numpy as np
import pickle as pkl
from tools.kernel_tools import *
from tools.dataset_tools import *
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.visualization_tools import *
from tools.dataset_tools import make_dataset, sample_dataset

compute_K = True
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

directory = 'real_approx/'

def relative_errors(X, Y, K_exact, approx_types, sigma, N_list1, N_list2, N_R, repeated=3):
    # need an exact kernel
    errs = {}
    for approx_type in approx_types:
        errs[approx_type] = np.zeros((len(N_list1), repeated))
        if approx_type == 'exact':
            continue
        if approx_type == 'SSR':
            N_list = N_list2
        else:
            N_list = N_list1
        for j, nn in enumerate(N_list):
            # print('current round: ', nn)
            for r in range(repeated):
                K = kernel(X, Y, sigma, approx_type, nn, N_R)
                errs[approx_type][j, r] = kernel_approximation_err(K_exact, K, 'rel f')
    return errs

def main(dataset, N_R, sig_frac):
    approx_types = ['RFF', 'ORF', 'QMC', 'SSR', 'SR-OMC']  #

    datasets = [dataset] #['Powerplant' 'LETTER', 'USPS', 'MNIST', 'CIFAR100', 'LEUKEMIA']


    # start_deg, max_deg, runs, shift, step, nsamples, delimiter
    sample_params = [0, 5, 10, 0, 1, 5000, 8500]
    nsamples = 5000
    repeated = 10
    # N_list1 = np.arange(1, 6, 1)
    N_list1 = np.arange(1, 6, 1)# * int(2/N_R)
    N_list2 = np.arange(1, 6, 1)
    for name in datasets:

        print('start dataset {}'.format(name))
        if name == 'Synthetic':
            X = get_synthetic_Gaussian_dataset(nsamples,d)
            Y = deepcopy(X)
        else:
            dataset, params = make_dataset(name, sample_params, 'datasets/')
            X = sample_dataset(nsamples, dataset)
            Y = sample_dataset(nsamples, dataset)
        print('dataset size: ', X.shape)

        #### Define the Gaussian kernel (be aware of the bandwidth)
        d = X.shape[1]
        sigma = 2*d**(1/4) * sig_frac
        #sigma = 2*d**(1/4) # theoretical gaurantee: >=2*d**(1/4)
        print('sigma: ', sigma)

        print('start to calculate exact kernel: ...')
        #or load from local
        if compute_K:
            K_exact = kernel(X, Y, sigma, 'exact', None, N_R)
            with open(directory + 'dataset_matrix/' + name + 'sigma=' + str(sigma), 'wb') as f:
                pkl.dump(K_exact, f)
                print('saved kernel')
        # or load from local
        else:
            with open(directory + 'dataset_matrix/' + name + 'sigma=' + str(sigma), 'rb') as f:
                K_exact = pkl.load(f)
            print('load kernel done!')

        print('calculate error: ...')
        if compute_err:
            errs = relative_errors(X, Y, K_exact, approx_types, sigma, N_list1, N_list2, N_R, repeated)
            with open(directory + 'err_bin/' + name + 'sigma=' + str(sigma) + 'N_R=' + str(N_R), 'wb') as f:
                pkl.dump(errs, f)
            print('saved errs')
        else:
            with open(directory + 'err_bin/' + name + 'sigma=' + str(sigma) + 'N_R=' + str(N_R), 'rb') as f:
                errs = pkl.load(f)
                print('load errs')
        

        sigma_str = '_sigma={:.2f}'.format(sigma)
        N_R_str = '_N_R=' + str(N_R)

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
            plt.fill_between(bins, err_m-std, err_m+std, color = color_list[approx_type], alpha = .2)
            # plt.plot(bins, err_m+std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
            # plt.plot(bins, err_m-std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
        plt.ylabel('Relative error', fontsize=24)
        plt.xlabel('Number of features', fontsize=24)
        plt.xlim([min(bins1[0], bins2[0]), min(bins1[-1], bins2[-1])])
        plt.title(r'{}'.format(name), fontsize=24)
        plt.legend(loc='upper right', fontsize=18)
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig('plottings/real_approx_' + name + sigma_str + N_R_str + '.png', format='png', dpi=200)
        plt.show()


        print('Finished!')


if __name__ == "__main__":
    dataset = sys.argv[1]
    N_R = sys.argv[2]
    sig_frac = sys.argv[3]
    main(dataset, int(N_R), float(sig_frac))

    # how to run the script:
    # python -u real_approx.py dataset-name / number of radial nodes / scaling factor of the bandwidth
    # e.g.: "python -u real_approx.py 'Powerplant' 1 1"