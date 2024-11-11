#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script is used to compare the relative error of different kernel approximation methods on real datasets.
"""


import numpy as np
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.kernel_tools import *
from tools.visualization_tools import *
from tools.dataset_tools import *
from tqdm import tqdm
from tools.dataset_tools import make_dataset, sample_dataset
import pickle as pkl

save = False
directory = 'real_approx/'
color_list = {'SR-MC': 'y',
              'SR-SOMC': 'red',
              'SR-OMC': 'lightcoral',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SOMC': 'blue',
              'SR-OKQ-OMC': 'cornflowerblue',
              'RFF': 'green',
              'ORF': 'gold', 
              'QMC': 'darkorange', 
              'GQ': 'turquoise',
              'SSR': 'purple'}


def relative_errors(X, Y, K_exact, approx_types, sigma, N_list1, N_list2, repeated=3):
    # need an exact kernel
    errs = {}
    for approx_type in tqdm(approx_types):
        errs[approx_type] = np.zeros((len(N_list1), repeated))
        if approx_type == 'exact':
            continue
        if approx_type == 'SSR':
            N_list = N_list2
        else:
            N_list = N_list1
        for j, nn in enumerate(tqdm(N_list)):
            for r in range(repeated):
                K = kernel(X, Y, sigma, approx_type, nn)
                errs[approx_type][j, r] = kernel_approximation_err(K_exact, K, 'rel f')
    return errs

def main():
    approx_types = ['RFF', 'SSR', 'SR-MC', 'SR-OMC', 'SR-SOMC', 'SR-OKQ-MC', 'SR-OKQ-OMC', 'SR-OKQ-SOMC']

    datasets = ['Powerplant'] #['LETTER''Powerplant''LETTER', 'USPS', 'MNIST', 'CIFAR100', 'LEUKEMIA']
    # datasets = ['Powerplant']

    # start_deg, max_deg, runs, shift, step, nsamples, delimiter
    sample_params = [0, 5, 10, 0, 1, 500, 8500]
    nsamples = 5000
    repeated = 20
    N_list1 = 2**np.arange(1, 5)  #np.arange(1, 11, 2)
    N_list2 = 3**np.arange(0, 4)

    for name in datasets:
        print('start dataset {}'.format(name))
        dataset, params = make_dataset(name, sample_params, 'datasets/')

        X = sample_dataset(nsamples, dataset)
        Y = sample_dataset(nsamples, dataset)
        print('dataset size: ', X.shape)

        #### Define the Gaussian kernel (be aware of the bandwidth)
        d = X.shape[1]
        sigma = 2*d**(1/4) # theoretical gaurantee: >=2*d**(1/4)
        print('sigma: ', sigma)

        print('start to calculate exact kernel: ...')
        if save:
            K_exact = kernel(X, Y, sigma, 'exact')
            with open(directory + 'dataset_matrix/' + name, 'wb') as f:
                pkl.dump(K_exact, f)
                print('saved kernel')
        # or load from local
        else:
            with open(directory + 'dataset_matrix/' + name, 'rb') as f:
                K_exact = pkl.load(f)[:nsamples,:nsamples]
            print('load kernel done!')

        print('calculate error')
        
        if save:
            errs = relative_errors(X, Y, K_exact, approx_types, sigma, N_list1, N_list2, repeated)
            with open(directory + 'spherical/' + name, 'wb') as f:
                pkl.dump(errs, f)
            print('saved errs')
        else:
            with open(directory + 'spherical/' + name, 'rb') as f:
                errs = pkl.load(f)
                print('load errs')

        for approx_type in approx_types:
            err = errs[approx_type]
            err_m = np.average(err, axis=1)
            std = np.sqrt(np.var(err, axis=1))
            bins1 = 3 * d * N_list1
            bins2 = ((d+1) * 2 ) * N_list2
            print(approx_type, err_m)
            if approx_type == 'SSR':
                bins = bins2
            else:
                bins = bins1
            plt.loglog(bins, err_m,  color = color_list[approx_type], linewidth = 2)
            plt.fill_between(bins, err_m-std, err_m+std, color = color_list[approx_type], alpha = .1)
            # plt.plot(bins, err_m+std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
            # plt.plot(bins, err_m-std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
            plt.scatter(bins, err_m, color = color_list[approx_type], marker='*', s=30, label = approx_type)
        plt.ylabel('Relative error')
        plt.xlabel('Number of features')
        plt.xlim([max(bins1[0], bins2[0]), min(bins1[-1], bins2[-1])])
        plt.title('Kernel approximation on {}'.format(name))
        plt.legend(loc='lower left')
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
        plt.savefig('plottings/real_spherical_' + name + '.eps', format='eps')
        plt.show()

        print('Finished!')


if __name__ == "__main__":
    main()