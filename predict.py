#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script is used to compare the prediction error of different kernel approximation methods on real datasets.
"""

import os
import sys
import numpy as np
import pickle as pkl
from sklearn import svm
from real_approx import marker_list
from tools.kernel_tools import *
from tools.dataset_tools import *
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.visualization_tools import *
from sklearn.metrics import accuracy_score
from tools.dataset_tools import make_dataset, sample_dataset


compute_err = True
directory = 'real_predict/'

color_list = {'SR-OMC': '#FA7F6F',
              'ORF': '#FFBE7A', 
              'QMC': '#8ECFC9', 
              'SSR': '#82B0D2'}

def get_scores(dataset_name, approx_type, N_list, clf, repeated, scores, N_R, sigma):
    dataset, _ = make_dataset(dataset_name)
    xtrain, ytrain, xtest, ytest = dataset
    d = xtrain.shape[1]
    n1 = xtrain.shape[0]
    n2 = xtest.shape[0]
    print('start training for {}, {}'.format(dataset_name, approx_type))
    print('training size: {:d}*{:d}, testing size: {:d}*{:d}'.format(n1, d, n2, d))

    scores[approx_type] = np.zeros((len(N_list), repeated))
    for k, nn in enumerate(N_list):
        if approx_type == 'exact' and k >=1:
            continue
        for i in range(repeated):
            if approx_type == 'exact' and i >=1:
                continue
            precomputed = kernel(xtrain, xtrain, sigma, approx_type, nn, N_R)
            precomputed_test = kernel(xtest, xtrain, sigma, approx_type, nn, N_R)
            clf.fit(precomputed, ytrain)
            if dataset_name in ['Powerplant']:
                predict = clf.predict(precomputed_test)
                # scores[approx_type][k][i] = np.mean(np.abs(predict - ytest)**2)
                scores[approx_type][k][i] = clf.score(precomputed_test, ytest)
            else:
                predict = clf.predict(precomputed_test)
                scores[approx_type][k][i] = accuracy_score(predict, ytest)
    return scores

dim_dic = {'Powerplant': 4,
           'LETTER':    16,
           'USPS':     256,
           'MNIST':    784}

def main(dataset_name, N_R, sig_frac):
    scores = {}
    repeated = 10
    d = dim_dic[dataset_name]

    if dataset_name in ['Powerplant']:
        clf = svm.SVR(kernel='precomputed')
    else: 
        clf = svm.SVC(kernel='precomputed')

    approx_types = ['exact', 'SSR', 'ORF', 'QMC', 'SR-OMC'] #add 'exact' 'SR-Ort', 'SR-OKQ-Ort', , 'SR-SymOrt', 'SR-OKQ-SOMC'
    # N_list1 = np.arange(1, 6, 1) 
    N_list1 = np.arange(1, 11, 2)
    N_list2 = np.arange(1, 6, 1)
    bins1 = N_R * d * N_list1
    bins2 = ((d+1) * 2 ) * N_list2

    sigma = 2*d**(1/4) * sig_frac
    print('sigma: {:.2f}'.format(sigma))

    sigma_str = '_sigma={:.2f}'.format(sigma)
    dim_str = '_d={:d}'.format(d)
    N_R_str = '_N_R=' + str(N_R)

    if compute_err:
        for approx_type in approx_types:
            if approx_type == 'SSR':
                bins = bins2
                N_list = N_list2
            else:
                bins = bins1
                N_list = N_list1
            scores = get_scores(dataset_name, approx_type, N_list, clf, repeated, scores, N_R, sigma)
        with open(directory + 'scores_' + dataset_name + dim_str + sigma_str, 'wb') as f:
            pkl.dump(scores, f)
            print('saved scores')
    else:
        with open(directory + 'scores_' + dataset_name + dim_str + sigma_str, 'rb') as f:
            scores = pkl.load(f)
            print('load scores')

    for approx_type in approx_types:
        if approx_type == 'SSR':
            bins = bins2
            N_list = N_list2
        else:
            bins = bins1
            N_list = N_list1
        if approx_type == 'exact':
            plt.plot(bins, [scores['exact'][0,0]]*len(bins), color = 'black', linestyle='dashed', linewidth = 2)
            continue
        average_error = np.average(scores[approx_type], axis = 1)
        std = np.sqrt(np.var(scores[approx_type], axis = 1))
        print(approx_type, average_error)
        plt.plot(bins, average_error, color = color_list[approx_type], marker=marker_list[approx_type], markersize=10, linewidth = 2, label = approx_type)
        plt.fill_between(bins, average_error-std, average_error+std, color = color_list[approx_type], alpha = .2)

    plt.title('{}'.format(dataset_name), fontsize=24)
    plt.xlim([min(bins1[0], bins2[0]), min(bins1[-1], bins2[-1])])
    # plt.ylim([0, 1.1])
    plt.xlabel('Number of features', fontsize=24)
    if dataset == 'Powerplant':
        plt.ylabel(r'$R^2$ score', fontsize=24)
    else:
        plt.ylabel('Accuracy', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plottings/real_predic_' + dataset_name + dim_str + sigma_str + N_R_str + '.png', format='png', dpi=200)
    plt.show()
    return


if __name__ == "__main__":
    dataset = sys.argv[1]
    N_R = sys.argv[2]
    sig_frac = sys.argv[3]
    main(dataset, int(N_R), float(sig_frac))



    # how to run the script:
    # python -u real_predic.py dataset-name / number of radial nodes / scaling factor of the bandwidth
    # e.g.: "python -u real_predic.py 'Powerplant' 2 1"
    