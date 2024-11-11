#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize the distribution of the 2-norm of the data points in the dataset
"""
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tools.dataset_tools import *


def main(name):
    nsamples = 5000
    sample_params = [0, 5, 100, 0, 1, nsamples, 8500]
    d_list = [4, 16, 256, 784]
    
    # load dataset
    bins = np.arange(0,10,0.2)
    if name == 'Synthetic':
        for d in d_list:
            X = get_synthetic_Gaussian_dataset(nsamples,d)
            X_norm = np.linalg.norm(X, axis = 1)
            sns.histplot(X_norm, bins=30, kde=True, label = d, edgecolor=None)
    else:
        for dataset_name in ['Powerplant', 'LETTER', 'USPS', 'MNIST']:
            dataset, _ = make_dataset(dataset_name, sample_params, 'datasets/')
            X = sample_dataset(nsamples, dataset) #nsamples*d
            X_norm = np.linalg.norm(X, axis = 1)
            sns.histplot(X_norm, bins=30, kde=True, label = dataset_name, edgecolor=None)

    plt.ylabel('Histogram', fontsize=22)
    plt.xlabel('2 norm', fontsize=22)
    plt.legend(loc='upper right', fontsize=18)
    plt.title(name, fontsize=22)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plottings/datavisual_' + name + '.png', format='png', dpi=200)
    plt.show()

    print('Finished!')


if __name__ == "__main__":
    dataset = sys.argv[1]
    main(dataset)