#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare the relative error of methods on dataset of different diameters.
"""


import numpy as np



from tools.function_tools import *
from tools.quadrature_tools import *
from tools.kernel_tools import *
from tools.visualization_tools import *
from tools.dataset_tools import *
from tqdm import tqdm


#### Change the dataset here
#### For now, we use a synthetic Gausssian dataset
d = 4
N_d = 10**3
# dataset = get_synthetic_Gaussian_dataset(N_d,d)

#### Define the Gaussian kernel (be aware of the bandwidth)
sigma = 2*np.sqrt(d)
my_asymptotic_kernel = Gaussian_kernel(sigma)


#### Define the SR quadrature that we will use, along with the corresponding empirical kernels
## The radial quadrature (shared between several SR quadratures):
alpha = d/2-1
N_R = 2
N_S_ = 8
N_S = N_S_*d
N = 2*N_R*N_S
r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)


## number of features vs. relative error in matrix spectral norm
n_model = 4

D_list = [0.2*i for i in range(2,21,2)]
max_err_list = np.zeros((n_model,len(D_list)))
mean_err_list = np.zeros((n_model,len(D_list)))

## Quadrature 2: An SR quadrature, the spherical part makes use of i.i.d. samples from the U(Sd), and optimal weights weighted
## with respect to the Gaussian kernel (be aware of the spherical bandwidth and the reg parameter)

sigma_S = 0.2
lambda_S = 0.00001
MCS_weights, MCS_nodes = [1/(2*N_S)]*(2*N_S), sample_seq_orthogonal_matrices(N_S_,d)
# MCS_weights, MCS_nodes = UG_on_2d_sphere(N_S,d)
OKQMCS_weights, OKQMCS_nodes = OKQ_on_hypersphere(MCS_nodes,d,sigma_S,lambda_S)
weights_2, nodes_2 = combine_radial_spherical_quadratures(r_weights,r_nodes,OKQMCS_weights,OKQMCS_nodes)
my_empirical_kernel_2 = empirical_kernel(weights_2,nodes_2,sigma)


## Quadrature 3: Vanilla MC (not an SR quadrature)
weights_MC = [1/N]*N
nodes_MC = simulate_M(d,N)
my_empirical_kernel_3 = empirical_kernel(weights_MC,nodes_MC,sigma)


## Quadrature 4:  QMC with Halton sequences
weights_QMC = [1/N]*N
nodes_QMC = quasi_monte_carlo_with_halton_nodes(d, N)
my_empirical_kernel_4 = empirical_kernel(weights_QMC,nodes_QMC,sigma)


## Quadrature 5:  sparse-grid quadrature
# nodes_gq, weights_gq = sparse_gauss_hermite_quadrature(d, N, deg=2)
# my_empirical_kernel_5 = empirical_kernel(weights_gq,nodes_gq,sigma)
# delta_empirical_kernel_5 = get_delta_functions(my_asymptotic_kernel,my_empirical_kernel_5)


## Quadrature 6:  MC with orthogonal random structure
weights_ort = [1/N]*N
nodes_ort = MC_with_orthogonal_random_structure(d, N)
my_empirical_kernel_6 = empirical_kernel(weights_ort,nodes_ort,sigma)


for i in tqdm(range(len(D_list)), ascii=True, ncols=100):
    
    cache_error = np.zeros((n_model, N_d))
    D = D_list[i]
    # generate random data with norm D
    dataset = get_synthetic_uniform_dataset(N_d,d,D)

    ground_truth = [my_asymptotic_kernel(dataset[i,:]) for i in range(N_d)]


    cache_error[0,:] = [my_empirical_kernel_2(dataset[i,:]) for i in range(N_d)]
    cache_error[1,:] = [my_empirical_kernel_3(dataset[i,:]) for i in range(N_d)]
    cache_error[2,:] = [my_empirical_kernel_4(dataset[i,:]) for i in range(N_d)]
    cache_error[3,:] = [my_empirical_kernel_6(dataset[i,:]) for i in range(N_d)]

    for j in range(n_model):
        max_err_list[j,i] = diameter_error(ground_truth, cache_error[j,:], 'max')
        mean_err_list[j,i] = diameter_error(ground_truth, cache_error[j,:], 'mean')

## plot
color_list = ['red', 'blue', 'green', 'purple', 'orange']
label_list = ['SRquad', 'vanilla MC', 'QMC', 'ortho MC']
marker_list = ['*', 's', 'o', 'D', 'v', '^', 'P']
# print('rel_err_list', rel_err_list)


for i in range(n_model):
    result = mean_err_list[i,:]
    plt.plot(2*D_list, result, color = color_list[i], linewidth = 0.5)
    plt.scatter(2*D_list, result, color = color_list[i], marker = marker_list[i], s=5, label = label_list[i])
plt.title('maximum error v.s. dataset diameter, d: {:d}, sigma: {:.2f}, N_R: {:d}, N_S: {:d}'.format(d, sigma, N_R, N_S), fontsize=14)
plt.xlabel('region diameter (D)', fontsize=14)
plt.ylabel('estimated mean error', fontsize=14)
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
plt.legend(fontsize=12)
plt.savefig('plottings/diameter_err.png', format='png', dpi=200)
plt.show()