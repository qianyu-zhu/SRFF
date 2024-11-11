import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import gamma
from tools.quadrature_tools import *

"""
this script is to compare the expected squared error of Monte-Carlo and Orthogonal Monte-Carlo in different dimensions
"""

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rcParams['figure.dpi'] = 200
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def sample_orthogonal_matrix(d):
    # Generate a random matrix with entries from a normal distribution
    random_matrix = np.random.randn(d, d)
    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix)
    return q

def eval_MC(rxy, d, N):
    """
    rxy: r(x-y), (d,)
    d: dimension of sample
    N: number of monte-carlo samples from a unitary sphere
    """
    x = np.random.multivariate_normal(np.zeros(d), np.eye(d), N) #(N,d)
    norm = np.linalg.norm(x, axis = 1)
    y = np.diag(1/norm)@x
    area = 2*np.pi**(d/2)/gamma(d/2)
    weight = area/N
    return np.sum(np.cos(y@rxy))*weight

def eval_orth(rxy, d, N):
    """
    rxy: r(x-y), (d,)
    d: dimension of sample
    N: number of orthogonal generated samples from a unitary sphere
    """
    n_matrix = int(np.ceil(N/d))
    x = np.concatenate([sample_orthogonal_matrix(d) for _ in range(n_matrix)], axis = 0) #(np.ceil(N/d),d)
    x = x[:N, :] #(N,d)
    area = 2*np.pi**(d/2)/gamma(d/2)
    weight = area/N
    return np.sum(np.cos(x@rxy))*weight


def ground_truth(rxy, d):
    # evauate the true value of cos(rxy^w) integrated on unit sphere
    N = 2**15
    result = [eval_orth(rxy, d, N) for _ in range(3)]
    return np.mean(result), np.var(result)


def sample_xy(d):
    x = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    x = x/np.linalg.norm(x)
    return x



def dimension_plotting(dd_list = np.arange(1, 6)):
    fig, ax = plt.subplots(1, len(dd_list), dpi = 120, figsize=(14,2))
    for j, dd in enumerate(dd_list):
        d = 2**dd
        N_list = [2**(i+dd) for i in range(6)]
        K = 200
        xy = sample_xy(d)
        print('start to compute ground truth: ...')
        truth, var = ground_truth(xy, d)
        print('true value of integral: ', truth, 'variance of estimation: ', var)

        # error and feature plotting for different dimension
        errs = np.zeros((2,len(N_list)))
        std = np.zeros((2,len(N_list)))
        for i, N in enumerate(tqdm(N_list)):
            cache = np.zeros((2,K))
            for k in range(K):
                cache[:,k] = [eval_MC(xy, d, N), eval_orth(xy, d, N)]
            cache = np.abs(cache-truth)**2
            errs[:,i] = np.average(cache, axis = 1)
            std[:,i] = np.sqrt(np.var(cache, axis = 1))


        ax[j].loglog(N_list, errs[0,:], marker='*', markersize=5, linewidth = 1, color = 'r', label = 'MC')
        ax[j].loglog(N_list, errs[1,:], marker='s', markersize=5, linewidth = 1, color = 'blue', label = 'OrtMC')
        ax[j].fill_between(N_list, errs[0,:]-std[0,:], errs[0,:]+std[0,:], color = 'r', alpha = .1)
        ax[j].fill_between(N_list, errs[1,:]-std[1,:], errs[1,:]+std[1,:], color = 'blue', alpha = .1)
        ax[j].set_xlim([N_list[0], N_list[-1]])
    ax[0].set_ylabel(r'E$[\epsilon]^2$')
    ax[2].set_xlabel(r'$M_S$')
    ax[2].set_title('Expected squared error for different dimensions')
    plt.tight_layout()
    plt.legend(loc=7)
    plt.show()

dimension_plotting(dd_list = np.arange(1, 6))