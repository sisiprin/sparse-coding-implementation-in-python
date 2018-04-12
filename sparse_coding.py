# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import time
#import matplotlib.pyplot as plt

from l1ls_featuresign import *
from l2ls_learn_basis_dual import *
from loadmat import *
from scipy.io import savemat


def sparse_coding(X, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, pars=None, Binit=[],
                  resample_size=None):
    # X_total: training set
    # num_bases: number of bases
    # beta: sparsity penalty parameter
    # sparsity_func: sparsity penalty function ('L1', or 'epsL1')
    # epsilon: epsilon for epsilon-L1 sparsity
    # num_iters: number of iteration
    # batch_size: small-batch size
    # fname_save: filename to save
    # pars: additional parameters to specify (see the code)
    # Binit: initial B matrix
    # resample_size: (optional) resample size

    # resample

    pars = dict()
    pars['patch_size'] = X.shape[0]
    pars['num_patches'] = X.shape[1]
    pars['num_bases'] = num_bases
    pars['num_trial'] = num_iters
    pars['batch_size'] = batch_size
    pars['sparsity_func'] = sparsity_func
    pars['beta'] = beta
    pars['epsilon'] = epsilon
    pars['noise_var'] = 1
    pars['sigma'] = 1
    pars['VAR_basis'] = 1

    # Sparsity parameters
    if 'tol' not in pars:
        pars['tol'] = 0.005

    # L1 sparsity function
    if pars['sparsity_func'] == 'epsL1':
        pars['epsilon'] = epsilon
        pars['reuse_coeff'] = False
    else:
        pars['epsilon'] = []
        pars['reuse_coeff'] = True

    # initialize basis
    if Binit is None:
        # Xnot empty np.all(X==0) is false
        # ??????????????????????
        B = np.random.random((pars['patch_size'], pars['num_bases'])) - 0.5
        B = B - numpy.matlib.repmat(np.mean(B, axis=0), B.shape[0], 1)
        B = B.dot(np.diag(1. / (np.sqrt(np.sum(np.multiply(B, B), axis=0)))))
        #B = loadmat('../../fast_sc/const/B.mat','B')
    else:
        print "Using Binitial...."
        B = Binit

    L, M = B.shape
    winsize = np.sqrt(L)

    S_all = np.zeros((M, pars['num_patches']), order='F')



    time_record = list()

    # initialize t only if it does not exist
    t = 0
    # optimization loop
    while t < pars['num_trial']:

        t = t + 1

        indperm = np.random.permutation(X.shape[1])
        #savemat('./res/indperm%s'%t, {'indperm':indperm})
        #indperm = loadmat('../../fast_sc/const/indperm%s.mat'%t,'indperm')[0] - 1

        # debug
        #if t == 2:
                #B  = loadmat('../../fast_sc/res/B1.mat','B')
                #S_all = loadmat('../../fast_sc/res/S1.mat','S_all')

        num_batches = X.shape[1] / pars['batch_size']
        start = time.time()
        for batch in range(num_batches):
            print('batch: %s' % batch)

            #print('.')
            if (batch + 1) % 20 == 0:
                print '/n'

            batch_idx = indperm[(batch * pars['batch_size']):((batch + 1) * pars['batch_size'])]
            Xb = X[:, batch_idx]

            # learn coefficients
            Sinit = None
            if t > 1 or not pars['reuse_coeff']:
                Sinit = S_all[:, batch_idx]




            S = l1ls_featuresign(B, Xb, pars['beta'] / pars['sigma'] * pars['noise_var'], Sinit)



            # S[np.where(np.isnan(S) == np.ones(S.shape))] = 0
            s_name = './res/S%s_%s' % (t - 1, batch)
            #np.save(s_name, S)
            S_all[:, batch_idx] = S

            if pars['sparsity_func'] == 'L1':
                _sum = np.count_nonzero(S)

                sparsity_S = float(np.count_nonzero(S)) / (S.shape[0] * S.shape[1])
                print('sum =  %s' %_sum)
                print('density_S =  %s'  %sparsity_S)

                # update basis
            B = l2ls_learn_basis_dual(Xb, S, pars['VAR_basis'], B)
            #if t == 1 and batch == 0:
                #savemat('B_',{'B': B})


        end = time.time()
        time_record.append(end - start)
        print(end-start)
        #savemat('./res/B%s' %t,{'B':B})
        #savemat('./res/S%s' %t,{'S':S_all})

    #savemat('./res/T',{'T':time_record})
    print(sum(time_record))
    return B, S_all, None