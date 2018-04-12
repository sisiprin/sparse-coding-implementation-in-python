# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 12:01:28 2016

@author: Fanchen_Kong
"""
import warnings

import numpy as np
from numpy.linalg import solve


# no sparse representation


# Solve the problem of ||Y-AX||^2 + gamma * l1_norm(Y)
# A: the basis codebook, (196 * 128)
# Y: the matrix of feature vectors (196 * 1,000)
# gamma:penalty constant
# return: Xout, a matrix of encoding vectors (128 * 1000)
# Xout.row = A.col, Xout.col = Y.col

def l1ls_featuresign(A, Y, gamma, Xinit=None):
    # Initialize an zombie matrix for return Xout
    Xout = np.zeros((A.shape[1], Y.shape[1]), order='F')

    # Calculate constant variables
    AtA = A.T.dot(A)
    AtY = A.T.dot(Y)
    rankA = np.min((A.shape[0] - 10, A.shape[1] - 10))

    xinit = None
    # for each col vector y in Y
    for i in range(Y.shape[1]):
        # print a '.' for every 100 col processed
        if np.mod(i, 100) == 99:
            print('.')
            #print('col %s' %i)

        if i == 9:
            print('here')

        if Xinit is not None:
            idx1 = Xinit[:, i].nonzero()[0]  # 1d
            maxn = np.amin((len(idx1), rankA))  # val
            xinit = np.zeros(len(Xinit[:, i]), order='C')
            xinit[idx1[0:maxn]] = Xinit[idx1[0:maxn], i]

        #if i == 53:
            #print('col 53!!')
            #savemat('y_',{'y': Y[:, i]})
            #savemat('Aty_', {'Aty':AtY[:,i]})
            #savemat('AtA',{'AtA':AtA})
        #res,fobj = ls_featuresign_sub(A, Y[:, i], AtA, AtY[:, i], gamma,xinit)
        #Xout[:, i] = res
        Xout[:, i], fobj = ls_featuresign_sub(A, Y[:, i], AtA, AtY[:, i], gamma,xinit, i)

    return Xout


# solve the subproblem of ||y-Ax||^2 + gamma * l1_norm(y)
# for y is one col of Y, and x is one col of X
# A: the basis codebook, (196 * 128)
# y: one col of Y, (196)
# AtA: (128 * 128)
# Aty: (128)
# gamma: penalty constant
# xinit: None or (128)

def ls_featuresign_sub(A, y, AtA, Aty, gamma, xinit=None, col=None):
    L, M = A.shape
    rankA = np.min((L - 10, M - 10))

    usexinit = False
    if xinit is None:
        x = np.zeros(M, order='C')
        theta = np.zeros(M, order='C')
        act = np.zeros(M, order='C')
        allowZero = False
    else:
        x = xinit
        theta = np.sign(xinit)
        act = np.absolute(theta)
        allowZero = True
        usexinit = True

    fobj = 0

    ITERMAX = 1000
    optimality1 = False
    # specify number of max iterations to run
    for i in range(ITERMAX):
        #if i == 22:
            #print('here')

        act_indx0 = np.where(act == 0)[0]
        # AtA.dot(x): (128)
        # Aty: (128)
        # grad:(128)
        grad = AtA.dot(x) - Aty
        theta = np.sign(x)

        # optimality0 = False

        abs_grad = np.absolute(grad[act_indx0])
        mx = np.max(abs_grad)
        indx = np.argmax(abs_grad)

        if mx >= gamma and (i > 0 or not (usexinit)):
            act[act_indx0[indx]] = 1
            theta[act_indx0[indx]] = -np.sign(grad[act_indx0[indx]])
            usexinit = False
        else:
            # optimality0 = True
            if optimality1:
                break

        act_indx1 = np.where(act == 1)[0]

        if act_indx1.shape[0] > rankA:
            warnings.warn('sparsity penalty is too small: too many coefficients are activated')
            return x, fobj

        if act_indx1.shape[0] == 0:
            if allowZero:
                allowZero = False
                continue
            return x, fobj

        k = 0
        while True:
            k = k + 1
            if k == 900:
                print('col %d has exceeded 900 iterations' % col)
            if k >= ITERMAX:
                print('col %d has exceeded max iter' %col)
                warnings.warn('Maximum number of iteration reached. The solution may not be optimal')
                return x, fobj

            if act_indx1.shape[0] == 0:
                if allowZero:
                    allowZero = False
                    break
                return x, fobj
            # step 3: feature-sign step

            #if i == 22:
                #print('22')
            #if i == 23:
                #print('stop')
            x, theta, act, act_indx1, optimality1, lsearch, fobj = compute_FS_step(x, A, y, AtA, Aty, theta, act,
                                                                                   act_indx1, gamma)

            # step4: Check optimality condition1
            if optimality1:
                break
            if lsearch > 0:
                continue

    if i >= ITERMAX:
        warnings.warn('Maximum number of iteration reached. The solution may not be optimal')

    fobj, g = fobj_featuresign(x, A, y, AtA, Aty, gamma)
    return x, fobj


# x: (128)
# A: (196 * 128)
# y: (196)
# AtA: (128 * 128)
# Aty: (128)
# theta: (128)
# act: (128)
# act_indx1: (x)
# gamma: const
def compute_FS_step(x, A, y, AtA, Aty, theta, act, act_indx1, gamma):
    x2 = x[act_indx1]  # (size of act_indx1)
    AtA2 = AtA[:, act_indx1]
    AtA2 = AtA2[act_indx1, :]  # (size of act_indx1 * size of act_indx1)
    theta2 = theta[act_indx1]  # (size of act_indx1)

    # size of act_indx1
    x_new = solve(AtA2, Aty[act_indx1] - gamma * theta2)

    optimality1 = False
    if np.array_equal(np.sign(x_new), np.sign(x2)):
        optimality1 = True
        x[act_indx1] = x_new
        fobj = 0
        lsearch = 1
        return x, theta, act, act_indx1, optimality1, lsearch, fobj

    # do line search: x -> x_new
    # progress: (size of act_indx1)
    progress = (0 - x2) / (x_new - x2)
    lsearch = 0

    a = 0.5 * np.sum(np.power(A[:, act_indx1].dot(x_new - x2), 2))
    # x2.dot is the same as x2.T.dot
    b = ((x2.dot(AtA2)).dot(x_new - x2) - (x_new - x2).dot(Aty[act_indx1]))
    fobj_lsearch = gamma * np.sum(np.absolute(x2))

    temp_progress = np.append(progress, 1)
    sort_lsearch = np.sort(temp_progress)
    ix_lsearch = np.argsort(temp_progress)
    remove_idx = np.array([])

    for i in range(len(sort_lsearch)):
        t = sort_lsearch[i]
        if t <= 0 or t > 1:
            continue
        # (size of act_indx1)
        s_temp = x2 + (x_new - x2) * t

        fobj_temp = a * np.power(t, 2) + b * t + gamma * np.sum(np.absolute(s_temp))

        if fobj_temp < fobj_lsearch:
            fobj_lsearch = fobj_temp
            lsearch = t
            if t < 1:
                remove_idx = np.append(remove_idx, ix_lsearch[i])
        elif fobj_temp > fobj_lsearch:
            break
        else:
            if (np.sum(x2 == 0)) == 0:
                lsearch = t
                fobj_lsearch = fobj_temp
                if t < 1:
                    remove_idx = np.append(remove_idx, ix_lsearch[i])

    if lsearch > 0:
        # update x
        x_new = x2 + (x_new - x2) * lsearch
        x[act_indx1] = x_new
        theta[act_indx1] = np.sign(x_new)

    # if x encounters zero along the line search, then remove it from active set
    if lsearch < 1 and lsearch > 0:
        eps = np.finfo(float).eps
        remove_idx = np.where(np.absolute(x[act_indx1]) < eps)[0]
        x[act_indx1[remove_idx]] = 0

        theta[act_indx1[remove_idx]] = 0
        act[act_indx1[remove_idx]] = 0
        act_indx1 = np.delete(act_indx1, remove_idx)

    fobj_new = 0
    fobj = fobj_new
    return x, theta, act, act_indx1, optimality1, lsearch, fobj


# x: (128)
# A: (196 * 128)
# y: (196)
# AtA: (128 * 128)
# Aty: (128)
# gamma: const
def fobj_featuresign(x, A, y, AtA, Aty, gamma, nargout=1):
    f = 0.5 * np.power(np.linalg.norm(y - A.dot(x)), 2)
    f = f + gamma * np.linalg.norm(x, ord=1)

    # if we do not need to compute g, change nargout = 1
    g = None
    if nargout > 1:
        g = AtA.dot(x) - Aty
        g = g + gamma * (np.sign(x))
    return f, g
