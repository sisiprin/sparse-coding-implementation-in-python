import numpy as np
from scipy.optimize import minimize
from numpy.linalg import solve


# X: random part of X, matrix of all patches, (196 * 1000)
# S: corresponding part of sparse coefficients, (128, 1000)
# l2norm: const
# Binit: None, or


def l2ls_learn_basis_dual(X, S, l2norm, Binit=None):
    # Learning basis using Lagrange dual (with basis normalization)
    #
    # This code solves the following problem:
    # min B = 0.5 ||X - BS|| ^2 (least sqaure problem)
    # s.t. ||B[:,j]||_2 <= l2norm for all j = 1 .. size(S,1)
    #
    # This is a python implementation of the algorithm mentioned in
    # 'Efficient Sparse Coding Algorithms'

    L, N = X.shape  # 196 * 1000
    M = S.shape[0]  # 128

    SSt = S.dot(S.T)  # 128 * 128
    XSt = X.dot(S.T)  # 196 * 128

    if Binit is None:
        dual_lambda = 10 * np.random.rand(M)  # rand by default generate positive values
    else:
        B_sol, res, rank, sin_val = np.linalg.lstsq(Binit, XSt)
        dual_lambda = np.diag(B_sol - SSt)

    # debug
    # dual_lambda = loadmat('dual_lambda.mat')

    c = np.power(l2norm, 2)
    trXXt = np.sum(np.sum(np.power(X, 2)))

    lb = np.zeros(dual_lambda.shape)

    # options = {'GradObj':'on', 'Hessian':'on'}
    n = np.array([None] * dual_lambda.shape[0]).reshape(dual_lambda.shape)
    bounds = np.c_[lb, n]

    #----original
    result = minimize(lambda x: fobj_basis_dual(x, SSt, XSt, X, c, trXXt),
                      x0= dual_lambda, bounds= bounds, method= 'L-BFGS-B',
                      options={'disp':False})
    x = result['x']
    fval = result['fun']
    exitFlag = result['success']

    # constrained nonlinear function

    fval_opt = -0.5 * N * fval
    dual_lambda = x

    Bt = solve((SSt + np.diag(dual_lambda)), XSt.T)
    B_dual = Bt.T
    fobjective_dual = fval_opt

    B = B_dual
    fobjective = fobjective_dual
    # toc?

    return B



def fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt, nargout= 1):
    L = XSt.shape[0]
    M = dual_lambda.shape[0]

    SSt_inv = np.linalg.inv(SSt + np.diag(dual_lambda))

    if L > M:
        f = - np.trace(SSt_inv.dot(XSt.T.dot(XSt))) + trXXt - c * np.sum(dual_lambda)
    else:
        f = - np.trace((XSt.dot(SSt_inv)).dot(XSt.T)) + trXXt - c * np.sum(dual_lambda)

    f = -f

    if nargout > 1:
        temp = XSt.dot(SSt_inv)
        g = np.sum(np.power(temp, 2)) - c
        g = -g
        return f, g

        if nargout > 2:
            H = -2 * (temp.T.dot(temp).dot(SSt_inv))
            H = -H
            return f, g, H

    else:
        return f


#solve instead of inverse. SLOWER.....
def fobj_basis_dual2(dual_lambda, SSt, XSt, X, c, trXXt, nargout= 1):
    L = XSt.shape[0]
    M = dual_lambda.shape[0]

    # !!!!!!!!!!!!!!!!!

    SSt_plus_diag = SSt + np.diag(dual_lambda)

    intermediate = None
    if L > M:
        intermediate = solve(SSt_plus_diag,XSt.T.dot(XSt))
    else:
        intermediate = XSt.dot(np.linalg.solve(SSt_plus_diag, XSt.T))

    f = - np.trace(intermediate) + trXXt - c * np.sum(dual_lambda)

    f = -f

    if nargout > 1:
        temp = XSt.dot(SSt_inv)
        g = np.sum(np.power(temp, 2)) - c
        g = -g
        return f, g

        if nargout > 2:
            H = -2 * (temp.T.dot(temp).dot(SSt_inv))
            H = -H
            return f, g, H

    else:
        return f