# with open('../fast_sc/','r') as img_file:
import scipy.io as sio
import numpy as np
from getdata_imagearray import *
from sparse_coding import *
from scipy.io import savemat
from loadmat import *

# take in a parameter
opt_choice = 2

IMAGES = sio.loadmat('IMAGES.mat')
X = getdata_imagearray(IMAGES, 14, 10000)

#X = loadmat('../../fast_sc/const/X.mat','X')
#B = loadmat('../../fast_sc/const/B.mat','B')

#savemat('./res/X',{'X':X})

# debug
#X = loadmat('X.mat')

num_bases = 128
beta = 0.4
batch_size = 1000
num_iters = 20

if opt_choice == 1:
    sparsity_func = 'epsL1'
    epsilon = 0.4
else:
    sparsity_func = 'L1'
    epsilon = None

fname_save = None
# !!!!!!!

B, S, stat = sparse_coding(X, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, Binit=B)







# print(X)
