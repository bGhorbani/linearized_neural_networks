# This code calculates kernels for massive RF models that 
# cannot be fitted directly. For example, we use this code
# to generate kernels for RF models with N=4.2 \times 10^6
# on CIFAR-2. 
#
# The code was written for Python/2.7

from __future__ import print_function
import numpy as np
import sys
from preprocess import prep_data

import os 
import time
import cPickle as pickle

# Noise level index
noise_ind = np.int(sys.argv[1])
# Number of hidden units. Should be divisible by p
N = np.int(sys.argv[2])
# To avoid memory overflow, we compute the inner product
# between random features part by part. In particular, we
# compute the inner product for blocks of 50K features and 
# in the end, sum the partial inner products. 
p = 50000
M = np.int(N / p)
print('Noise index is %d'%(noise_ind))
print('The number of nodes is %d'%(N))
print('increment size: %d'%(p))
print('Number of iterations: %d'%(M))

directory = './CIFAR2/RF_Kernel_noise_%d'%(noise_ind)
if not os.path.exists(directory):
	os.makedirs(directory)
X, Y, Xtest, Ytest = prep_data('CIFAR2', False, noise_ind)

# Hyper-parameters
n = X.shape[0]
d = X.shape[1]
nt = Xtest.shape[0]

one = np.ones((n, 1))
onet = np.ones((nt, 1))
K = np.dot(one, one.T)
KT = np.dot(onet, one.T)

#K = K.astype(np.float32)
#KT = KT.astype(np.float32)
def fun(x):
	return np.maximum(0, x)
random_seed = 1990
print('random_seed: %d'%(random_seed))
np.random.seed(random_seed)
for i in range(M):
	print(i)
	W = np.random.normal(size=(d, p)) / np.sqrt(d)
	#W = W.astype(np.float32)
	sys.stdout.flush()
	Z = fun(np.dot(X, W))
	#print(Z.shape)
	K += np.dot(Z, Z.T) 
	ZT = fun(np.dot(Xtest, W))
	KT += np.dot(ZT, Z.T)

print('Z has the type %s'%(Z.dtype))
print('ZT has the type %s'%(ZT.dtype))
print('Test Kernel Done!')

# save K
fileName = directory + "/" + 'RF_Kernel_Train_N_%d.npy'%(N)
np.save(fileName, K)
fileName = directory + "/" + 'RF_Kernel_Test_N_%d.npy'%(N)
np.save(fileName, KT)