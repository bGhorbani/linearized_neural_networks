"""
This code implements functionalities required for computing the NT Kernel for multi-layer
fully-connected neural networks. The computed kernels are saved to the disk. 

The code is written for Python 3.6. 

Inputs: 
	noise_id: The index of the noise intensity: valid range 0 to 14.
	num_layers: The number of layers: valid range {2, 3, 4}.
"""

from __future__ import print_function
import math
import os 
import sys
import time
from preprocess import prep_data
import numpy as np

from jax import random
from neural_tangents import stax

noise_id = np.int(sys.argv[1])
num_layers = np.int(sys.argv[2])
dataset = 'NFMNIST'
X, Y, Xtest, Ytest = prep_data(dataset, False, noise_id)	

if num_layers == 2:
	init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(512), stax.Relu(), stax.Dense(1))
elif num_layers == 3:
	init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(512), stax.Relu(), stax.Dense(512), stax.Relu(), stax.Dense(1))
elif num_layers == 4:
	init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(512), stax.Relu(), stax.Dense(512), stax.Relu(), stax.Dense(512), stax.Relu(), stax.Dense(1))
else:
	raise Exception('Non-valid Kernel')

n = X.shape[0]
kernel = np.zeros((n, n), dtype=np.float32)
m = n / 10
m = np.int(m)
# To avoid memory overflow, for training data, we fill the kernel matrix block by block
for i in range(10):
	for j in range(10):
		print('%d and %d'%(i, j))
		x1 = X[i * m:(i + 1) * m, :]
		x2 = X[j * m:(j + 1) * m, :]
		kernel[i * m:(i + 1) * m, j * m:(j + 1) * m] = kernel_fn(x1, x2, 'ntk')
print(kernel.shape)
directory = './NTK_Kernels/'
if not os.path.exists(directory):
	os.makedirs(directory)
file_name = 'Train_NTK_%d_layers_%d_%s.npy'%(noise_id, num_layers, dataset)
np.save(directory + file_name, kernel)

file_name = 'Test_NTK_%d_layers_%d_%s.npy'%(noise_id, num_layers, dataset)
kernel = kernel_fn(Xtest, X, 'ntk')
np.save(directory + file_name, kernel)