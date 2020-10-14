"""
Code for generating the Myrtle-5 convolutional neural tangent kernel for CIFAR10.
In order to parallelize the kernel generation process, each job generates 
only 200 rows of the kernel and saves the result to the disk. Moreover, 
to save computation, we only generate the upper diagonal part of the kernel
matrix. The code that fits the KRR needs to read all these rows, put them in 
the matrix format and populate the full matrix before starting the fitting
process.

To be able to run this code at scale, GPU supported JAX is necessary.

Inputs:
	noise_index: Noise level, valid range 0 to 9
	row_id: Index of the group of rows to be computed, valid range 0 to 249
	model_name: Architecture name, Myrtle
	exp_name: Name of the experiment (used for record keeping)
	job_id: The i.d of the job (used for record keeping)
"""

import numpy as onp
import sys
from preprocess import prep_data
import os 
import time

import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap

import functools
import neural_tangents as nt
from neural_tangents import stax

# Inputs passed to the function:
noise_index = onp.int(sys.argv[1])
row_id = onp.int(sys.argv[2])
model_name = sys.argv[3]
exp_name = sys.argv[4]
job_id = onp.int(sys.argv[5])

# The directory used to save the results
directory = './CNN_Kernels/%s_CIFAR10_%s_%d'%(exp_name, model_name, noise_index)
if not os.path.exists(directory):
	os.makedirs(directory)
files = os.listdir(directory)

fileName = directory + "/" + 'log_file_%d_%d.txt'%(row_id, job_id)
_file = open(fileName, 'w', buffering=1)

X, _, Xtest, _ = prep_data('CIFAR10', False, noise_index)

n = X.shape[0]
ntest = Xtest.shape[0]
W_std = 1.0
b_std = 0.0
# Number of rows generated at each job
m = onp.int(200)

if model_name == 'Myrtle':
	init_fn, apply_fn, kernel_fn = stax.serial(stax.Conv(512, (3, 3), strides=(1, 1), W_std=W_std, b_std=b_std, padding='SAME'),\
	 stax.Relu(),\
	 stax.Conv(512, (3, 3), strides=(1, 1), W_std=W_std, b_std=b_std, padding='SAME'),\
	 stax.Relu(),\
	 stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),\
	 stax.Conv(512, (3, 3), strides=(1, 1), W_std=W_std, b_std=b_std, padding='SAME'),\
	 stax.Relu(),\
	 stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),\
	 stax.Conv(512, (3, 3), strides=(1, 1), W_std=W_std, b_std=b_std, padding='SAME'),\
	 stax.Relu(),\
	 stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),\
	 stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),\
	 stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),\
	 stax.Flatten(),\
	 stax.Dense(10, W_std, b_std))
else:
	raise Exception('Invalid Input Error')

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnums=(2,))
kernel_fn = nt.batch(kernel_fn, batch_size=20)

X1 = X[row_id * m : (row_id + 1) * m, :, :, :]
assert X1.shape[0] == m and X1.shape[1] == 32 and X1.shape[2] == 32 and X1.shape[3] == 3


# Training kernel
K = onp.zeros((m, n), dtype=onp.float32)
col_count = onp.int(n / m)
for col_id in range(row_id, col_count):
	t1 = time.time()
	X2 = X[col_id * m : (col_id + 1) * m, :, :, :]
	assert X2.shape[0] == m and X2.shape[1] == 32 and X2.shape[2] == 32 and X2.shape[3] == 3
	temp = kernel_fn(X1, X2, 'ntk')
	K[:, col_id * m : (col_id + 1) * m] = temp.astype(onp.float32)
	t2 = time.time()
	print('Train Col index %d, took %f'%(col_id, t2 - t1), file=_file)

file_name = directory + "/" + 'train_ntk_%d.npy'%(row_id)
np.save(file_name, K)
del K

#Test kernel
K = onp.zeros((m, ntest), dtype=onp.float32)
col_count = onp.int(ntest / m)
for col_id in range(col_count):
	t1 = time.time()
	X2 = Xtest[col_id * m : (col_id + 1) * m, :, :, :]
	assert X2.shape[0] == m and X2.shape[1] == 32 and X2.shape[2] == 32 and X2.shape[3] == 3
	temp = kernel_fn(X1, X2, 'ntk')
	K[:, col_id * m : (col_id + 1) * m] = temp.astype(onp.float32)
	t2 = time.time()
	print('Test Col index %d, took %f'%(col_id, t2 - t1), file=_file)

file_name = directory + "/" + 'test_ntk_%d.npy'%(row_id)
np.save(file_name, K)
_file.close()
