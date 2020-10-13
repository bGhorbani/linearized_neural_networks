"""
This code is used to fit KRR. Note that KRR for large
datasets can require a lot of memory. For FMNIST models
we used 64GB and for synthetic datasets (with n = 10^5),
we used 128GB of RAM. This code is written for Python 2.7.13. 

Inputs:

experiment_name: The name of the experiment (used for record keeping).

param_1: overloaded argument: 
	When kernel name = polynomial: Power of the polynomial kernel 
	When dataset == SYNTH: the number of observations
	otherwise, the number of layers for ntk kernel

p2_ind:	The constant term in the polynomial kernel.

kernel_name: The name of the kernel: ntk, gp (ReLU RF), poly (polynomial).

job_id: Job id (used for record keeping).

dataset: The name of the dataset: 
	FMNIST (high-frequency noise)/ NFMNIST (low-frequency noise)
	CIFAR10 (low-frequency noise)/ CIFAR2 (high-frequency noise)
	SYNTH (synthetic data).

noise_index: The index of the noise level. An integer typically ranging from zero (no noise) to 14.
"""

from __future__ import print_function
import cPickle as pickle
import math
import numpy as np
import os 
import sys
import scipy.linalg as scl
import scipy.sparse as ss
import time
from preprocess import prep_data

experiment_name = sys.argv[1] 
param_1 = np.int(sys.argv[2]) 
p2_ind = np.int(sys.argv[3]) 
kernel_name = sys.argv[4]
job_id = np.int(sys.argv[5])
dataset = sys.argv[6]
noise_index = np.int(sys.argv[7])

# Read user provided directories:
user_dirs = {}
with open("./directories.txt") as f:
	for line in f:
		(key, val) = line.split()
		user_dirs[key] = val

# The hyper-parameters used for each dataset:
# expand: whether to one-hot-encode the labels
# mean: the mean of the labels to be removed before fitting
# reg_list: the grid of l_2 regularization intensities
# p2_grid: the grid of values used for the constant term in polynomial kernels
if dataset == 'CIFAR10':
	expand = True
	mean = 0.1
	reg_list = 10 ** np.linspace(-6, 1, num=20)
	p2_grid = 2 ** np.linspace(-3, 3, num=10)
elif dataset == 'SYNTH':
	expand = False
	mean = 0.0
	reg_list = 10 ** np.linspace(0, 6, num=10)
	p2_grid = 2 ** np.linspace(-3, 3, num=10)
elif dataset == 'CIFAR2':
	expand = False
	mean = 0.5
	reg_list = 10 ** np.linspace(-2, 4, num=20) 
	# Added for RF
	reg_list = np.concatenate([reg_list, 10 ** np.linspace(4.2, 10, num=20)])	
	p2_grid = 2 ** np.linspace(-3, 3, num=10)	
elif dataset in ['FMNIST', 'NFMNIST']:
	expand = True
	mean = 0.1
	# Changed for the new kernels
	if kernel_name == 'ntk' and param_1 == 3:
		reg_list = 10 ** np.linspace(-4, 3, num=20)
		print('Regularization Param Chosen for Three Layer NTK')
	else:
		# Changed Base Case	
		reg_list = 10 ** np.linspace(-1, 5, num=20) 
	p2_grid = 2 ** np.linspace(-3, 3, num=10)
else:
	print('Dataset not recognized')
	expand = False
	mean = 0.0
	reg_list = 10 ** np.linspace(0, 6, num=20)
	p2_grid = 2 ** np.linspace(-3, 3, num=10)

param_2 = p2_grid[p2_ind]
# Directory used for saving the KRR results
directory = user_dirs['rkhs_dir'] + '%s_%s_%s_%d_%d_%d_%d'%(experiment_name, dataset, kernel_name, param_1, p2_ind, noise_index, job_id)
if not os.path.exists(directory):
	os.makedirs(directory)

fileName = directory + "/" + 'log_file.txt'
_file = open(fileName, 'w', buffering=1)
print('Arguments:', file=_file)
print('The noise index is %d'%(noise_index), file=_file)
print('The kernel hyper_param is %d, %d'%(param_1, p2_ind), file=_file)
print('Kernel type is: %s'%(kernel_name), file=_file)
print('Numpy version %s'%(np.version.version), file=_file)
print('Scipy version %s'%(scl.__version__), file=_file)
print('=========', file=_file)

def NTK2(X, Z):
	"""This function computes NTK kernel for two-layer ReLU neural networks via
	an analytic formula.

	Input:
	X: d times n_1 matrix, where d is the feature dimension and n_i are # obs.
	Z: d times n_2 matrix, where d is the feature dimension and n_i are # obs.

	output:
	C: The kernel matrix of size n_1 times n_2.
	"""
	pi = math.pi
	assert X.shape[0] == Z.shape[0]
	# X is sized d \times n
	nx = np.linalg.norm(X, axis=0, keepdims=True)
	nx = nx.T    
	nz = np.linalg.norm(Z, axis=0, keepdims=True)    

	C = np.dot(X.T, Z) #n_1 * n_2
	C = np.multiply(C, (nx ** -1))
	C = np.multiply(C, (nz ** -1))
	# Fixing numerical mistakes
	C = np.minimum(C, 1.0)
	C = np.maximum(C, -1.0)			

	C = np.multiply(1.0 - np.arccos(C) / pi, C) + np.sqrt(1 - np.power(C, 2)) / (2 * pi)
	C = np.multiply(nx, np.multiply(C, nz))
	return C

def RFK2(X, Z):
	"""This function computes RF kernel for two-layer ReLU neural networks via
	an analytic formula.

	Input:
	X: d times n_1 matrix, where d is the feature dimension and n_i are # obs.
	Z: d times n_2 matrix, where d is the feature dimension and n_i are # obs.

	output:
	C: The kernel matrix of size n_1 times n_2.
	"""
	pi = math.pi
	assert X.shape[0] == Z.shape[0]
	# X is sized d \times n
	nx = np.linalg.norm(X, axis=0, keepdims=True)
	nx = nx.T    
	nz = np.linalg.norm(Z, axis=0, keepdims=True)    

	C = np.dot(X.T, Z) #n_1 * n_2
	C = np.multiply(C, (nx ** -1))
	C = np.multiply(C, (nz ** -1))
	# Fixing numerical mistakes
	C = np.minimum(C, 1.0)
	C = np.maximum(C, -1.0)
	C = np.multiply(np.arcsin(C), C) / pi + C / 2.0 + np.sqrt(1 - np.power(C, 2)) / pi
	C = 0.5 * np.multiply(nx, np.multiply(C, nz))
	return C

def compute_kernel(name, hyper_param):
	"""This function computes the test and the training kernels.
	Inputs:
		name: Kernel name.
		hyper_param: Kernel hyper-parameters.
	Outputs:
		Training Kernel: n times n np.float32 matrix.
		Test Kernel: nt times n np.float32 matrix.
		ytrain: vector of training labels. n times 1 np.float32.
		ytest: vector of test labels. nt times 1 np.float32.
	"""	
	X, ytrain, Xtest, ytest = prep_data(dataset, False, noise_index)
	nt = Xtest.shape[0]
	n = X.shape[0]
	d = X.shape[1] + 0.0	
	# Read precomputed CNTK kernel and form the kernel matrix 
	if dataset == 'CIFAR10' and name == 'ntk':
		K = np.zeros((n, n), dtype=np.float32)
		KT = np.zeros((n, nt), dtype=np.float32)
		main_dir = user_dirs['cntk_dir'] + 'LFGaussian_CIFAR10_Myrtle_%d/'%(noise_index)
		m = 200
		count = 250
		for i in range(count):
			K[(m * i):(m * (i + 1)), :] = np.load(main_dir + 'train_ntk_%d.npy'%(i))
			KT[(m * i):(m * (i + 1)), :] = np.load(main_dir + 'test_ntk_%d.npy'%(i))
		KT = KT.T
		for i in range(n):
			K[:, i] = K[i, :]
	elif dataset == 'SYNTH'	and name == 'ntk':
		n = hyper_param[0]		
		K = np.load(user_dirs['synth_dir'] + 'NTK_TRAIN_%d.npy'%(noise_index))
		K = K[:n, :n]
		KT = np.load(user_dirs['synth_dir'] + 'NTK_TEST_%d.npy'%(noise_index))
		KT = KT[:, :n]
		ytrain = ytrain[:n, :]
	elif name == 'polynomial':
		print('Request to use degree %d polynomial kernel with intercept %f'%(hyper_param[0], hyper_param[1]), file=_file)
		p = hyper_param[0]
		intercept = hyper_param[1]
		intercept = intercept.astype(np.float32)
		K =  (np.power(intercept + np.dot(X, X.T) / np.sqrt(d), p))
		KT = (np.power(intercept + np.dot(Xtest, X.T) / np.sqrt(d), p))
	elif name == 'rf':
		directory = user_dirs['rf_dir'] + 'RF_Kernel_noise_%d'%(noise_index)
		name = directory + '/RF_Kernel_Train_N_4200000.npy'
		K = np.load(name)
		name = directory + '/RF_Kernel_Test_N_4200000.npy'
		KT = np.load(name)
		
		K = K.astype(np.float32)
		KT = KT.astype(np.float32)		
	elif name == 'ntk':
		# ntk KRR		
		layers = hyper_param[0]		
		if layers < 3:
			# For two-layers networks, compute the kernel directly		
			K = NTK2(X.T, X.T)
			KT = NTK2(Xtest.T, X.T)
		else:
			# For multilayer networks, read it from the disk
			K = np.load(user_dirs['ntk_dir'] + 'Train_NTK_%d_layers_%d_NFMNIST.npy'%(noise_index, hyper_param[0]))
			KT = np.load(user_dirs['ntk_dir'] + 'Test_NTK_%d_layers_%d_NFMNIST.npy'%(noise_index, hyper_param[0]))
	elif name == 'gp':
		# ReLU RF KRR 
		K = RFK2(X.T, X.T)	
		KT = RFK2(Xtest.T, X.T)
	else:
		raise Exception('Non-valid Kernel')
	assert K.shape[0] == n and K.shape[1] == n
	assert K.dtype == np.float32
	assert KT.shape[0] == nt and KT.shape[1] == n
	assert KT.dtype == np.float32
	return (K, KT, ytrain, ytest)

def compute_accuracy(true_labels, preds):
	"""This function computes the classification accuracy of the vector
	preds. """
	if true_labels.shape[1] == 1:
		n = len(true_labels)
		true_labels = true_labels.reshape((n, 1))
		preds = preds.reshape((n, 1))		
		preds = preds > 0
		inds = true_labels > 0
		return np.mean(preds == inds)	
	groundTruth = np.argmax(true_labels, axis=1).astype(np.int32)
	predictions = np.argmax(preds, axis=1).astype(np.int32)
	return np.mean(groundTruth == predictions)

N = len(reg_list)
errors = np.zeros((N, 4)) #Train Loss, Train Accuracy, Test Loss, Test Accuracy
K, KT, ytrain, ytest = compute_kernel(kernel_name, [param_1, param_2])
print('Kernel Computation Done!', file=_file)

n = len(ytrain)
nt = len(ytest)
if expand:
	# Expand the labels
	maxVal = np.int(ytrain.max())
	Y = np.zeros((n, maxVal + 1), dtype=np.float32)
	Y[np.arange(n), ytrain[:, 0].astype(np.int32)] = 1.0

	YT = np.zeros((nt, maxVal + 1), dtype=np.float32)
	YT[np.arange(nt), ytest[:, 0].astype(np.int32)] = 1.0
	Y = Y - mean
	YT = YT - mean
	assert Y.dtype == np.float32
	assert YT.dtype == np.float32
	ytrain = Y
	ytest = YT
	mean = 0.0

for j in range(N):
	reg = reg_list[j].astype(np.float32)
	print('The regularization parameter is %f corresponding to the index %d'%(reg, j), file=_file)
	RK = K + reg * np.eye(n, dtype=np.float32)	
	assert K.dtype == np.float32
	assert RK.dtype == np.float32
	print('Solving kernel regression with %d observations and regularization param %f'%(n, reg), file=_file)
	t1 = time.time()
	if expand:		
		x = scl.solve(RK, ytrain, assume_a='sym')
	else:
		cg = ss.linalg.cg(RK, ytrain[:, 0] - mean, maxiter=400, atol=1e-4, tol=1e-4)
		x = np.copy(cg[0]).reshape((n, 1))
	t2 = time.time()
	print('iteration took %f seconds'%(t2 - t1), file=_file)
	yhat = np.dot(K, x) + mean
	preds = np.dot(KT, x) + mean
	errors[j, 0] = np.linalg.norm(ytrain - yhat) ** 2 / (len(ytrain) + 0.0)
	errors[j, 2] = np.linalg.norm(ytest - preds) ** 2 / (len(ytest) + 0.0)
	errors[j, 1] = compute_accuracy(ytrain - mean, yhat - mean)
	errors[j, 3] = compute_accuracy(ytest - mean, preds - mean)
	print('Training Error is %f'%(errors[j, 0]), file=_file)
	print('Test Error is %f'%(errors[j, 2]), file=_file)
	print('Training Accuracy is %f'%(errors[j, 1]), file=_file)
	print('Test Accuracy is %f'%(errors[j, 3]), file=_file)

_file.close()
save_dict = {'errors': errors, 'reg_params': [reg_list, p2_grid], 'hyper-params': [param_1, param_2, dataset, kernel_name], 'x': x, 'yhat': yhat, 'ythat': preds}
file_name = directory + '/stats.pkl'
with open(file_name, 'wb') as output:    
	pickle.dump(save_dict, output, -1)