"""This file contains the code for loading and preprocessing the data. 
Note that for the functions to run properly, the appropriate datasets should 
already be downloaded to the correct directory:

	CIFAR-10: Should be saved to ./datasets/cifar10py/
	FMNIST: Should be saved to ./datasets/FMNIST/initial_ds/
	synthetic data: Should be saved to ./datasets/synthetic/
 """
import numpy as np
import sys
import pickle as pickle
import time
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.fftpack import dct, idct
from skimage.color import rgb2gray

def unpickle(add):
	if '3.6.1' in sys.version:
		with open(add, 'rb') as fo:
			val = pickle.load(fo, encoding='latin1')
	else:		
		with open(add, 'rb') as fo:
			val = pickle.load(fo)
	return val

def cifarTrain():
	"""This function reads the CIFAR-10 training examples from the disk."""
	X = []
	Y = []
	for i in range(1, 6):
		data = unpickle('./datasets/cifar10py/data_batch_%d'%(i))
		tempX = data['data']
		n = tempX.shape[0]
		tempX = tempX.reshape((n, 3, 32, 32))
		tempX = tempX.transpose((0, 2, 3, 1))
		X.append(tempX)

		tempy = np.array(data['labels'])
		tempy = tempy.reshape((len(tempy), 1))
		Y.append(tempy)
	X = np.concatenate(X, axis=0)
	Y = np.concatenate(Y, axis=0)
	return (Y, X)

def cifarTest():
	"""This function reads the CIFAR-10 test examples from the disk."""
	data = unpickle('./datasets/cifar10py/test_batch')
	tempX = data['data']
	n = tempX.shape[0]
	tempX = tempX.reshape((n, 3, 32, 32))
	tempX = tempX.transpose((0, 2, 3, 1))

	tempy = np.array(data['labels'])
	tempy = tempy.reshape((len(tempy), 1))
	return (tempy, tempX)

def cifar_input(mode, greyscale, CNN):
	"""This function preprocesses the CIFAR-10 data. 
	Inputs:
		mode: 'train' or 'test'
		greyscale: boolean. If true the images are converted to greyscale.
		CNN: boolean. If false, the images are flattened. """
	if mode == 'train':
		Y, X = cifarTrain()
	else:
		Y, X = cifarTest()        
	
	n = len(Y)
	n1 = 32
	n2 = 32
	nc = 3
	if greyscale:
		nc = 1    
	Z = np.zeros((n, n1, n2, nc))
	d = n1 * n2 * nc
	for i in range(n):
		image = X[i]
		if greyscale:
			image = rgb2gray(image)
		image = image - np.mean(image)    
		image = image / np.linalg.norm(image.flatten()) * np.sqrt(d)    
		if greyscale:
			Z[i, :, :, 0] = image
		else:
			Z[i] = image
	return (Y, Z)

def addHFNoise(X, mfilter, multiplier):
	"""This function adds Gaussian noise to frequencies identified by mfilter.
	Inputs:
		X: A data array of size n x d. 
		mfilter: A matrix of {0, 1} of size sqrt(d) x sqrt(d).
		multiplier: A positive constant corresponding to noise to signal ratio.
	Outputs:
		Z: A data array of noisy flattened images of size n x d."""
	n = X.shape[0]
	d = X.shape[1]
	n1 = np.int(np.sqrt(d))
	Z = np.zeros((n, d))    
	for i in range(n):
		image = X[i, :] + 0.0
		image = image - np.mean(image)		
		image = image.reshape((n1, n1))
		freq = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

		noise = np.random.normal(size=(n1, n1))
		noise = np.multiply(noise, mfilter)
		noise = noise / np.linalg.norm(noise)
		noise = noise * (multiplier * np.linalg.norm(freq))
		noisy_freq = freq + noise
		newIm = idct(idct(noisy_freq, axis=0, norm='ortho'), axis=1, norm='ortho') 
		newIm = newIm / np.linalg.norm(newIm)
		Z[i, :] = newIm.flatten() * np.sqrt(d)
	return Z

def addTHNoise(X, thresh, covHalf, mu):
	"""This function adds low-frequency Gaussian noise to greyscale images.
	Inputs:
		X: A data array of size n x d.
		thresh: A frequency threshold (valid range [0, sqrt(d)]).
		covHalf: The square root of the noise covariance.
		mu: The mean of the noise.
	Outputs:
		Z: A data array of size n x d containing noise flattened images."""
	n = X.shape[0]
	d = X.shape[1]
	n1 = np.int(np.sqrt(d))
	Z = np.zeros((n, d))
	noise = np.random.normal(size=(n, d))
	noise = np.dot(noise, covHalf) + mu
	for i in range(n):
		image = X[i].reshape((n1, n1))
		image = image - np.mean(image)        
		image = image / np.linalg.norm(image)
		image = image * np.sqrt(d)
		freq = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
		freq[:thresh, :thresh] = noise[i, :].reshape((n1, n1))[:thresh, :thresh]
		newIm = idct(idct(freq, axis=0, norm='ortho'), axis=1, norm='ortho')         
		Z[i, :] = newIm.flatten()
	return Z

def addCifarNoise(X, thresh, Shalf, mu):
	"""This function adds low frequency noise to the Cifar10 images (As described in Appendix A.6).
	Inputs: 
		X: The data tensor with shape n x 32 x 32 x 3.
		thresh: Frequency threshold (valid range: [0, 32]).
		Shalf: Square root of the covariance matrix of the noise.
		mu: Mean of the noise. 
	Outputs:
		images: A tensor of n x 32 x 32 x 3 of noisy images. """
	n = X.shape[0]    
	n1 = X.shape[1]
	c = X.shape[3]
	d = n1 * n1 * c
	noise = np.random.normal(size=(n, d))
	noise = np.dot(noise, Shalf) + mu    
	Z = np.zeros((n, n1, n1, c))
	for i in range(n):
		noiseInstance = noise[i, :].reshape((n1, n1, c))
		for j in range(c):
			image = X[i, :, :, j]
			Z[i, :, :, j] = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
		Z[i, :thresh, :thresh, :] = noiseInstance[:thresh, :thresh, :]	
	images = np.zeros((n, n1, n1, c))
	for i in range(n):
		for j in range(c):
			newIm = idct(idct(Z[i, :, :, j], axis=0, norm='ortho'), axis=1, norm='ortho')         
			images[i, :, :, j] = newIm
	return images

def prep_data(dataset='FMNIST', CNN=False, noise_index=0):
	np.random.seed(2048)
	if dataset == 'SYNTH':
		X = np.load('./datasets/synthetic/X_train_anisotropic_1024_16_%d.npy'%(noise_index))
		Y = np.load('./datasets/synthetic/y_train_anisotropic_1024_16_%d.npy'%(noise_index))	
		YT = np.load('./datasets/synthetic/y_test_anisotropic_1024_16_%d.npy'%(noise_index))
		XT = np.load('./datasets/synthetic/X_test_anisotropic_1024_16_%d.npy'%(noise_index))
		assert Y.dtype == np.float32 and YT.dtype == np.float32
		assert X.dtype == np.float32 and XT.dtype == np.float32
		assert len(Y.shape) == 2 and len(YT.shape) == 2
		assert Y.shape[1] == 1 and YT.shape[1] == 1
	elif dataset == 'CIFAR2':
		eps = np.linspace(0, 3, num=15)
		Y, X = cifar_input('train', True, False)
		# Choosing cats and airplanes
		inds = [i for i in range(len(Y)) if Y[i, 0] in [3, 0]]
		Y = Y[inds, :]
		X = X[inds, :]
		for i in range(len(Y)):
			if Y[i, 0] == 3:
				Y[i, 0] = 0
			else:
				Y[i, 0] = 1
		YT, XT = cifar_input('test', True, False)		
		# Choosing cats and airplanes
		inds = [i for i in range(len(YT)) if YT[i, 0] in [3, 0]]
		YT = YT[inds, :]
		XT = XT[inds, :]
		for i in range(len(YT)):
			if YT[i, 0] == 3:
				YT[i, 0] = 0
			else:
				YT[i, 0] = 1
		d = X.shape[1]
		n1 = np.int(np.sqrt(d))
		cfilter = np.zeros((n1, n1))
		for i in range(n1):
			for j in range(n1):
				radius = (n1 - i - 1) ** 2 + (n1 - j - 1) ** 2
				radius = np.sqrt(radius)
				if radius <= (n1 - 1):
					cfilter[i, j] = 1.0

		X = addHFNoise(X, cfilter, eps[noise_index])
		XT = addHFNoise(XT, cfilter, eps[noise_index])	
	elif dataset == 'CIFAR10':
		threshs = [0, 4, 8, 12, 14, 17, 19, 20, 22, 24, 25, 27, 28, 29, 30]
		threshs = np.array(threshs)
		thresh = threshs[noise_index]
		print('Chosen Threshold is %d'%(thresh))
		Y, X0 = cifarTrain()
		YT, XT0 = cifarTest()

		n = X0.shape[0]
		n1 = X0.shape[1]
		c = X0.shape[3]
		d = n1 * n1 * c
		
		Z = np.zeros((n, d))
		freq = np.zeros((n1, n1, c))
		for i in range(n):
			for j in range(c):
				image = X0[i, :, :, j]
				freq[:, :, j] = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')        
			Z[i, :] = freq.flatten()

		mu = np.mean(Z, axis=0, keepdims=True)
		Z = Z - mu
		S = np.dot(Z.T, Z) / (n + 0.0)
		w, Q = np.linalg.eigh(S)
		wHalf = np.diag(w ** 0.5)
		Shalf = np.dot(Q, np.dot(wHalf, Q.T))

		X = addCifarNoise(X0, thresh, Shalf, mu)
		XT = addCifarNoise(XT0, thresh, Shalf, mu)
		
		# Computing Normalization Statistics
		mu = np.mean(X, axis=0, keepdims=True)
		mu = np.mean(mu, axis=1, keepdims=True)
		mu = np.mean(mu, axis=2, keepdims=True)

		sd = np.zeros((1, 1, 1, 3))
		for i in range(3):
			sd[0, 0, 0, i] = np.std(X[:, :, :, i])
		
		X = (X - mu) / sd
		XT = (XT - mu) / sd
	elif dataset == 'FMNIST':
		eps = np.linspace(0, 3, num=15)
		print('Noise strength is %f'%(eps[noise_index]))
		X = np.load('./datasets/FMNIST/initial_ds/X.npy')
		Y = np.load('./datasets/FMNIST/initial_ds/Y.npy')
		YT = np.load('./datasets/FMNIST/initial_ds/Yt.npy')
		XT = np.load('./datasets/FMNIST/initial_ds/Xt.npy')		
		assert len(Y.shape) == 2 and len(YT.shape) == 2
		assert Y.shape[1] == 1 and YT.shape[1] == 1

		n = X.shape[0]
		d = X.shape[1]
		n1 = np.int(np.sqrt(d))
		cfilter = np.zeros((n1, n1))
		for i in range(n1):
			for j in range(n1):
				radius = (n1 - i - 1) ** 2 + (n1 - j - 1) ** 2
				radius = np.sqrt(radius)
				if radius <= (n1 - 1):
					cfilter[i, j] = 1.0
		
		X = addHFNoise(X, cfilter, eps[noise_index])
		XT = addHFNoise(XT, cfilter, eps[noise_index])
	elif dataset == 'NFMNIST':
		thresholds = np.arange(1, 28, step=2)		
		thresh = thresholds[noise_index]
		print('Chosen Threshold is %d'%(thresh))
		X = np.load('./datasets/FMNIST/initial_ds/X.npy')
		Y = np.load('./datasets/FMNIST/initial_ds/Y.npy')
		YT = np.load('./datasets/FMNIST/initial_ds/Yt.npy')
		XT = np.load('./datasets/FMNIST/initial_ds/Xt.npy')		
		assert len(Y.shape) == 2 and len(YT.shape) == 2
		assert Y.shape[1] == 1 and YT.shape[1] == 1		
		n = X.shape[0]
		d = X.shape[1]
		n1 = np.int(np.sqrt(d))
		Z = np.zeros((n, d))
		for i in range(n):
			image = X[i].reshape((n1, n1))
			image = image - np.mean(image)             
			image = image / np.linalg.norm(image)
			image = image * np.sqrt(d)
			freq = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')        
			Z[i, :] = freq.flatten()
		mu = np.mean(Z, axis=0, keepdims=True)
		Z = Z - mu
		S = np.dot(Z.T, Z) / (n + 0.0)
		w, Q = np.linalg.eigh(S)
		wHalf = np.diag(w ** 0.5)
		Shalf = np.dot(Q, np.dot(wHalf, Q.T))
		X = addTHNoise(X, thresh, Shalf, mu)
		XT = addTHNoise(XT, thresh, Shalf, mu)		
	else:
		raise Exception('Unidentified dataset')
	X = X.astype(np.float32)
	Y = Y.astype(np.float32)
	XT = XT.astype(np.float32)
	YT = YT.astype(np.float32)
	return (X, Y, XT, YT)
