"""
This code uses TensorFlow to fit NNs, CNNs,
Random Feature Models and Neural Tangent Models to the data.

The code was written for python/2.7.13 and TensorFlow v 1.12
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, './linear_algebra/')
from optimization_utils import OptimizationUtils
sys.path.insert(0, './model_zoo/')
from neural_networks import TwoLayerReluNT, RF, FullyConnected, Myrtle

from rf_optimizer import RF_Optimizer
from preprocess import prep_data

import scipy.optimize as sco
import scipy.linalg as scl
import scipy.sparse as ss
import time
import os 
import cPickle as pickle

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for SGD')
flags.DEFINE_float('drop_rate', 0.0, 'Dropout rate')
flags.DEFINE_integer('reg_index', 0, 'index for regularization coefficient')
flags.DEFINE_integer('max_batch_size', 10000, 'Maximum allowed batch size')
flags.DEFINE_integer('num_units', 4096, 'The number of hidden units')
flags.DEFINE_integer('num_layers', 1, 'The number of layers')
flags.DEFINE_integer('job_id', -1, 'Unique job id assigned by the cluster')

flags.DEFINE_integer('max_cg_iters', 750, 'Maximum number of CG / SGD iterations')
flags.DEFINE_integer('max_ncg_iters', 0, 'Maximum number of ncg iterations')

flags.DEFINE_integer('noise_ind', 0, 'Index for the noise added to the data')
flags.DEFINE_enum('loss', 'square', ['square', 'cross_entropy'], 'The loss used for training the model')
flags.DEFINE_string('exp_name', 'test', 'The name of the experiment')
flags.DEFINE_enum('model', '2layerNTK', ['2layerNTK', 'rf', 'FullyConnected', 'Myrtle'], 'The model used for fitting the data')
flags.DEFINE_enum('dataset', 'FMNIST', ['NFMNIST', 'FMNIST', 'SYNTH', 'CIFAR2', 'CIFAR10'], 'The dataset used for the experiment.')

#FLAGS = flags.FLAGS

class Experiment(object):
	def __init__(self):
		FLAGS = tf.app.flags.FLAGS
		print(FLAGS)
		self._dataset = FLAGS.dataset
		self._model_name = FLAGS.model
		self._params = {}		
		if FLAGS.job_id < 0:
			FLAGS.job_id = np.random.randint(0, 10 ** 5)
		self._mkdir(FLAGS.exp_name, FLAGS.model, FLAGS.job_id, FLAGS.num_units, FLAGS.dataset, FLAGS)
		print('Job ID is %d'%(FLAGS.job_id), file=self._file)

		# process flags to form the directory and choose the model 
		self._m = FLAGS.num_layers
		model_fn = self._choose_model(FLAGS.model, FLAGS.reg_index)		
		self._import_data(FLAGS.dataset, FLAGS.model, FLAGS.noise_ind)

		self._params['loss_type'] = FLAGS.loss		
		if FLAGS.model in ['FullyConnected']:
			self._params['dropout'] = FLAGS.drop_rate
		else:
			self._params['dropout'] = None
		max_batch_size = FLAGS.max_batch_size

		# dataset specific hyper-parameters
		if FLAGS.dataset == 'SYNTH':
			self._params['mean'] = 0.0			    	    		
    			self._params['expandFinal'] = False
			self._params['num_classes'] = 1
			self._params['max_class'] = 1			
		elif FLAGS.dataset == 'CIFAR2':
			self._params['mean'] = 0.5
    			self._params['expandFinal'] = False
			self._params['num_classes'] = 1
			self._params['max_class'] = 1			
			max_batch_size = 2000
		else:
			self._params['mean'] = 0.0
			self._params['expandFinal'] = True
			self._params['num_classes'] = 10
			self._params['max_class'] = 9	
			max_batch_size = np.minimum(10000, max_batch_size)
			
		# Hyper-parameters
		self._n = self._X.shape[0]
		self._d = self._X.shape[1]		
		self._N = FLAGS.num_units		
		self._tol = 2e-2		
		graph = tf.Graph()
		tf.reset_default_graph()
		with graph.as_default():
			tf.set_random_seed(91)
			self._batch_size = tf.placeholder(tf.int64, shape=[])
			x, y, iterator = self._data_pipeline(FLAGS.model)			
			# Note that when training CNNs, 'num_hidden' is used for
			# the number of channcel
			self._params['num_hiddens'] = self._N
			self._params['num_layers'] = self._m
			self._params['feature_dims'] = self._d			
			self._params['train_vars'] = True
			self._params['filter_size'] = 3
			# For the scenario where CG is not available
			if self._model_name in ['rf', '2layerNTK']:
				self._params['optimizer'] = 'adam'
			else:
				self._params['optimizer'] = 'mom'				
			self._params['include_bias'] = True
			self._params['reg_param'] = self._model_reg			
			print('Model Reg is %f'%(self._model_reg))
			model = model_fn(x, y, self._params)			
			if FLAGS.model == 'rf':
				self._optim_tool = None 
				ShapeList = [np.prod(g.shape) for g in model._variables]
    				self._num_params = np.int(np.sum(ShapeList))
			else:
				self._optim_tool = OptimizationUtils(model._loss, model._preds, model._variables, y, self._optim_reg, self._n)
				self._num_params = self._optim_tool.num_params()
			print('The number of parameters is %d'%(self._num_params), file=self._file)
			# Train the network
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				if FLAGS.model == 'rf':	
					np.random.seed(1990)					
					W0 = np.random.normal(size=(self._d, self._N))
					norms = np.linalg.norm(W0, axis=0, keepdims=True)
					W0 = W0 / norms
					# Forming NTK approx
					sess.run(model._rf_assign, {model._tph: W0})
					del W0, norms	
				print('Initialized Variables', file=self._file)
				save_dict = self._optim_fun(model, sess, iterator, FLAGS.max_cg_iters, FLAGS.max_ncg_iters, FLAGS.learning_rate, max_batch_size)
				save_dict['train_loss'], save_dict['train_accuracy'] = self._record_stats(sess, 'train', model, iterator, max_batch_size)
				save_dict['test_loss'], save_dict['test_accuracy'] = self._record_stats(sess, 'test', model, iterator, max_batch_size)
		self._save_results(save_dict, FLAGS)

	def _import_data(self, dataset, model, dim):
		self._X, self._Y, self._Xtest, self._Ytest = prep_data(dataset, model == 'CNN', dim)		

	def _save_results(self, save_dict, exp_flag):
		save_dict['num_param'] = self._num_params		
		save_dict['reg_coeff'] = (self._optim_reg + self._model_reg)
		print('Regularization Loss is %f'%(save_dict['reg_loss']), file=self._file)		
		print('Train loss is %f'%(save_dict['train_loss']), file=self._file)            		
		print('Test loss is %f'%(save_dict['test_loss']), file=self._file)		
		temp = exp_flag.flag_values_dict()
		print(temp, file=self._file)
		self._file.close()
		for key in temp.keys():
			save_dict[key] = temp[key]
		fileName = self._directory + "/" + 'stats.pkl'
		with open(fileName, 'wb') as output:    
			pickle.dump(save_dict, output, -1)

	def _data_pipeline(self, model):
		if self._model_name in ['Myrtle']:
			self._features_placeholder = tf.placeholder(self._X.dtype, (None, self._X.shape[1], self._X.shape[2], self._X.shape[3]))
		else:
			self._features_placeholder = tf.placeholder(self._X.dtype, (None, self._X.shape[1]))
		self._labels_placeholder = tf.placeholder(self._Y.dtype, (None, 1))
		if self._model_name in ['rf', '2layerNTK']:
			dataset = tf.data.Dataset.from_tensor_slices((self._features_placeholder, self._labels_placeholder)).repeat(1).batch(self._batch_size, drop_remainder=True)
		else:
			dataset = tf.data.Dataset.from_tensor_slices((self._features_placeholder, self._labels_placeholder)).shuffle(128).repeat(1).batch(self._batch_size, drop_remainder=True)
		dataset = dataset.prefetch(buffer_size=16)
		iterator = dataset.make_initializable_iterator()
		x, y = iterator.get_next()
		return (x, y, iterator)

	def _record_stats(self, sess, mode, model, iterator, batch_size):
		"""This function calculates training / test loss and accuracies over the full dataset."""		
		acc_list = []
		loss_list = []
		drop_rate = 0.0
		if mode == 'train':
			batch_size = min(batch_size, self._Y.shape[0])
			sess_dict = {self._features_placeholder: self._X, self._labels_placeholder: self._Y, self._batch_size: batch_size}
			loss_measure = model._loss
			drop_rate = self._params['dropout']
		else:
			batch_size = 100 #min(batch_size, self._Ytest.shape[0])
			sess_dict = {self._features_placeholder: self._Xtest, self._labels_placeholder: self._Ytest, self._batch_size: batch_size}			
			loss_measure = model._pure_loss
			drop_rate = 0.0
		sess.run(iterator.initializer, feed_dict=sess_dict)
		end_of_data = False
		while not end_of_data:
			try:
				if self._params['dropout'] is None:
					temp_loss, temp_accuracy = sess.run([loss_measure, model._accuracy])
				else:
					temp_loss, temp_accuracy = sess.run([loss_measure, model._accuracy], feed_dict={model._drop_rate: drop_rate})
				loss_list.append(temp_loss)
				acc_list.append(temp_accuracy)
			except tf.errors.OutOfRangeError:
				end_of_data = True  
		return (np.mean(loss_list), np.mean(acc_list))

	def _rf_optimizer(self, model, sess, iterator, num_iters, ncg_iters, lr, max_batch_size):
		""" This function uses the RF_Optimizer class to train large RF models with CG. While both _rf_optimizer
		 and _ls_optimizer can be used to train the models with CG, RF_Optimizer is more optimized and hence more
		 suitable for training large RF models. """

		sess_dict = {self._features_placeholder: self._X, self._labels_placeholder: self._Y, self._batch_size: max_batch_size}
		print('Batch Size is %d'%(max_batch_size), file=self._file)
		n = self._n
		self._optim_tool = RF_Optimizer(model, sess, iterator.initializer, sess_dict, self._optim_reg, n, self._labels_placeholder, True)
		print('Optimizer Created', file=self._file)
		xerr = np.zeros((self._num_params,))    						
		def matvec(x):
			return self._optim_tool.Hv(x)
	    
		rep_stats = np.zeros((2,))
		rep_stats[1] = 10000
		cg_error_hist = []
		def report(xk):                
			if rep_stats[0] % 30 == 0:
				error = self._optim_tool.fun(xk)
				cg_error_hist.append(error)
				print('Iteration: %d, Curr Error: %f, Min Error: %f'%(rep_stats[0], error, rep_stats[1]), file=self._file)
				if error < rep_stats[1]:
					rep_stats[1] = error 
					xerr[:] = np.copy(xk)
			rep_stats[0] += 1 

		b = self._optim_tool.Atx() * 2.0 / np.sqrt(n + 0.0)
		print('b vector computed', file=self._file)
		print(np.linalg.norm(b), file=self._file)
		
		t1 = time.time()
		lin_op = ss.linalg.LinearOperator((self._num_params, self._num_params), matvec=matvec, rmatvec=matvec, dtype=np.float32)
		cg = ss.linalg.cg(lin_op, b, maxiter=num_iters, atol=1e-3, tol=1e-4, callback=report)
		t2 = time.time()  
		print('Initial CG took %f seconds'%(t2 - t1), file=self._file)
		x0 = np.copy(cg[0])
		first_stage_loss = self._optim_tool.fun(x0)    
		print('Loss for the final CG iteration is %f'%(first_stage_loss), file=self._file)
		print('Loss after initial CG is %f'%(first_stage_loss), file=self._file)
		save_dict = {}
		save_dict['cg_hist'] = cg_error_hist
		save_dict['res_norm'] = None
		save_dict['Message'] = 'Optimization terminated after CG.'
		save_dict['x'] = x0
		save_dict['reg_loss'] = self._optim_reg * np.linalg.norm(save_dict['x']) ** 2
		return save_dict

	def _ls_optimizer(self, model, sess, iterator, num_iters, ncg_iters, lr, max_batch_size):
		""" This function uses TensorFlow capabilities to train a TwoLayerReluNT model with CG."""
		xerr = np.zeros((self._num_params,))    		
		sess_dict = {self._features_placeholder: self._X, self._labels_placeholder: self._Y, self._batch_size: max_batch_size}
		n = self._n
		def fun(x):
			sess.run(iterator.initializer, feed_dict=sess_dict)
			val, reg = self._optim_tool.loss(x, sess)
			return val + reg			
	        
	        def Atx(x):
	                # Note x is n dimensional
	                assert len(x) == n
	                new_dict = {self._features_placeholder: self._X, self._labels_placeholder: x.reshape(n, 1), self._batch_size: max_batch_size}    
	                sess.run(iterator.initializer, feed_dict=new_dict)
	                atx = self._optim_tool.ATx(sess)	        
	                return atx[:, 0]

		def hv(x, p):
			"""Warning: This function assume that the Hessian is constant."""        
			sess.run(iterator.initializer, feed_dict=sess_dict)
			vec = self._optim_tool.Hv(None, p, sess)
			return vec[:, 0]

		def grad(x):        
			sess.run(iterator.initializer, feed_dict=sess_dict)
			gradient = self._optim_tool.gradient(x, sess)
			return gradient[:, 0]
	    
		def matvec(x):
			return hv(None, x)
	    
		rep_stats = np.zeros((2,))
		rep_stats[1] = 10000
		cg_error_hist = []
		def report(xk):                
			if rep_stats[0] % 30 == 0:
				error = fun(xk)
				cg_error_hist.append(error)
				print('Iteration: %d, Curr Error: %f, Min Error: %f'%(rep_stats[0], error, rep_stats[1]), file=self._file)
				if error < rep_stats[1]:
					rep_stats[1] = error 
					xerr[:] = np.copy(xk)
			rep_stats[0] += 1 

		b = Atx(self._Y[:, 0]) * 2.0 / np.sqrt(n + 0.0)
		t1 = time.time()
		lin_op = ss.linalg.LinearOperator((self._num_params, self._num_params), matvec=matvec, rmatvec=matvec, dtype=np.float32)                
		#cg = ss.linalg.cg(lin_op, b, maxiter=num_iters, atol=1e-4, tol=1e-5, callback=report)
		cg = ss.linalg.cg(lin_op, b, maxiter=num_iters, atol=1e-3, tol=1e-4, callback=report)
		t2 = time.time()  
		print('Initial CG took %f seconds'%(t2 - t1), file=self._file)
		x0 = np.copy(cg[0])
		first_stage_loss = fun(x0)    
		print('Loss for the final CG iteration is %f'%(first_stage_loss), file=self._file)
		if first_stage_loss > rep_stats[1]:
			first_stage_loss = rep_stats[1]
			x0 = np.copy(xerr)
		print('Loss after initial CG is %f'%(first_stage_loss), file=self._file)
		save_dict = {}
		save_dict['cg_hist'] = cg_error_hist
		save_dict['res_norm'] = None
		save_dict['x0'] = x0
		if first_stage_loss < self._tol or ncg_iters == 0:
			print('Tolerance achieved', file=self._file)
			save_dict['Message'] = 'Optimization terminated after CG.'
			save_dict['x'] = x0
			save_dict['grad'] = np.linalg.norm(grad(x0))
		else:
			t1 = time.time()
			optim_obj = sco.minimize(fun, x0, method='trust-ncg', jac=grad, hessp=hv, options={'disp':True, 'maxiter': ncg_iters})                
			t2 = time.time()
			print('trust-ncg took %f seconds'%(t2 - t1), file=self._file)        
			print('Loss after trust-ncg is %f'%(fun(optim_obj['x'])), file=self._file)
			save_dict['Message'] = optim_obj['message']
			save_dict['x'] = optim_obj['x']
			save_dict['grad'] = np.linalg.norm(optim_obj['jac'])
		save_dict['reg_loss'] = self._optim_tool._reg_param * np.linalg.norm(save_dict['x']) ** 2
		return save_dict

	def _fc_optimizer(self, model, sess, iterator, sgd_iters, ncg_iters, lr, max_batch_size):
		""" This function is used for training the models with first-order methods."""
	        def lr_fun(t, max_iters):
			pi = 3.141592
			warm_up = 15
			if t < warm_up:
				return lr * 2 * t / (warm_up + 0.0)
			else:          
				n = t - warm_up
				N = max_iters - warm_up + 0.0
				return np.maximum(lr * (1 + np.cos(n * pi / N)), lr / 15.0)

        	def batch_fun(t):
        		if self._model_name in ['rf', '2layerNTK']:
        			return max_batch_size     
			if self._model_name in ['Myrtle']:
				return 128
        		if self._dataset == 'CIFAR2':
        			return 250
			if self._dataset in ['FMNIST', 'NFMNIST']:
	            		if t < 15:                
	                		return 500
        	    		return 1000
            		if t < 15:                
                		return 512
            		return 1024

	        epoch = 0
	        train = np.zeros((sgd_iters, 3))
		test = np.zeros((sgd_iters, 2))		
		sess_dict = {self._features_placeholder: self._X, self._labels_placeholder: self._Y, self._batch_size: batch_fun(epoch)}
	        sess.run(iterator.initializer, feed_dict=sess_dict)
	        errs = []
	        accs = []
	        regLoss = []
	        while epoch < sgd_iters:
			try:                
		                _, err, acc, regL = sess.run([model._train_op, model._loss, model._accuracy, model._reg_loss],\
		                	feed_dict={model._lr: lr_fun(epoch, sgd_iters), model._drop_rate: self._params['dropout']})
		                errs.append(err)
		                accs.append(acc)
		                regLoss.append(regL)
			except tf.errors.OutOfRangeError:          
		                train[epoch, 0] = np.mean(errs)
		                train[epoch, 1] = np.mean(accs)
		                train[epoch, 2] = np.mean(regLoss)
		                errs = []
		                accs = []
		                regLoss = []
		                sess_dict = {self._features_placeholder: self._Xtest, self._labels_placeholder: self._Ytest, self._batch_size: max_batch_size}
		                sess.run(iterator.initializer, feed_dict=sess_dict)	                	                
		                test[epoch, 0], test[epoch, 1] = sess.run([model._pure_loss, model._accuracy], feed_dict={model._drop_rate: 0.0})
		                print('epoch: %d, Train loss: %f, Test loss: %f, Train Accuracy: %f, Test Accuracy: %f'%(\
		                	epoch, train[epoch, 0], test[epoch, 0], train[epoch, 1], test[epoch, 1]), file=self._file)
		                epoch += 1
		                sess_dict = {self._features_placeholder: self._X, self._labels_placeholder: self._Y, self._batch_size: batch_fun(epoch)}
		                sess.run(iterator.initializer, feed_dict=sess_dict)
	        x0 = sess.run(model._variables)
		save_dict = {}
		save_dict['train_hist'] = train
		save_dict['test_hist'] = test	
		save_dict['x0'] = x0				
		# Saving final stats
	        save_dict['reg_loss'] = sess.run(model._reg_loss)	        	        
	        return save_dict

	def _choose_model(self, model, reg_ind):
		""" This function adjusts the hyper-parameters based on the dataset / model to be fitted."""
		self._params['form_train_op'] = False			
		self._optim_fun = self._ls_optimizer
		self._optim_reg = None
		if model == 'rf':
			self._optim_fun = self._rf_optimizer
			model_fn = RF
			if self._dataset == 'SYNTH':	
				reg_list = 10 ** np.linspace(-5, 2, num=10)
			elif self._dataset in ['CIFAR10.Gray', 'FMNIST', 'NFMNIST']:
				reg_list = 10 ** np.linspace(-5, 3, num=20)
				self._params['form_train_op'] = True
				self._optim_fun = self._fc_optimizer
				self._optim_reg = 0
			else:
				reg_list = 10 ** np.linspace(-4, 4, num=20)
		elif model == '2layerNTK':
			model_fn = TwoLayerReluNT
			if self._dataset == 'SYNTH':
				reg_list = 10 ** np.linspace(-4, 2, num=10)
			elif self._dataset in ['CIFAR10.Gray', 'FMNIST', 'NFMNIST']:
				reg_list = 10 ** np.linspace(-5, 3, num=20)
				self._params['form_train_op'] = True
				self._optim_fun = self._fc_optimizer
				self._optim_reg = 0
			else:
				reg_list = 10 ** np.linspace(-4, 4, num=20)
		elif model == 'Myrtle':
			reg_list = 10 ** np.linspace(-5, -2, num=10)
			self._params['form_train_op'] = True
			self._optim_fun = self._fc_optimizer
			model_fn = Myrtle
			self._optim_reg = 0
		elif model == 'FullyConnected':
			if self._dataset == 'SYNTH':
				reg_list = 10 ** np.linspace(-8, -5, num=15)
				reg_list = np.concatenate([reg_list, 10 ** np.linspace(-4.8, -4, num=5)])
				reg_list = np.concatenate([reg_list, 10 ** np.linspace(-3.8, -2, num=5)])
			elif self._dataset == 'CIFAR10.Gray':
				reg_list = 10 ** np.linspace(-5, -1, num=20)
			elif self._dataset in ['FMNIST', 'NFMNIST']:
				reg_list = 10 ** np.linspace(-6, -2, num=20)
				if self._m == 2:
					reg_list = 10 ** np.linspace(-7, -5, num=10) # Shuffle MNIST 2 layer
					print('Hyper-parameters Chosen for two hidden layers')
			else:
				reg_list = 10 ** np.linspace(-6, -2, num=20)
			self._params['form_train_op'] = True
			self._optim_fun = self._fc_optimizer
			model_fn = FullyConnected
			self._optim_reg = 0		
		else:
			raise Exception('Model is not recognized')		
		# Choose the regularization format
		if self._optim_reg is None:
			self._optim_reg = reg_list[reg_ind]
			self._model_reg = 0
		else:
			self._optim_reg = 0
			self._model_reg = reg_list[reg_ind] * 2.0
		print('Regularization adjusted to %f'%(self._optim_reg + self._model_reg), file=self._file)
		return model_fn
			
	def _mkdir(self, exp_name, model_name, job_id, num_units, dataset, exp_flag):
		user_dirs = {}
		with open("./directories.txt") as f:
			for line in f:
				(key, val) = line.split()
				user_dirs[key] = val
		noise_id = exp_flag.noise_ind
		self._directory = user_dirs['nn_dir'] + '%s_%s_%d_%s_%d_%d'%(exp_name, model_name, num_units, dataset, noise_id, job_id)
		if not os.path.exists(self._directory):
			os.makedirs(self._directory)
		# Printing the arguments to the log-file
		fileName = self._directory + "/" + 'log_file.txt'
		self._file = open(fileName, 'w', buffering=1)
		print('Arguments:', file=self._file)
		print(exp_flag, file=self._file)
		temp = exp_flag.flag_values_dict()
		print(temp, file=self._file)
		print('=========', file=self._file)


experiment = Experiment()
print('The Experiment is Finished!')