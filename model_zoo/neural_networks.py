# This file contains our implementations for the models trained in the paper.
from __future__ import print_function
import tensorflow as tf
import numpy as np

class MultiLayerNetwork(object):
	def __init__(self, data, labels, params):
		self._num_classes = params['num_classes']
		self._max_class = params['num_classes'] - 1
		self._lossType = params['loss_type']
		self._expandedFinalLayer = params['expandFinal']
		if self._lossType == 'cross_entropy':
			assert self._expandedFinalLayer		
		self._num_layers = params['num_layers']
		self._trainable = params['train_vars']
		self._optimizer = params['optimizer']
		self._reg_param = params['reg_param'] #L2-type regularization
		form_train_op = params['form_train_op']		
		self._reg_loss = 0
		self._global_step_variable = tf.Variable(0, name='global_step', trainable=False)
		if 'dropout' in params:
			self._drop_rate = tf.constant(0.0)
		else: 
			self._drop_rate = tf.constant(0.0)

		if not self._expandedFinalLayer:
			self._num_classes = 1

		self._labels = labels
		self._data = data
		self._variables = None
		self._accuracy = None
		self._reg_loss = None		
		self._mean = params['mean']
		if self._expandedFinalLayer:
			assert self._mean == 0
		self._pure_loss = None
		self._initVals = [] #Initialization values
		self._loss, self._preds = self._model(self._data, self._labels)		
		self._grads_vars = None
		self._train_op = None
		self._lr = tf.placeholder(dtype=tf.float32, name='learning_rate', shape=[])
		if form_train_op:
			self._form_train_op()

	def _fcLayer(self, x, num_hiddens, name, include_bias, trainable, std):
		assert len(x.shape) == 2
		d = int(x.shape[1])
		N = num_hiddens
		W0 = np.random.normal(size=(d, N)) * std #/ np.sqrt(d)
		with tf.variable_scope(name):			
			w = tf.get_variable(initializer=tf.constant_initializer(W0), name="W", shape=[d, N], dtype=tf.float32, trainable=trainable)
			if include_bias:
				b = tf.get_variable(initializer=tf.constant_initializer(0.01), name="bias",\
				 shape=[1, N], dtype=tf.float32, trainable=trainable)
			else:
				b = 0.0
		output = tf.matmul(x, w) + b
		output = tf.nn.relu(output)
		#output = tf.nn.tanh(output)
		output = tf.nn.dropout(output, keep_prob=1.0 - self._drop_rate)
		#output = tf.nn.sigmoid(output)
		return (output, W0)

	def _conv_layer(self, x, filter_size, n_in, n_out, name, trainable, strides=[1, 2], include_bias=True):
		assert x.shape[3] == n_in #n_in should be equal to the number of channels
		n = filter_size * filter_size * n_out
		with tf.variable_scope(name):
			W = tf.get_variable("W", shape=[filter_size, filter_size, n_in, n_out],\
				initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / (n + 0.0))), trainable=trainable)            
			y = tf.nn.conv2d(x, filter=W, strides=[1, strides[0], strides[0], 1], padding='SAME')
			if include_bias:
				b = tf.get_variable(initializer=tf.constant_initializer(0.01), name="bias",\
					shape=[1, 1, 1, n_out], dtype=tf.float32, trainable=trainable)
			else:
				b = 0
			if strides[1] > 0:
				y = tf.nn.max_pool(y, ksize=[1, strides[1], strides[1], 1], strides=[1, strides[1], strides[1], 1], padding='VALID')
		y = y + b
		y = tf.nn.relu(y)
		return y

	def _form_train_op(self):
		print('Warning: Using the generic train op')
		if self._optimizer == 'mom':
			opt = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.9)
		elif self._optimizer == 'sgd':
			opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
		elif self._optimizer == 'adam':
			opt = tf.train.AdamOptimizer(learning_rate=self._lr)
		else:
			raise Exception('Optimizer was not recognized!')
		self._grads_vars = opt.compute_gradients(self._loss, self._variables)
		self._train_op = opt.apply_gradients(self._grads_vars, name='train_step', global_step=self._global_step_variable)

	def _compute_accuracy(self, yhat, labels):
		if self._expandedFinalLayer:
			adjusted_label = tf.to_int32(labels[:, 0])
			class_pred = tf.argmax(yhat, axis=1, output_type=tf.int32)
			zero_one_loss_vec = tf.to_float(tf.equal(class_pred, adjusted_label))
			assert len(zero_one_loss_vec.shape) == 1
		else:			
			# Note this is only valid for two-class classification			
			adjusted_label = tf.to_int32(tf.sign(labels - self._mean))
			class_pred = tf.sign(yhat - self._mean)
			class_pred = tf.to_int32(class_pred)
			zero_one_loss_vec = tf.to_float(tf.equal(class_pred, adjusted_label))		
			assert zero_one_loss_vec.shape[1] == 1 and len(zero_one_loss_vec.shape) == 2
		self._accuracy = tf.reduce_mean(zero_one_loss_vec)

class TwoLayerReluNT(MultiLayerNetwork):
	def __init__(self, data, labels, params):		
		self._N = params['num_hiddens']
		self._d = int(data.shape[1])
		super(TwoLayerReluNT, self).__init__(data, labels, params)			
		
	def _model(self, data, labels):		
		num_classes = self._num_classes
		N = self._N
		d = self._d
		np.random.seed(1990)		
		W0 = np.random.normal(size=(d, N))
		norms = np.linalg.norm(W0, axis=0, keepdims=True)
		W0 = W0 / norms
		
		a0 = np.random.normal(size=(1, num_classes, N))
		# Forming NTK approx
		w = tf.constant(W0, name='fixed_w', shape=[d, N], dtype=tf.float32)	
		at = tf.get_variable(initializer=tf.constant_initializer(0.0),\
			name='layer2', shape=[N, num_classes], dtype=tf.float32)	
		c = tf.get_variable(initializer=tf.constant_initializer(0.0),\
			name='layer2_bias', shape=[1, num_classes], dtype=tf.float32)				
		z = tf.matmul(data, w)
		q = tf.nn.relu(z)
		RF = tf.matmul(q, at) + c # BS x num_classes
	    	
	    	G = tf.get_variable(initializer=tf.constant_initializer(0.0),\
	    		name='layer1_var', shape=[N, d], dtype=tf.float32)
	    	zero_one_mat = 0.5 * (tf.sign(z) + 1.0)  #batchsize x N
	    	expanded_zero_one_mat = tf.expand_dims(zero_one_mat, 2) #batchsize x N x 1
	    	expanded_zero_one_mat = tf.transpose(expanded_zero_one_mat, [0, 2, 1]) #batchsize x 1 x N    	
	    	U = tf.multiply(expanded_zero_one_mat, a0)
	    	assert U.shape[1] == num_classes and U.shape[2] == N
	    	#print(U.shape) #batchsize x num_classes * N
		q2 = tf.tensordot(U, G, axes=[[2], [0]]) # bs x num_class x d
		aux_data = tf.transpose(tf.expand_dims(data, 2), [0, 2, 1]) # bs x 1 x d
		NT = tf.reduce_sum(tf.multiply(q2, aux_data), axis=2, keepdims=False) #bs x num_class
		yhat = RF + NT	
		if self._lossType == 'cross_entropy':
			true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), num_classes)
			loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_labels, logits=yhat)		
		elif self._lossType == 'square':
			if self._expandedFinalLayer:
				true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), num_classes, dtype=tf.float32)
				loss_vec = (true_labels - yhat) ** 2	
				print("Shape before Summation")
				print(loss_vec.shape)			
				loss_vec = tf.reduce_sum(loss_vec, axis=1, keepdims=True)
				print("Shape after Summation")
				print(loss_vec.shape)
			else:
				loss_vec = (labels - yhat) ** 2			
		else:
			raise Exception('Loss type is not recognized')
		assert loss_vec.shape[1] == 1 and len(loss_vec.shape) == 2
		self._pure_loss = tf.reduce_mean(loss_vec)
		self._variables = tf.trainable_variables()
		self._reg_loss = tf.nn.l2_loss(G) #tf.constant(0.0)
		self._reg_loss = self._reg_param * self._reg_loss
		augmented_loss = self._pure_loss + self._reg_loss
		# Classification Accuracy	
		self._compute_accuracy(yhat, labels)		
		return (augmented_loss, yhat)

class RF(MultiLayerNetwork):
	def __init__(self, data, labels, params):
		self._num_classes = params['num_classes']
		self._N = params['num_hiddens']
		self._d = int(data.shape[1])
		self._tph = tf.placeholder(tf.float32, (self._d, self._N))
		super(RF, self).__init__(data, labels, params)

	def _model(self, data, labels):		
		num_classes = self._num_classes
		N = self._N
		d = self._d
		#np.random.seed(1990)
		##W0 = np.random.normal(size=(d, N)) / np.sqrt(d)
		#W0 = np.random.normal(size=(d, N))
		#norms = np.linalg.norm(W0, axis=0, keepdims=True)
		#W0 = W0 / norms
		## Forming NTK approx
		#w = tf.constant(W0, name='fixed_w', shape=[d, N], dtype=tf.float32)
		#del W0, norms
		w = tf.get_variable(name='fixed_w', shape=[d, N], trainable=False, dtype=tf.float32)
		self._rf_assign = tf.assign(w, self._tph)
		
		at = tf.get_variable(initializer=tf.constant_initializer(0.0),\
			name='layer2', shape=[N, num_classes], dtype=tf.float32)	
		c = tf.get_variable(initializer=tf.constant_initializer(0.0),\
			name='layer2_bias', shape=[1, num_classes], dtype=tf.float32)				
		self._phi = tf.matmul(data, w)
		self._phi = tf.nn.relu(self._phi)
		yhat = tf.matmul(self._phi, at) + c # BS x num_classes
		
		self._pten = tf.zeros(shape=[N, 1], dtype=tf.float32)
		p_vec = tf.get_variable(initializer=tf.constant_initializer(0.0), name='Hvec', shape=[N, 1], trainable=False, dtype=tf.float32)
		self._assign_op = tf.assign(p_vec, self._pten)
    		omega = tf.matmul(self._phi, p_vec)
    		self._wsum = tf.reduce_mean(omega * self._phi, axis=0)

		if self._lossType == 'cross_entropy':
			true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), num_classes)
			loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_labels, logits=yhat)
		elif self._lossType == 'hinge':
			# Assumes the labels are plus / minus one and the final layer is not expanded
			loss_vec = tf.maximum(tf.constant(1.0, dtype=tf.float32) - tf.constant(20.0, dtype=tf.float32) * tf.multiply(labels, yhat),\
			 tf.constant(0.0, dtype=tf.float32))
		elif self._lossType == 'square':
			if self._expandedFinalLayer:
				true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), num_classes, dtype=tf.float32)
				loss_vec = (true_labels - yhat) ** 2	
				print("Shape before Summation")
				print(loss_vec.shape)			
				loss_vec = tf.reduce_sum(loss_vec, axis=1, keepdims=True)
				print("Shape after Summation")
				print(loss_vec.shape)
			else:
				loss_vec = (labels - yhat) ** 2			
		else:
			raise Exception('Loss type is not recognized')
		
		assert loss_vec.shape[1] == 1 and len(loss_vec.shape) == 2

		self._pure_loss = tf.reduce_mean(loss_vec)
		self._variables = tf.trainable_variables()
		self._reg_loss = tf.nn.l2_loss(at) #tf.constant(0.0)
		self._reg_loss = self._reg_param * self._reg_loss
		augmented_loss = self._pure_loss + self._reg_loss
		# Classification Accuracy
		self._compute_accuracy(yhat, labels)
		return (augmented_loss, yhat)

	def _form_train_op(self):
		print('Warning: Using RF train op')
		if self._optimizer == 'mom':
			opt = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.9, use_nesterov=True)
		elif self._optimizer == 'sgd':
			opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
		elif self._optimizer == 'adam':
			opt = tf.train.AdamOptimizer(learning_rate=self._lr)
		else:
			raise Exception('Optimizer was not recognized!')		
		self._grads_vars = opt.compute_gradients(self._loss, self._variables)
		self._train_op = opt.apply_gradients(self._grads_vars, name='train_step', global_step=self._global_step_variable)

class FullyConnected(MultiLayerNetwork):
	def __init__(self, data, labels, params):
		self._N = params['num_hiddens']
		self._d = int(data.shape[1])
		self._include_bias = params['include_bias']
		super(FullyConnected, self).__init__(data, labels, params)

	def _model(self, data, labels):
		num_classes = self._num_classes
		N = self._N
		d = self._d
		x, _ = self._fcLayer(data, N, 'init', self._include_bias, self._trainable, 1.0 / np.sqrt(d + 0.0))		
		for i in range(self._num_layers - 1):
			x, _ = self._fcLayer(x, N, 'layer_%d'%(i + 2), self._include_bias, self._trainable, 1.0) 
			x = x / (N + 0.0)
			#self._initVals.append(W0)
		#a = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(N + 0.0)), name='Softmax_layer', shape=[N, num_classes], dtype=tf.float32)
		a = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0), name='Softmax_layer', shape=[N, num_classes], dtype=tf.float32)		
		c = tf.get_variable(initializer=tf.constant_initializer(0.0), name='Softmax_bias', shape=[1, num_classes], dtype=tf.float32)
		yhat = tf.matmul(x, a) / (N + 0.0) + c 
		if self._lossType == 'cross_entropy':
			true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), num_classes)
			loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_labels, logits=yhat)		
		elif self._lossType == 'square':
			if self._expandedFinalLayer:
				true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), num_classes, dtype=tf.float32)
				loss_vec = (true_labels - yhat) ** 2				
				loss_vec = tf.reduce_sum(loss_vec, axis=1, keepdims=True)
			else:
				loss_vec = (labels - yhat) ** 2			
		else:
			raise Exception('Loss type is not recognized')
		
		assert loss_vec.shape[1] == 1 and len(loss_vec.shape) == 2
		self._pure_loss = tf.reduce_mean(loss_vec)
		self._variables = tf.trainable_variables()
		self._reg_loss = 0
		
		print('Using Absolute L2 Regularization')
		for g in self._variables:
			if g.op.name == 'init/W':
				print('W added')
				self._reg_loss += tf.nn.l2_loss(g)
			#elif 'Softmax_layer' in g.op.name:
			#	print('Softmax_layer added')
			#	self._reg_loss += tf.nn.l2_loss(g) 
			elif 'W' in g.op.name:
				self._reg_loss += tf.nn.l2_loss(g) / np.sqrt(N)
		self._reg_loss = self._reg_param * self._reg_loss
	
		augmented_loss = self._pure_loss + self._reg_loss		
		# Classification Accuracy
		self._compute_accuracy(yhat, labels)		
		return (augmented_loss, yhat)

	def _form_train_op(self):
		print('Using the Mean Field Train Op')
		if self._optimizer == 'mom':
			opt = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.9)
		elif self._optimizer == 'sgd':
			opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
		elif self._optimizer == 'adam':
			opt = tf.train.AdamOptimizer(learning_rate=self._lr)
		else:
			raise Exception('Optimizer was not recognized!')
		self._grads_vars = opt.compute_gradients(self._loss, self._variables)		
		m = len(self._grads_vars)
		for i in range(m):
			var_name = self._grads_vars[i][1].op.name
			print(var_name)
			if not var_name == 'Softmax_bias':
				if 'init' in var_name:
					coeff = self._N + 0.0
				else:
					coeff = np.int(np.prod(self._grads_vars[i][1].shape)) + 0.0
				print(coeff)
				self._grads_vars[i] = (coeff * self._grads_vars[i][0], self._grads_vars[i][1])
		self._train_op = opt.apply_gradients(self._grads_vars, name='train_step', global_step=self._global_step_variable)

class Myrtle(MultiLayerNetwork):
	def __init__(self, data, labels, params):
		self._num_classes = params['num_classes']
		self._filter_size = params['filter_size']
		self._num_filters = params['num_hiddens']
		super(Myrtle, self).__init__(data, labels, params)

	def _model(self, data, labels):
		num_filters = self._num_filters
		x = self._conv_layer(data, self._filter_size, 3, num_filters, 'input_filter', True, [1, 0])
		x = self._conv_layer(x, self._filter_size, num_filters, num_filters, 'filter_0', True, [1, 0])
		x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

		x = self._conv_layer(x, self._filter_size, num_filters, num_filters, 'filter_1', True, [1, 0])
		x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')        
		x = self._conv_layer(x, self._filter_size, num_filters, num_filters, 'filter_2', True, [1, 0])
		x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')        
		x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')        
		x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')        

		#for i in range(self._num_layers - 1):
		#    x = self._conv_layer(x, self._filter_size, 3, 32, 'filter_%d'%(i + 2), True, [1, 2])            
		x = tf.layers.flatten(x)
		print(x.shape)
		N = np.int(x.shape[1])
		a = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(N + 0.0)), name='Softmax_layer', shape=[N, self._num_classes], dtype=tf.float32)
		c = tf.get_variable(initializer=tf.constant_initializer(0.0), name='Softmax_bias', shape=[1, self._num_classes], dtype=tf.float32)
		yhat = tf.matmul(x, a) + c 
        
		if self._lossType == 'cross_entropy':
			true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), self._num_classes, dtype=tf.float32)
			loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_labels, logits=yhat)
			print(loss_vec.shape)
		elif self._lossType == 'square':
			true_labels = tf.one_hot(tf.to_int32(labels[:, 0]), self._num_classes, dtype=tf.float32)
			loss_vec = (true_labels - yhat) ** 2
			loss_vec = tf.reduce_sum(loss_vec, axis=1, keepdims=True)
			assert loss_vec.shape[1] == 1 and len(loss_vec.shape) == 2
        	else:
            		raise Exception('Loss type is not recognized')        
		self._pure_loss = tf.reduce_mean(loss_vec)
		self._variables = tf.trainable_variables()

		self._regularizer = tf.reduce_sum([tf.nn.l2_loss(g) for g in self._variables if 'W' in g.op.name])        
		self._reg_loss = self._reg_param * self._regularizer
		augmented_loss = self._pure_loss + self._reg_loss
		# Classification Accuracy
		self._compute_accuracy(yhat, labels)
		return (augmented_loss, yhat)