""" This class provides backend for using Tensorflow to compute fast Hessian-vector products
 for second order optimization."""

import tensorflow as tf
import numpy as np
import collections
from tensor_utils import AssignmentHelper

class OptimizationUtils(object):	
	def __init__(self, loss, preds, params, labels, reg_param, num_obs):        		
		self._loss = loss
                self._n = num_obs
                if len(labels.shape) < 2:
        		modified_labels = tf.expand_dims(labels, 1)
            	else:
            		modified_labels = labels
                if len(preds.shape) < 2:
			self._preds = tf.expand_dims(preds, 1)		
		else:
			self._preds = preds        
                weighted_sum = tf.matmul(tf.transpose(modified_labels), self._preds)
                self._grad_theta = tf.gradients(weighted_sum, params)
                self._variable_assiger = AssignmentHelper(params)

                # Code for Hessian-vector product
                self._v_dict = collections.OrderedDict()            
                with tf.variable_scope('v'):
                        for variable in params:
                                v_variable = tf.get_variable(variable.op.name, shape=variable.shape,\
                                        initializer=tf.ones_initializer(tf.float32), trainable=False, dtype=tf.float32)
                                self._v_dict[variable.op.name] = v_variable        
                self._assignment_obj = AssignmentHelper(self._v_dict.values())                
                v = [self._v_dict[k.op.name] for k in params]
                # gradient w.r.t parameters
                self._gradients = tf.gradients(self._loss, params)
                self._hv_list = tf.gradients(self._gradients, params, grad_ys=v)
                self._reg_param = reg_param
                self._num_params = self._assignment_obj.total_num_params()                        
        
        def num_params(self):
                return self._num_params

        def set_variables(self, x, sess):
                assert len(x) == self.num_params()
                self._variable_assiger.assign(x, sess)

        def ATx(self, sess):    
                """Let A be the matrix of features given by NTK. This function computes A^Tx / np.sqrt(n).
                The input has to be n dimentional and the output is p dimentional. """                
                p = self._assignment_obj.total_num_params()                                      
                ATx = np.zeros((p, 1)).astype(np.float64)                                        
                end_of_data = False
                while not end_of_data:
                        try:
                                grad_list = sess.run(self._grad_theta)    
                                # Pack grad_list as a vector
                                temp = AssignmentHelper._pack(grad_list)
                                temp = temp.reshape((p, 1)) / np.sqrt(self._n + 0.0)
                                ATx += temp.astype(np.float64)                
                        except tf.errors.OutOfRangeError:
                                end_of_data = True                        
                return ATx
    
        def Ax(self, x, sess):    
                """ Computes a n dimensional vector corresponding to [A/sqrt(n)]x. 
                x should be a p dimensional vector. Output is n dimensional."""                
                n = self._n
                Ax = np.zeros((n, 1)).astype(np.float64)                                        
                count = 0
                end_of_data = False
                self.set_variables(x, sess)
                while not end_of_data:
                        try:                       
                                preds = sess.run(self._preds)
                                bs = len(preds)
                                Ax[count * bs : (count + 1) * bs, 0] = preds[:, 0] / np.sqrt(n)
                                count += 1
                        except tf.errors.OutOfRangeError:
                                end_of_data = True                
                return Ax

        def loss(self, x, sess):
                """ Computes the full batch loss + regularization loss."""        
                total_loss = 0.0
                # initialize the data
                end_of_data = False
                count = 0.0
                self.set_variables(x, sess)
                while not end_of_data:
                        try:
                                loss = sess.run(self._loss)
                                total_loss += loss                                
                                count += 1.0
                        except tf.errors.OutOfRangeError:
                                end_of_data = True                        
                return (total_loss / (count + 0.0), self._reg_param * (np.linalg.norm(x) ** 2))
    
        def Hv(self, x, p, sess):
                """ Assumes the Hessian is constant."""
                n = self._assignment_obj.total_num_params()
                hv = np.zeros((n, 1)).astype(np.float64)
                if x is None:
                        x = None # x is not used
                else:
                        self.set_variables(x, sess)
                end_of_data = False
                count = 0.0
                self._assignment_obj.assign(p, sess)
                while not end_of_data:
                        try:                                
                                hv_list_val = sess.run(self._hv_list)            
                                temp = AssignmentHelper._pack(hv_list_val)
                                temp = temp.reshape((n, 1))
                                hv += temp.astype(np.float64) 
                                count += 1.0
                        except tf.errors.OutOfRangeError:
                                end_of_data = True
                hv = hv / count
                hv += p.reshape((n, 1)) * self._reg_param * 2
                return hv

        def gradient(self, x, sess):
                n = self._assignment_obj.total_num_params()
                grad = np.zeros((n, 1)).astype(np.float64)                
                self.set_variables(x, sess)
                end_of_data = False
                count = 0.0
                while not end_of_data:
                        try:                
                                temp = sess.run(self._gradients)            
                                temp = AssignmentHelper._pack(temp)
                                temp = temp.reshape((n, 1))
                                grad += temp.astype(np.float64) 
                                count += 1.0
                        except tf.errors.OutOfRangeError:
                                end_of_data = True
                grad = grad / count        
                grad += x.reshape((n, 1)) * self._reg_param * 2
                return grad
