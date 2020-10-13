import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, './linear_algebra/')
from tensor_utils import AssignmentHelper

class RF_Optimizer(object):
	def __init__(self, model, sess, initializer, sess_dict, optim_reg, n, y_place_holder, penalize_const=False):
		self._N = np.int(model._variables[0].shape[0])
		self._n = n
		self._assignment_obj = AssignmentHelper(model._variables)
		self._sess = sess
		self._init = initializer
		self._dict = sess_dict
		self._Y = np.copy(sess_dict[y_place_holder])
		self._ph = y_place_holder
		# Model Quantities
		self._loss = model._loss
		labels = model._labels
                if len(labels.shape) < 2:
        		modified_labels = tf.expand_dims(labels, 1)
            	else:
            		modified_labels = labels                
                self._weighted_sum = tf.matmul(tf.transpose(modified_labels), model._phi)		

		self._ws = model._wsum
		self._pten = model._pten
		self._assign_op = model._assign_op
		ShapeList = [np.prod(g.shape) for g in model._variables]
    		self._params = np.int(np.sum(ShapeList))
		assert self._params == self._N + 1

		self._pen_const = penalize_const
		self._reg = optim_reg

		self._ave = self.AtxGPU(np.ones((self._n, 1)) / np.sqrt(self._n))
		self._ave = self._ave.T

		temp = self.AtxGPU(self._Y) 
		self._aty = np.zeros((self._params,), dtype=np.float64)
		self._aty[:-1] = temp[0, :]
		self._aty[-1] = np.sum(self._Y) / np.sqrt(self._n)
		self._aty = self._aty.astype(np.float32)		

	def fun(self, x):
	        self._sess.run(self._init, feed_dict=self._dict)          
	        total_loss = 0.0
	        # initialize the data
	        end_of_data = False
	        count = 0.0
	        self._assignment_obj.assign(x, self._sess)
	        while not end_of_data:
	                try:
	                        loss = self._sess.run(self._loss)
	                        total_loss += loss                                
	                        count += 1.0
	                except tf.errors.OutOfRangeError:
	                        end_of_data = True                        
	        val = total_loss / (count + 0.0)
	        # Taking the intercept into account
	        if self._pen_const:
	        	reg = self._reg * (np.linalg.norm(x) ** 2)        
        	else:
        		reg = self._reg * (np.linalg.norm(x[:-1]) ** 2)
	        return val + reg

	def Atx(self):
		return self._aty

	def AtxGPU(self, x):
		x = x.reshape((self._n, 1)) 
		x = x.astype(np.float32)
		self._dict[self._ph] = x
		end_of_data = False      		
		result = np.zeros((1, self._N), dtype=np.float64)		
		# initialize the data		
		self._sess.run(self._init, feed_dict=self._dict)		
		while not end_of_data:
			try:
	            		temp = self._sess.run(self._weighted_sum)
	            		temp = temp.astype(np.float64)
	            		result += temp / np.sqrt(self._n + 0.0)    	                        
			except tf.errors.OutOfRangeError:
				end_of_data = True          		
    		self._dict[self._ph] = self._Y
    		return result

	def Hv(self, x):
        	"""Warning: This function assume that the Hessian is constant."""        
	        x = x.reshape((self._params, 1))
	        x0 = x[:-1]
	        c0 = x[-1]        
	        self._sess.run(self._init, feed_dict=self._dict)
	        end_of_data = False    
	        total = np.zeros((self._params, 1), dtype=np.float64) 
	        count = 0.0
	        self._sess.run(self._assign_op, {self._pten: x0})
	        while not end_of_data:
	                try:
	                        temp = self._sess.run(self._ws)
	                        temp = temp.astype(np.float64)                                                
	                        total[:-1, 0] += temp * 2.0 
	                        count += 1
	                except tf.errors.OutOfRangeError:
	                        end_of_data = True                                        
	        
	        total = total / (count + 0.0)
	        total[:-1] += self._ave * c0 * 2
	        total[-1] = 2 * (c0 + np.dot(self._ave.T, x0))
	        if self._pen_const:
	        	total += 2 * self._reg * x
        	else:
        		total[:-1, :] += 2 * self._reg * x0
	        return total[:, 0].astype(np.float32)

        def num_params(self):
        	return self._params