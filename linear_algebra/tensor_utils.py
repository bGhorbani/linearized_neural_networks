""" This class provides functionalities for switching between a list of tensors and 
its corresponding numpy array. Code downloaded from https://github.com/google/spectral-density/"""

import tensorflow as tf
import numpy as np
import collections

class AssignmentHelper(object):
    """Helper for assigning variables between python and TensorFlow."""

    def __init__(self, variables_list):
        """Constructor for assignment helper.

        Args:
            variables_list: A list of tf.Variable that we want to assign to.
        """
        self._variables_list = variables_list

        # Ops and functions for assigning to model variables.
        self._assign_ops = []
        self._assign_feeds = []
        for var in self._variables_list:
            zeros = tf.zeros_like(var)
            self._assign_ops.append(tf.assign(var, zeros))
            self._assign_feeds.append(zeros)

        self._component_shapes = [x.shape.as_list() for x in self._variables_list]
        self._component_sizes = np.cumsum([np.prod(x) for x in self._component_shapes])

    # Utilities for packing/unpacking and converting to numpy.
    @staticmethod
    def _pack(x):
        """Converts a list of np.array into a single vector."""
        return np.concatenate([np.reshape(y, [-1]) for y in x]).astype(np.float64)

    def _unpack(self, x):
        """Converts a vector into a list of np.array, according to schema."""
        shapes_and_slices = zip(self._component_shapes, np.split(x, self._component_sizes[:-1]))
        return [np.reshape(y, s).astype(np.float64) for s, y in shapes_and_slices]

    def assign(self, x, sess):
        """Assigns vectorized np.array to tensorflow variables."""
        assign_values = self._unpack(x)
        sess.run(self._assign_ops, feed_dict=dict(zip(self._assign_feeds, assign_values)))

    def retrieve(self, sess):
        """Retrieves tensorflow variables to single numpy vector."""
        values = sess.run(self._variables_list)
        return AssignmentHelper._pack(values)

    def total_num_params(self):
        """Returns the total number of parameters in the model."""
        return self._component_sizes[-1]
