import tensorflow as tf
from tensorflow.python.framework import dtypes


class Dense():
    """ Dense (fully-connected) layer, initialized with xavier init"""

    def __init__(self, scope="dense_layer", num_nodes=None, dropout=1.0, nonlin = tf.identity):
        """Initialize fully connected layer (with dropout)

        Args:
            scope(string)  : tensorflow variable scope for layer
            num_nodes(int)       : number of nodes in layer
            dropout(float) : apply dropout to layer (with probability `dropout`)
            nonlin(tf.op)  : specify non-linear unit in layer
        """
        assert num_nodes, "Specify number of nodes in layer!"
        self.scope = scope
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.nonlin = nonlin

    def __call__(self, x):
        """apply Dense layer
        
        Args:
            x(tf.Tensor): input data
        Returns:
            tf.Tensor: return input data applied to layer
        """
        if not 'w' in self.__dict__ or not 'b' in self.__dict__:
            self._init(x)
        return self._apply(x)

    def _init(self, x):
        """Initialize Dense layer with modified Xavier initialization for non-linear units.
        See https://arxiv.org/pdf/1502.01852.pdf 

        Args:
            x(tf.Tensor): input tensor
        """
        with tf.name_scope(self.scope):
            input_nodes = x.get_shape()[1].value
            sd = tf.cast((2 / input_nodes) ** 0.5, tf.float32)

            w = tf.random_normal([input_nodes, self.num_nodes], stddev = sd)
            b = tf.zeros([self.num_nodes])
            self.w = tf.Variable(w, trainable=True, name="weights")
            self.b = tf.Variable(b, trainable=True, name="biases")
            self.w = tf.nn.dropout(self.w, self.dropout) # apply dropout
            # self.w = tf.get_variable("weights", w.get_shape(), trainable=True, initializer=xavier_init)
            # self.b = tf.get_variable("biases", b.get_shape(), trainable=True, initializer=tf.constant_initializer(0.0))



    def _apply(self, x):
        return self.nonlin(tf.matmul(x, self.w) + self.b)

def xavier_init(shape, dtype=dtypes.float32):
    sd = tf.cast((2 / shape[0]) ** 0.5, tf.float32)
    w = tf.random_normal([shape[0], shape[1]], stddev = sd)
    return w

