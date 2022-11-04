import tensorflow as tf
import cv2
import numpy as np
import random
import tensorflow.contrib.slim as slim 
import math

def log_prob_tf(myu, sigma, x):
    return -0.5 * ((x - myu) / sigma) ** 2# - tf.log(sigma) - 0.5 * np.log(2 * np.pi)

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature = 1., hard=True):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def dense(x, W, b, activation = 'relu', norm = None):
    x = tf.matmul(x, W)
    if b is not None:
        x = tf.add(x, b)
    if norm is None:
        pass
    elif norm[:2] == 'BN':
        is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
        x = tf.layers.batch_normalization(x, training = is_training)
    elif norm[:2] == 'GN':
        x = group_norm(x, scope = norm)
    if activation == 'x': return x
    if activation == 'sigmoid': return tf.nn.sigmoid(x)
    if activation == 'tanh': return tf.nn.tanh(x) 
    if activation == 'relu': return tf.nn.relu(x)
    if activation == 'lrelu': return tf.nn.leaky_relu(x)

def conv2D(x, W, b, strides = 1, activation = 'relu', norm = None, padding = 'SAME', ws = False):
    if ws:
        if norm is None:
            print('WARNING : weight standardization without normalization(BN, GN, ...)')
        mean, var = tf.nn.moments(W, [0, 1, 2], keep_dims = True)
        W = (W - mean) / tf.sqrt(var + 1e-5)
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding = padding)
    if b is not None:
        x = tf.add(x, b)
    if norm is None:
        pass
    elif norm[:2] == 'BN':
        is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
        x = tf.layers.batch_normalization(x, training = is_training)
    elif norm[:2] == 'GN':
        x = group_norm(x, scope = norm)
    if activation == 'relu': return tf.nn.relu(x)
    if activation == 'lrelu': return tf.nn.leaky_relu(x)
    if activation == 'x': return x

def convT2D(x, W, b, strides = 1, activation = 'relu', norm = None, padding = 'SAME', ws = False):
    if padding == 'VALID':
        output_shape = tf.stack([tf.shape(x)[0], (x.shape[1] - 1) * strides + W.shape[0], (x.shape[2] - 1) * strides + W.shape[1], W.shape[3]])
    if padding == 'SAME':
        output_shape = tf.stack([tf.shape(x)[0], x.shape[1] * strides, x.shape[2] * strides, W.shape[3]])
    W = tf.transpose(W, [0, 1, 3, 2])
    x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,strides,strides,1], padding = padding)
    if activation == 'x': return x
    if activation == 'sigmoid': return tf.nn.sigmoid(x)
    if activation == 'tanh': return tf.nn.tanh(x) 
    if activation == 'relu': return tf.nn.relu(x)
    if activation == 'lrelu': return tf.nn.leaky_relu(x)  

def maxpool2D(x, k = 2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding = "SAME")

class rewardTracker():
    def __init__(self, GAMMA):
        self.mean = 0
        self.N = 0
        self.var = 0
        self.SSE = 0
        self.X0 = None
        self.GAMMA = GAMMA

    def update(self, x0):
        if self.X0 is None:
            self.X0 = x0
        else:
            self.X0 = self.X0 * self.GAMMA + x0
        #print(self.X0)
        for x in self.X0:
            self.N += 1
            error = x - self.mean
            self.mean += (error / self.N)
            self.SSE += error * (x - self.mean)

    def get_std(self):
        return math.sqrt(self.SSE / self.N) + 1e-8

def excludeNone(grad, var):
    gradients, variables = [], []
    for i, grad in enumerate(grad):
        if grad is None:
            continue
        gradients.append(grad)
        variables.append(var[i])
    return gradients, variables

from constants import NUM_CHUNKS
def makeOptimizer(lr, loss, decay = False, var_list = None):
    if decay:
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(lr, global_step, 1000, 0.96, staircase = False)
        opt = tf.train.AdamOptimizer(lr, epsilon = 1e-4)
        if var_list == None:
            gradients, variables = zip(*opt.compute_gradients(loss))
        else:
            gradients, variables = zip(*opt.compute_gradients(loss, var_list = var_list))
        gradients, variables = excludeNone(gradients, variables)
        accum_grads = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in variables]
        acc_init = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]
        accum_ops = [accum_grads[i].assign_add(gradients / NUM_CHUNKS) for i, gradients in enumerate(gradients)]
        accum_gradsC, _ = tf.clip_by_global_norm(accum_grads, 5.0)
        final_opt = opt.apply_gradients([(accum_gradsC[i], var) for i, var in enumerate(variables)])
        #final_opt = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
    else:
        opt = tf.train.AdamOptimizer(lr, epsilon = 1e-4)
        if var_list == None:
            gradients, variables = zip(*opt.compute_gradients(loss))
        else:
            gradients, variables = zip(*opt.compute_gradients(loss, var_list = var_list))
        gradients, variables = excludeNone(gradients, variables)
        print(variables)
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        accum_grads = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in variables]
        acc_init = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]
        #print(gradients, variables)
        accum_ops = [accum_grads[i].assign_add(gradients / NUM_CHUNKS) for i, gradients in enumerate(gradients)]
        accum_gradsC, _ = tf.clip_by_global_norm(accum_grads, 5.0)
        final_opt = opt.apply_gradients([(accum_gradsC[i], var) for i, var in enumerate(variables)])
        #final_opt = opt.apply_gradients(zip(gradients, variables))
    return accum_ops, acc_init, final_opt, tf.global_norm(accum_grads)

def variable_summaries(var):
    summaries = []
    name = "_"+var.name
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        summaries.append(tf.summary.scalar('mean'+name, mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            summaries.append(tf.summary.scalar('stddev'+name, stddev))
        summaries.append(tf.summary.scalar('max'+name, tf.reduce_max(var)))
        summaries.append(tf.summary.scalar('min'+name, tf.reduce_min(var)))
        summaries.append(tf.summary.histogram('histogram'+name, var))
    return summaries

def group_norm(x, G=16, eps=1e-5, scope='group_norm'):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        N, H, W, C = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        G = min(G, C)
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(x, [N, H, W, C]) * gamma + beta
    return x

def lr_schedule(lr, step, decay_step, ratio):
    if step < decay_step:
        return lr * step / decay_step
    else:
        return lr * np.power(ratio, (step / decay_step - 1))

def AdaIn(features, z, w, b):
    style = dense(z, w, b, activation = 'relu')
    scale, bias = style[:, :features.shape[3]], style[:, features.shape[3]:] 
    mean, variance = tf.nn.moments(features, list(range(len(features.get_shape())))[1:-1], keep_dims=True)
    rsigma = tf.rsqrt(variance + 1e-8)
    normalized = (features - mean) * rsigma
    scale_broadcast = tf.reshape(scale, tf.shape(mean))
    bias_broadcast = tf.reshape(bias, tf.shape(mean))
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized 

# Modified from https://gist.github.com/kmjjacobs/eab1e840aecf0ac232cc8370a9be9093
class GRU:
    """Implementation of a Gated Recurrent Unit (GRU) as described in [1].
    
    [1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
    
    Arguments
    ---------
    input_dimensions: int
        The size of the input vectors (x_t).
    hidden_size: int
        The size of the hidden layer vectors (h_t).
    dtype: obj
        The datatype used for the variables and constants (optional).
    """
    
    def __init__(self, name, input_dimensions, hidden_size, dtype=tf.float32):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.name = name
        
        with tf.variable_scope(self.name):
            # Weights for input vectors of shape (input_dimensions, hidden_size)
            self.Wr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wr')
            self.Wz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wz')
            self.Wh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wh')
            
            # Weights for hidden vectors of shape (hidden_size, hidden_size)
            self.Ur = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ur')
            self.Uz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uz')
            self.Uh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uh')
            
            # Biases for hidden vectors of shape (hidden_size,)
            self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='br')
            self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bz')
            self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bh')  

    def forward(self, h_tm1, x_t):
        """Perform a forward pass.
        
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        # Definitions of z_t and r_t
        z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
        r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)
        
        # Definition of h~_t
        h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)
        
        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
        
        return h_t
