import tensorflow as tf
import cv2
import numpy as np
import random
import tensorflow.contrib.slim as slim 
import math

def dense(x, W, b, activation = 'lrelu', use_bn = False):
	x = tf.matmul(x, W)
	if b is not None:
		x = tf.add(x, b)
	if use_bn:
		is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
		x = tf.layers.batch_normalization(x, training = is_training)
	if activation == 'x': return x
	if activation == 'sigmoid': return tf.nn.sigmoid(x)
	if activation == 'tanh': return tf.nn.tanh(x) 
	if activation == 'relu': return tf.nn.relu(x)
	if activation == 'lrelu': return tf.nn.leaky_relu(x)

def conv2D(x, W, b, strides = 1, activation = 'lrelu', use_bn = True):
	x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding = "VALID")
	if use_bn:
		is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
		x = tf.layers.batch_normalization(x, training = is_training)
	else: x = tf.nn.bias_add(x, b)
	if activation == 'relu': return tf.nn.relu(x)
	if activation == 'lrelu': return tf.nn.leaky_relu(x)

def maxpool2D(x, k = 2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding = "SAME")

def unzip_obs(obs, IMG_H, IMG_W, RAW_WAV_LENGTH):
	imgs, wavs = [], []
	for i in range(NUM_AGENTS):
		img, wav = obs['img'][i], obs['wav'][i]
		img = np.reshape(img, [2, IMG_H, IMG_W])
		wav = np.reshape(wav, [2, RAW_WAV_LENGTH])
		#wav = abs(np.fft.rfft(wav))[:int(MAX_FREQ/FREQ_STEP)]
		#print(np.min(wav), np.max(wav))
		wav0, wav1 = wav[0], wav[1]
		wav0 = abs(np.fft.rfft(wav0))[:250]
		wav0 = np.log10(wav0 + 1e-8)
		wav1 = abs(np.fft.rfft(wav1))[:250]
		wav1 = np.log10(wav1 + 1e-8)
		wav = np.array([wav0, wav1])
		#wav = np.reshape(wav0 - wav1, [2, 250])
		#print(np.min(wav), np.max(wav))
		#print(np.min(wav0), np.max(wav0), np.min(wav1), np.max(wav1))
		#print(wav0)
		#print(img.mean(), img.var())
		#print(np.max(wav0), np.max(wav1))
		imgs.append(img), wavs.append(wav)
	obs['img'] = np.array(imgs)
	obs['wav'] = np.array(wavs)

def wav2freq(wav):
	wav0, wav1 = wav[0], wav[1]
	wav0 = abs(np.fft.rfft(wav0))[:250]
	wav0 = np.log10(wav0 + 1e-8)
	wav1 = abs(np.fft.rfft(wav1))[:250]
	wav1 = np.log10(wav1 + 1e-8)
	wav = np.array([wav0, wav1])
	#print(np.max(wav0), np.max(wav1))
	return wav


def getActionFromPolicy(p):
	res = np.zeros_like(p).astype(np.int32)
	for i in range(p.shape[0]):
		#p0 = (p[i] + 0.05) / (1 + 0.05 * ACTION_LENGTH)
		p0 = p[i]
		action = np.random.choice(p.shape[1], 1, p = p0)
		res[i][action] = 1
	return res

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

def makeOptimizer(lr, loss, decay = False):
	if decay:
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(lr, global_step, 1000, 0.96, staircase = False)
		opt = tf.train.AdamOptimizer(lr)
		gradients, variables = zip(*opt.compute_gradients(loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		final_opt = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
	else:
		opt = tf.train.AdamOptimizer(lr)
		gradients, variables = zip(*opt.compute_gradients(loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		final_opt = opt.apply_gradients(zip(gradients, variables))
	return final_opt

def getRewardFromPos(pos, action, wav):
	wav = wav.flatten()
	wavL, wavR = np.max(wav[:250]), np.max(wav[250:])
	wav = wavL - wavR
	cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
	if pos[0] * wav > 0:
		print("HOLYSHIT", wavL, wavR, pos)
	if action[0] == 1:
		if pos[0] < 0:
			return 0.03
		else:
			return -0.03
	if action[1] == 1:
		if pos[0] > 0:
			return 0.03
		else:
			return -0.03
	if action[2] == 1:
		return 0.1 * cosdeg - 0.05
	'''
	if cosdeg >= 0.9:
		if action[2] == 1:
			return 0.03
		else:
			return 0
	elif pos[2] < 0:
		if action[0] == 1:
			return 0.03
		else:
			return 0
	else:
		if action[1] == 1:
			return 0.03
		else:
			return 0
	'''

def getActionFromPos(pos, wav):
	wav = wav.flatten()
	wavL, wavR = np.max(wav[:250]), np.max(wav[250:])
	wav = wavL - wavR
	cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
	if cosdeg > 0.8:
		return np.array([0, 0, 1])
	if wav > 0:
		return np.array([1, 0, 0])
	else:
		return np.array([0, 1, 0])
	
def variable_summaries(var):
  name = "_"+var.name
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean'+name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev'+name, stddev)
    tf.summary.scalar('max'+name, tf.reduce_max(var))
    tf.summary.scalar('min'+name, tf.reduce_min(var))
    tf.summary.histogram('histogram'+name, var)


