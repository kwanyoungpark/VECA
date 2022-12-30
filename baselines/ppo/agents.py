import tensorflow as tf
from .layer import *
import numpy as np
from .legacy.constants import *

zero_init = tf.constant_initializer(0.)
one_init = tf.constant_initializer(1.)

def u_init(mn, mx):
    return tf.random_uniform_initializer(mn,mx)

def n_init(mean, std):
    return tf.random_normal_initializer(mean, std)

def c_init(x):
    return tf.constant_initializer(x)

def o_init(x):
    return tf.orthogonal_initializer(x)

def normalize(x, axis):
    mean, var = tf.nn.moments(x, axes = axis, keep_dims = True)
    return (x - mean) * tf.rsqrt(var + 1e-8)
'''
def split(data):
    data_new = []
    for i in range(len(data)):
        if dat is None:
            data_new.append(None)
            continue
        dat = tf.reshape(dat, [NUM_AGENTS * TIME_STEP // RNN_STEP, RNN_STEP] + dat.get_shape().as_list()[2:])
        data_new.append(dat)
    res = []
    for i in range(RNN_STEP):
        res.append([])
        for dat in data:
'''

class UniversalRNNEncoder():
    def __init__(self, name, SIM):
        self.name = name
        self.enc = UniversalEncoder(name, SIM)
        with tf.variable_scope(self.name):
            #print("name", name)
            #print("scope", tf.get_variable_scope().name)
            #self.GRU = tf.nn.rnn_cell.GRUCell(STATE_LENGTH)
            self.GRU = GRU(name, STATE_LENGTH, STATE_LENGTH)
            #print("GRU", self.GRU.name)

    def get_params(self):
        return self.enc.get_params() + self.RNN.get_params()

    def forward(self, data):
        img, wav, obj, touch, h_0 = data
        NUM_AGENTS = tf.shape(h_0)[0]
        dat = (img, wav, obj, touch)
        with tf.variable_scope(self.name):
            i_t = self.enc.forward(dat)
            res = self.GRU.forward(i_t, h_0)
            #i_t = tf.reshape(i_t, [NUM_AGENTS, 1, STATE_LENGTH]) 
            #_, res = tf.nn.dynamic_rnn(self.GRU, i_t, initial_state = h_0)
        return res

    def forward_train(self, data):
        img, wav, obj, touch, h_0 = data
        NUM_AGENTS = tf.shape(h_0)[0]
        dat = (img, wav, obj, touch)
        with tf.variable_scope(self.name):
            z = self.enc.forward(dat) # [None, STATE_LENGTH]
            z = tf.reshape(z, [NUM_AGENTS * TIME_STEP // RNN_STEP, RNN_STEP, STATE_LENGTH])
            h_0 = tf.reshape(h_0, [NUM_AGENTS * TIME_STEP // RNN_STEP, RNN_STEP, STATE_LENGTH])
            h = [h_0[:, 0]]
            for i in range(RNN_STEP):
                h_iR = h_0[:, i]
                h_i = h[i]
                isSame = tf.tile(tf.reduce_mean(tf.square(h_iR - h_i), axis = 1, keep_dims = True) < 1e-6, [1, STATE_LENGTH])
                h_i = tf.where(isSame, h[i], h_i)
                h.append(self.GRU.forward(z[:, i], h_i))
            res = h[RNN_STEP]
            #_, res = tf.nn.dynamic_rnn(self.GRU, z, initial_state = h_0)
        return res

class UniversalEncoder():
    def __init__(self, name):
        self.name = name
        self.hidden_size = 256
        self.feature_size = 512
        self.initialized = False
        '''
        if SIM == 'VECA':
            IMG_C = 6
            WAV_C, WAV_LENGTH = 2, 66*13
            NUM_OBJS = 3
            #TACTILE_LENGTH = 1182 + 2 * 66     # GrabObject Before
            #TACTILE_LENGTH = 2 * 66            # GrabObject Before w/o tactile
            #TACTILE_LENGTH = 1182 + 2 * 58     # GrabObject
            TACTILE_LENGTH = 5 * 82 + 9888     # RunBaby
            #TACTILE_LENGTH = 5 * 82            # RunBaby w/o tactile
            #TACTILE_LENGTH = 1                  # COGNIANav simple tactile
        #IMG_C = env.observation_space['image'][0]
        #WAV_C, WAV_LENGTH = env.observation_space['audio']
        #NUM_OBJS = env.observation_space['obj']
        
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.get_variable('wc1iA', [8, 8, IMG_C, 32]),
                'wc2iA': tf.get_variable('wc2iA', [4, 4, 32, 64]),
                'wc3iA': tf.get_variable('wc3iA', [3, 3, 64, 64]),
                #'wc4iA': tf.get_variable('wc4iA', [3, 3, 64, 64]),
                'wd1iA': tf.get_variable('wd1iA', [3136, 256]),
                
                'wd1wA': tf.get_variable('wd1wA', [WAV_C * WAV_LENGTH, 256]),

                'wd1wO': tf.get_variable('wd1wO', [NUM_OBJS, 256]),

                'wd1wT': tf.get_variable('wd1wT', [TACTILE_LENGTH, 256]),
                
                'wd1dA': tf.get_variable('wd1dA', [256 + 256 + 256, STATE_LENGTH])
            }
            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [32]),
                'bc2iA': tf.get_variable('bc2iA', [64]),
                'bc3iA': tf.get_variable('bc3iA', [64]),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [256]),

                'bd1wA': tf.get_variable('bd1wA', [256]),
                
                'bd1wT': tf.get_variable('bd1wT', [256]),

                'bd1dA': tf.get_variable('bd1dA', [STATE_LENGTH], initializer = c_init(0.1))
            }
            '''

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())

    def normalize_filters(self, x):
        h, w, c1, c2 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = tf.transpose(tf.reshape(x, [h*c1, w*c2, 1, 1]), [2, 0, 1, 3])
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))       
        return tf.cast(x*255, tf.uint8)

    def summarize_filters(self):
        tf.summary.image('wc1', self.normalize_filters(self.weights['wc1iA']))#, max_outputs = 64)
        tf.summary.image('wc2', self.normalize_filters(self.weights['wc2iA']))#, max_outputs = 64)
        tf.summary.image('wc3', self.normalize_filters(self.weights['wc3iA']))#, max_outputs = 64)

    def forward(self,data):
        img, wav, obj, touch = None, None, None, None
        for key in data:
            if "agent/img" in key: img = data[key]
            elif "agent/wav" in key: wav = data[key]
            elif "agent/obj" in key: obj = data[key]
            elif "agent/touch" in key: touch = data[key]
            elif "agent/goodobj" in key: 
                B = img.get_shape()[0]
                print("GoodObj Recognized")
                goodobj = (tf.reshape(data[key], (B,3)) - 12.5) / 12.5
            elif "agent/badobj" in key:
                B = img.get_shape()[0]
                print("BadObj Recognized")
                badobj = (tf.reshape(data[key], (B,3)) - 12.5) / 12.5
        B = img.get_shape()[0]
        im4, au1, ob1, to1 = (tf.zeros([B, self.hidden_size]), tf.zeros([B, self.hidden_size]), 
            tf.ones([B, self.hidden_size]), tf.zeros([B, self.hidden_size]))

        if not self.initialized:
            self.weights, self.biases = {}, {}
        with tf.variable_scope(self.name): 
            if img is not None:
                B,A,N,H,W,C = img.get_shape().as_list()
                img = tf.reshape(tf.transpose(img, [0,1,3,4,2,5]), [B * A, H, W, N * C])
                if not self.initialized:
                    self.weights['wc1iA'] = tf.get_variable('wc1iA', [8, 8, N * C, 32] )
                    self.biases['bc1iA'] = tf.get_variable('bc1iA', [32])
                im1 = conv2D(img, self.weights['wc1iA'], self.biases['bc1iA'], strides = 4, padding = "VALID")

                if not self.initialized:
                    self.weights['wc2iA'] = tf.get_variable('wc2iA', [4, 4, 32, 64])
                    self.biases['bc2iA'] = tf.get_variable('bc2iA', [64])
                im2 = conv2D(im1, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "VALID")

                if not self.initialized:
                    self.weights['wc3iA'] = tf.get_variable('wc3iA', [3, 3, 64, 64])
                    self.biases['bc3iA'] = tf.get_variable('bc3iA', [64])
                im3 = conv2D(im2, self.weights['wc3iA'], self.biases['bc3iA'], strides = 1, padding = "VALID")
                B_, H_, W_, C_ = im3.get_shape().as_list()

                im3 = tf.reshape(im3, [B_, H_ * W_ * C_])
                
                if not self.initialized:
                    self.weights['wd1iA'] = tf.get_variable('wd1iA', [H_ * W_ * C_, self.hidden_size])
                    self.biases['bd1iA'] = tf.get_variable('bd1iA', [self.hidden_size])
                im4 = dense(im3, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'relu')
            
            if wav is not None:
                B,A,N,C = wav.get_shape().as_list()
                wav = tf.reshape(wav, [B* A, N * C])
                
                if not self.initialized:
                    self.weights['wd1wA'] = tf.get_variable('wd1wA', [N * C, self.hidden_size])
                    self.biases['bd1wA'] = tf.get_variable('bd1wA', [self.hidden_size])
                au1 = dense(wav, self.weights['wd1wA'], self.biases['bd1wA'], activation = 'relu')

            if obj is not None:
                B,A,C = obj.get_shape().as_list()
                
                if not self.initialized:
                    self.weights['wd1wO'] = tf.get_variable('wd1wO', [C, self.hidden_size])
                ob1 = tf.matmul(obj, self.weights['wd1wO'])

            if touch is not None:
                B,A,C = touch.get_shape().as_list()
                
                if not self.initialized:
                    self.weights['wd1iT'] = tf.get_variable('wd1iA', [C, self.hidden_size])
                    self.biases['bd1iT'] = tf.get_variable('bd1iA', [self.hidden_size])
                to1 = dense(touch, self.weights['wd1wT'], self.biases['bd1wT'])

            da0 = tf.concat([im4 * ob1, au1, to1, goodobj, badobj], axis = 1)
            
            if not self.initialized:
                self.weights['wd1dA'] = tf.get_variable('wd1dA', [self.hidden_size * 3 + 3 * 2, self.feature_size])
                self.biases['bd1dA'] = tf.get_variable('bd1dA', [self.feature_size])
            res = dense(da0, self.weights['wd1dA'], self.biases['bd1dA'], activation = 'tanh')
            self.initialized = True
        return res

class AtariEncoder():
    def __init__(self, name, env):
        IMG_C = env.observation_space['image'][0]
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.get_variable('wc1iA', [8, 8, IMG_C, 32]),
                'wc2iA': tf.get_variable('wc2iA', [4, 4, 32, 64]),
                'wc3iA': tf.get_variable('wc3iA', [3, 3, 64, 64]),
                #'wc4iA': tf.get_variable('wc4iA', [3, 3, 64, 64]),
                'wd1iA': tf.get_variable('wd1iA', [3136, STATE_LENGTH]),
            }
            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [32], initializer = zero_init),
                'bc2iA': tf.get_variable('bc2iA', [64], initializer = zero_init),
                'bc3iA': tf.get_variable('bc3iA', [64], initializer = zero_init),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [STATE_LENGTH], initializer = zero_init),
            }

    def normalize_filters(self, x):
        h, w, c1, c2 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = tf.transpose(tf.reshape(x, [h*c1, w*c2, 1, 1]), [2, 0, 1, 3])
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))       
        return tf.cast(x*255, tf.uint8)

    def summarize_filters(self):
        tf.summary.image('wc1', self.normalize_filters(self.weights['wc1iA']))#, max_outputs = 64)
        tf.summary.image('wc2', self.normalize_filters(self.weights['wc2iA']))#, max_outputs = 64)
        tf.summary.image('wc3', self.normalize_filters(self.weights['wc3iA']))#, max_outputs = 64)

    def forward(self, data):
        img = data
        with tf.variable_scope(self.name): 
            batch_size = tf.shape(img)[0]
            im1 = conv2D(img, self.weights['wc1iA'], self.biases['bc1iA'], strides = 4, padding = "VALID")
            im2 = conv2D(im1, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "VALID")
            im3 = conv2D(im2, self.weights['wc3iA'], self.biases['bc3iA'], strides = 1, padding = "VALID")
            im3 = tf.reshape(im3, [batch_size, 3136])
            res = dense(im3, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'tanh')
        return res
'''
class MLPEncoder():
    def __init__(self, name, env):
        NUM_OBS = env.observation_space['touch']
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = {
                'wd1iA': tf.get_variable('wd1iA', [NUM_OBS, STATE_LENGTH]),
            }
            self.biases = {
                'bd1iA': tf.get_variable('bd1iA', [STATE_LENGTH], initializer = zero_init),
            }

    def forward(self, data):
        touch = data
        with tf.variable_scope(self.name): 
            batch_size = tf.shape(touch)[0]
            res = dense(touch, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'tanh')
        return res
'''
class AgentContinuousPPO():
    def __init__(self, name, encoder, action_space):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space]),
                'wdsiA': tf.get_variable('wdsiA', [STATE_LENGTH, action_space]),
           }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init),
                'bdsiA': tf.get_variable('bdsiA', [action_space], initializer = zero_init)
           }

    def forward(self, data):
        z = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(z, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            ims = dense(z, self.weights['wdsiA'], self.biases['bdsiA'], activation = 'tanh')
            self.myu = imm
            self.sigma = tf.exp(ims)
        return (self.myu, self.sigma)
    
    def get_loss(self, data, oldmyu, oldsigma, action, adv, ent_coef, CLIPRANGE = 0.2, CLIPRANGE2 = 0.05):
        with tf.variable_scope(self.name):
            oldlogP = -2 * tf.square((action - oldmyu) / oldsigma)# - tf.log(oldsigma)
            myu, sigma = self.forward(data)
            logP = -2 * tf.square((action - myu) / sigma)# - tf.log(sigma)
            clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            print("ratio", ratio.shape, logP.shape, oldlogP.shape)
           
            # Defining Loss = - J is equivalent to max J
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * clipped_ratio
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
 
            entropy = tf.reduce_mean(tf.reduce_mean(tf.log(sigma), 1))# + tf.reduce_mean(tf.abs(myu))
            #entropy = -tf.reduce_mean(tf.reduce_mean(logP - tf.log(1 - tf.square(tf.tanh(action)) + 1e-4), axis = 1))

            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (myu, sigma, oldmyu, oldsigma, oldlogP, logP)
            return loss, clipfrac, entropy, approxkl, pg_loss, ratio, obs

class AgentDiscretePPO():
    def __init__(self, name, encoder, action_space):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space], initializer = n_init(0,0.06)) 
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init)
            }

    def forward(self, data):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(im4, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            self.prob = tf.nn.softmax(imm, axis=1)
        return self.prob
    
    def get_loss(self, data, oldprob, action, adv, ent_coef, CLIPRANGE = 0.2, CLIPRANGE2 = 0.05):
        with tf.variable_scope(self.name):
            oldlogP = tf.log(tf.reduce_sum(oldprob * action, axis = 1) + 1e-4)
            prob = self.forward(data)
            logP = tf.log(tf.reduce_sum(prob * action, axis = 1) + 1e-4)
            clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            
            # Defining Loss = - J is equivalent to max J
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * clipped_ratio
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            
            entropy = tf.reduce_mean(tf.reduce_sum(-prob * tf.log(prob + 1e-4), 1))
            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (prob, oldprob, oldlogP, logP)
            return loss, clipfrac, entropy, approxkl, pg_loss, ratio, obs

class AgentContinuousSAC():
    def __init__(self, name, encoder, critic, action_space):
        self.name = name
        self.enc = encoder
        self.critic = critic
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space]),
                'wdsiA': tf.get_variable('wdsiA', [STATE_LENGTH, action_space]),
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init),
                'bdsiA': tf.get_variable('bdsiA', [action_space], initializer = zero_init)
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values()) + list(self.enc.get_params())

    def forward(self, data):
        z = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(z, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            ims = dense(z, self.weights['wdsiA'], self.biases['bdsiA'], activation = 'tanh')
            self.myu = imm
            #ims = -20 + 22 * (ims + 1) * 0.5
            self.sigma = tf.exp(ims)
        return (self.myu, self.sigma)
    
    def get_loss(self, data, oldmyu, oldsigma, ent_coef):
        with tf.variable_scope(self.name):
            myu, sigma = self.forward(data)
            action = myu + sigma * tf.random.truncated_normal(tf.shape(myu))
            #oldlogP = log_prob_tf(oldmyu, oldsigma, action)
            oldlogP = -0.5 * tf.square((action - oldmyu) / oldsigma)# - tf.log(oldsigma)
            #logP = log_prob_tf(myu, sigma, action)
            myu, sigma = tf.stop_gradient(myu), tf.stop_gradient(sigma)
            logP = -0.5 * tf.square((action - myu) / sigma)# - tf.log(sigma)
            #clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            print("ratio", ratio.shape, logP.shape, oldlogP.shape) 
 
            pg_loss = -tf.reduce_mean(self.critic.forward(data, action))
            #entropy = tf.reduce_mean(tf.reduce_sum(tf.log(sigma), 1))
            #entropy = -tf.reduce_mean(tf.reduce_sum(logP - tf.log(1 - tf.square(tf.tanh(action)) + 1e-4), axis = 1))
            entropy = -tf.reduce_mean(tf.reduce_mean(logP - tf.log(1 - tf.square(tf.tanh(action)) + 1e-4), axis = 1))
            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (myu, sigma, oldmyu, oldsigma, oldlogP, logP)
            return loss, entropy, approxkl, pg_loss, ratio, obs

class AgentDiscreteSAC():
    def __init__(self, name, encoder, critic, action_space):
        self.name = name
        self.enc = encoder
        self.critic = critic
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space]),
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values()) + list(self.enc.get_params())

    def forward(self, data):
        z = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(z, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            self.prob = tf.nn.softmax(imm)
        return self.prob
    
    def get_loss(self, data, oldprob, ent_coef):
        with tf.variable_scope(self.name):
            prob = self.forward(data)
            action = gumbel_softmax(tf.log(prob))
            #print(action.shape)
            #action = tf.random.
            oldlogP = tf.reduce_sum(tf.log(oldprob) * action, axis = 1)
            logP = tf.reduce_sum(tf.log(prob) * action, axis = 1)
            #clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            print("ratio", ratio.shape, logP.shape, oldlogP.shape) 
 
            pg_loss = -tf.reduce_mean(self.critic.forward(data, action))
            entropy = -tf.reduce_mean(logP)
            #entropy = tf.reduce_mean(tf.reduce_sum(tf.log(sigma), 1))
            #entropy = -tf.reduce_mean(logP)
            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (prob, oldprob, oldlogP, logP)
            return loss, entropy, approxkl, pg_loss, ratio, obs

class AgentContinuous_MDN():
    def __init__(self, name, encoder):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [256, ACTION_LENGTH * NUM_QR]),
                'wdsiA': tf.get_variable('wdsiA', [256, ACTION_LENGTH * NUM_QR]),
                'wdpiA': tf.get_variable('wdpiA', [256, ACTION_LENGTH * NUM_QR])
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [ACTION_LENGTH * NUM_QR], initializer = tf.random_uniform_initializer(-1, 1)),
                'bdsiA': tf.get_variable('bdsiA', [ACTION_LENGTH * NUM_QR], initializer = zero_init),
                'bdpiA': tf.get_variable('bdpiA', [ACTION_LENGTH * NUM_QR], initializer = zero_init)
            }

    def forward(self, data):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            batch_size = tf.shape(img)[0]
            imm = dense(im4, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            ims = dense(im4, self.weights['wdsiA'], self.biases['bdsiA'], activation = 'tanh')
            imp = dense(im4, self.weights['wdpiA'], self.biases['bdpiA'], activation = 'x')
            imm = tf.reshape(imm, [batch_size, ACTION_LENGTH, NUM_QR])
            ims = tf.reshape(ims, [batch_size, ACTION_LENGTH, NUM_QR])
            imp = tf.nn.softmax(tf.reshape(imp, [batch_size, ACTION_LENGTH, NUM_QR]), axis = 2)
            self.myu = imm
            self.sigma = tf.exp(ims)
            self.P = imp
        return (self.myu, self.sigma, self.P)
    
    def get_loss(self, img0, obj, oldmyu, oldsigma, oldprob, action, adv, ent_coef, CLIPRANGE = 0.2, CLIPRANGE2 = 0.05):
        with tf.variable_scope(self.name):
            num_batch = tf.shape(action)[0]
            action = tf.tile(tf.reshape(action, [num_batch, ACTION_LENGTH, 1]), [1, 1, NUM_QR])
            oldlogP = -2 * tf.square((action - oldmyu) / oldsigma) - 0.5*np.log(2*np.pi) - tf.log(oldsigma)
            myu, sigma, prob = self.forward(img0, obj)
            logP = -2 * tf.square((action - myu) / sigma) - 0.5*np.log(2*np.pi) - tf.log(sigma)
            
            oldP = tf.reduce_sum(oldprob * tf.exp(oldlogP), axis = 2)
            P = tf.reduce_sum(prob * tf.exp(logP), axis = 2)
            ratio = P / oldP #ratio = tf.exp(logP - oldlogP)
           
            # Defining Loss = - J is equivalent to max J
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE))) 

            entropy = tf.reduce_mean(ratio * tf.log(P))
            loss = pg_loss - ent_coef * entropy
            approxkl = tf.reduce_mean(-tf.log(ratio))
            return loss, clipfrac, entropy, approxkl, pg_loss, ratio

class CriticPPO():
    def __init__(self, name, encoder):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wd2iC': tf.get_variable('wd2iC', [STATE_LENGTH, 1], initializer = n_init(0, 0.06)),
            }
            self.biases = {
                'bd2iC': tf.get_variable('bd2iC', [1], initializer = zero_init),
            }

    def forward(self, data):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            res = dense(im4, self.weights['wd2iC'], self.biases['bd2iC'], activation = 'x')
        return res
    
    def get_loss(self, data, Vtarget, oldV0, CLIPRANGE = 0.2):
        with tf.variable_scope(self.name):
            vpred = self.forward(data)
            vpredclipped = oldV0 + tf.clip_by_value(vpred - oldV0, - CLIPRANGE, CLIPRANGE)
            # Unclipped value
            vf_losses1 = tf.square(vpred - Vtarget)
            # Clipped value
            vf_losses2 = tf.square(vpredclipped - Vtarget)
            vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            #vf_loss = tf.reduce_mean(vf_losses1)
            return vf_loss

class CriticContinuousSAC():
    def __init__(self, name, encoder1, encoder2, ACTION_LENGTH):
        self.name = name
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        with tf.variable_scope(self.name):
            self.Q1 = SubCriticContinuousSAC('Q1', self.encoder1, ACTION_LENGTH)
            self.Q2 = SubCriticContinuousSAC('Q2', self.encoder2, ACTION_LENGTH)

    def get_params(self):
        return self.Q1.get_params() + self.Q2.get_params() + self.encoder1.get_params() + self.encoder2.get_params()

    def forward(self, data, action):
        action = tf.tanh(action)
        Q1 = self.Q1.forward(data, action)
        Q2 = self.Q2.forward(data, action)
        Q = tf.minimum(Q1, Q2)
        return Q

    def get_loss(self, data, action, Qtarget):
        action = tf.tanh(action)
        Qf_loss1 = self.Q1.get_loss(data, action, Qtarget)
        Qf_loss2 = self.Q2.get_loss(data, action, Qtarget)
        return Qf_loss1 + Qf_loss2

class SubCriticContinuousSAC():
    def __init__(self, name, encoder, ACTION_LENGTH):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wd1iC': tf.get_variable('wd1iC', [ACTION_LENGTH + STATE_LENGTH, STATE_LENGTH], initializer = n_init(0, 0.06)),
                'wd2iC': tf.get_variable('wd2iC', [ACTION_LENGTH + STATE_LENGTH, 1], initializer = n_init(0, 0.06)),
            }
            self.biases = {
                'bd1iC': tf.get_variable('bd1iC', [STATE_LENGTH], initializer = zero_init),
                'bd2iC': tf.get_variable('bd2iC', [1], initializer = zero_init),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())# + self.enc.get_params()

    def forward(self, data, action):
        z0 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            da0 = tf.concat([z0, action], axis = 1)
            z1 = dense(da0, self.weights['wd1iC'], self.biases['bd1iC'], activation = 'relu')
            da1 = tf.concat([z1, action], axis = 1)
            res = dense(da1, self.weights['wd2iC'], self.biases['bd2iC'], activation = 'x')
        return res
    
    def get_loss(self, data, action, Qtarget):
        with tf.variable_scope(self.name):
            Qpred = self.forward(data, action)
            Qf_loss = tf.reduce_mean(tf.square(Qpred - Qtarget))
            return Qf_loss

class CriticDiscreteSAC():
    def __init__(self, name, encoder1, encoder2, ACTION_LENGTH):
        self.name = name
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        with tf.variable_scope(self.name):
            self.Q1 = SubCriticDiscreteSAC('Q1', self.encoder1, ACTION_LENGTH)
            self.Q2 = SubCriticDiscreteSAC('Q2', self.encoder2, ACTION_LENGTH)

    def get_params(self):
        return self.Q1.get_params() + self.Q2.get_params() + self.encoder1.get_params() + self.encoder2.get_params()

    def forward(self, data, action):
        Q1 = self.Q1.forward(data, action)
        Q2 = self.Q2.forward(data, action)
        Q = tf.minimum(Q1, Q2)
        return Q

    def get_loss(self, data, action, Qtarget):
        Qf_loss1 = self.Q1.get_loss(data, action, Qtarget)
        Qf_loss2 = self.Q2.get_loss(data, action, Qtarget)
        return Qf_loss1 + Qf_loss2

class SubCriticDiscreteSAC():
    def __init__(self, name, encoder, ACTION_LENGTH):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wd2iC': tf.get_variable('wd2iC', [STATE_LENGTH, ACTION_LENGTH], initializer = n_init(0, 0.06)),
            }
            self.biases = {
                'bd2iC': tf.get_variable('bd2iC', [ACTION_LENGTH], initializer = zero_init),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())# + self.enc.get_params()

    def forward(self, data, action):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            res = dense(im4, self.weights['wd2iC'], self.biases['bd2iC'], activation = 'x')
            Q = tf.reduce_sum(action * res, axis = 1, keep_dims = True)
            #print(Q.shape)
        return Q
    
    def get_loss(self, data, action, Qtarget):
        with tf.variable_scope(self.name):
            Qpred = self.forward(data, action)
            Qf_loss = tf.reduce_mean(tf.square(Qpred - Qtarget))
            return Qf_loss
