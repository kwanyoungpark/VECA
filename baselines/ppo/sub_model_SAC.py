import tensorflow as tf
from utils import *
import numpy as np
import sys
import os
import cv2 
import time
from constants import *

def log_prob(A, AR):
    myu, sigma = A
    res = -0.5 * ((AR - myu) / sigma) ** 2 - np.log(1 - np.tanh(AR) ** 2 + 1e-4)
    # - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    #print(res.shape)
    return res

class Model():
    def __init__(self, env, sess, encoderC1, encoderC10, encoderC2, encoderC20, encoderA, encoderA0, name = None): 
        self.env = env
        if name is None:
            name = env.name
        self.name = name
        NUM_AGENTS = self.env.num_agents
        ACTION_LENGTH = self.env.action_space
        if NUM_CHUNKS is None:
            BATCH_SIZE = 256
        else:
            BATCH_SIZE = (TIME_STEP * NUM_AGENTS) // NUM_CHUNKS
        if 'image' in self.env.observation_space:
            IMG_C, IMG_H, IMG_W = self.env.observation_space['image']
        if 'audio' in self.env.observation_space:
            WAV_C, WAV_LENGTH = self.env.observation_space['audio']
        if 'obj' in self.env.observation_space:
            VEC_OBJ = env.VEC_OBJ
            if VEC_OBJ: 
                NUM_OBJS = env.observation_space['obj']
        if 'touch' in self.env.observation_space:
            TACTILE_LENGTH = self.env.observation_space['touch']

        if self.env.mode == 'CONT':
            from agents import AgentContinuousSAC as Agent
            from agents import CriticContinuousSAC as Critic
        elif self.env.mode == 'DISC':
            from agents import AgentDiscreteSAC as Agent
            from agents import CriticDiscreteSAC as Critic
            #print('SAC does not support discrete action space!!!')
        
        with tf.variable_scope(self.name):
            # Plotting placeholders
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.raw_reward = tf.placeholder(tf.float32, [None, 1])
            self.rewardStd = tf.placeholder(tf.float32)
            self.helper_reward = tf.placeholder(tf.float32, [None, 1])
            self.tot_reward = tf.placeholder(tf.float32)
            self.lrA = tf.placeholder(tf.float32)
            self.ent_coef = tf.placeholder(tf.float32)

            # Inference placeholders
            if 'image' in env.observation_space:
                self.Iimg0 = tf.placeholder(tf.float32, shape = [NUM_AGENTS, IMG_H, IMG_W, IMG_C])
            if 'audio' in env.observation_space:
                self.Iwav0 = tf.placeholder(tf.float32, shape = [NUM_AGENTS, WAV_C * WAV_LENGTH])
            if 'obj' in env.observation_space:
                if VEC_OBJ:
                    self.Iobj = tf.placeholder(tf.float32, shape= [NUM_AGENTS, NUM_OBJS])
                else:
                    self.Iobj = tf.placeholder(tf.float32, shape= [NUM_AGENTS, IMG_H, IMG_W, 1])       
            if 'touch' in env.observation_space:
                self.Itouch = tf.placeholder(tf.float32, shape = [NUM_AGENTS, TACTILE_LENGTH])
            
            # Placeholders
            if 'image' in env.observation_space:
                self._img0 = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, IMG_C])
            if 'audio' in env.observation_space:
                self._wav0 = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, WAV_C * WAV_LENGTH])
            if 'obj' in env.observation_space:
                if VEC_OBJ:
                    self._obj = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, NUM_OBJS])
                else:
                    self._obj = tf.placeholder(tf.float32, shape= [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, 1])
            if 'touch' in env.observation_space:
                self._touch = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, TACTILE_LENGTH])

            if env.mode == 'CONT':
                self._myu = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
                self._sigma = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            elif env.mode == 'DISC':
                self._prob = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])

            self._actionReal = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            #self._advs = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, 1])
            self._Vtarget = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, 1])
            #self._oldV0 = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, 1])

            # Data storage(Only used in training)
            if 'image' in env.observation_space:
                self.Mimg0 = tf.get_variable("Mimg0", [NUM_CHUNKS, BATCH_SIZE, IMG_H, IMG_W, IMG_C], trainable = False)
            if 'audio' in env.observation_space:
                self.Mwav0 = tf.get_variable("Mwav0", [NUM_CHUNKS, BATCH_SIZE, WAV_C * WAV_LENGTH], trainable = False)
            if 'obj' in env.observation_space:
                if VEC_OBJ:
                    self.Mobj = tf.get_variable("Mobj", [NUM_CHUNKS, BATCH_SIZE, NUM_OBJS], trainable = False)
                else:
                    self.Mobj = tf.get_variable("Mobj", [NUM_CHUNKS, BATCH_SIZE, IMG_H, IMG_W, 1], trainable = False)
            if 'touch' in env.observation_space:
                self.Mtouch = tf.get_variable("Mtouch", [NUM_CHUNKS, BATCH_SIZE, TACTILE_LENGTH], trainable = False)

            if env.mode == 'CONT':
                self.Mmyu = tf.get_variable("Mmyu", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
                self.Msigma = tf.get_variable("Msigma", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
            elif env.mode == 'DISC':
                self.Mprob = tf.get_variable("Mprob", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.MactionReal = tf.get_variable("MactionReal", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
            #self.Madvs = tf.get_variable("Madvs", [NUM_CHUNKS, BATCH_SIZE, 1], trainable = False)
            self.MVtarget = tf.get_variable("MVtarget", [NUM_CHUNKS, BATCH_SIZE, 1], trainable = False)
            #self.MoldV0 = tf.get_variable("MoldV0", [NUM_CHUNKS, BATCH_SIZE, 1], trainable = False)
           
            # Real values
            if 'image' in env.observation_space:
                self.img0 = tf.get_variable("img0", [BATCH_SIZE, IMG_H, IMG_W, IMG_C], trainable = False)
            if 'audio' in env.observation_space:
                self.wav0 = tf.get_variable("wav0", [BATCH_SIZE, WAV_C * WAV_LENGTH], trainable = False)
            if 'obj' in env.observation_space:
                if env.VEC_OBJ:
                    self.obj = tf.get_variable("obj", [BATCH_SIZE, NUM_OBJS], trainable = False)
                else:
                    self.obj = tf.get_variable("obj", [BATCH_SIZE, IMG_H, IMG_W, 1], trainable = False)
            if 'touch' in env.observation_space:
                self.touch = tf.get_variable("touch", [BATCH_SIZE, TACTILE_LENGTH], trainable = False)

            if env.mode == 'CONT':
                self.myu = tf.get_variable("myu", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
                self.sigma = tf.get_variable("sigma", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
            elif env.mode == 'DISC':
                self.prob = tf.get_variable("prob", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.actionReal = tf.get_variable("actionReal", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
            #self.advs = tf.get_variable("advs", [BATCH_SIZE, 1], trainable = False)
            self.Vtarget = tf.get_variable("Vtarget", [BATCH_SIZE, 1], trainable = False)
            #self.oldV0 = tf.get_variable("oldV0", [BATCH_SIZE, 1], trainable = False)

            self._data, self.Mdata, self.data = [], [], []
            num_input = 0
            if 'image' in env.observation_space:
                self._data += [self._img0]
                self.Mdata += [self.Mimg0]
                self.data += [self.img0]
                num_input += 1
            if 'audio' in env.observation_space:
                self._data += [self._wav0]
                self.Mdata += [self.Mwav0]
                self.data += [self.wav0]
                num_input += 1
            if 'obj' in env.observation_space: 
                self._data += [self._obj]
                self.Mdata += [self.Mobj]
                self.data += [self.obj]
                num_input += 1
            if 'touch' in env.observation_space: 
                self._data += [self._touch]
                self.Mdata += [self.Mtouch]
                self.data += [self.touch]
                num_input += 1
            if env.mode == "CONT": 
                self._data += [self._myu, self._sigma]
                self.Mdata += [self.Mmyu, self.Msigma]
                self.data += [self.myu, self.sigma]
            elif env.mode == "DISC":
                self._data += [self._prob]
                self.Mdata += [self.Mprob]
                self.data += [self.prob] 
            self._data += [self._actionReal, self._Vtarget]
            self.Mdata += [self.MactionReal, self.MVtarget]
            self.data += [self.actionReal, self.Vtarget]

            self.memory_op_small = []
            for i in range(num_input):
                source = tf.reshape(self._data[i], self.Mdata[i].shape)#self.Mdata[i].shape)
                self.memory_op_small.append(tf.assign(self.Mdata[i], source, validate_shape = True))

            self.memory_op = []
            for i in range(len(self._data)):
                source = tf.reshape(self._data[i], self.Mdata[i].shape)#self.Mdata[i].shape)
                self.memory_op.append(tf.assign(self.Mdata[i], source, validate_shape = True))

            self.data_op = []
            for i in range(NUM_CHUNKS):
                self.data_op.append([])
            for i in range(NUM_CHUNKS):
                for j in range(len(self.data)):
                    #print(self.Mdata[j].shape)
                    self.data_op[i].append(tf.assign(self.data[j], self.Mdata[j][i], validate_shape = True)) 
            
            self.encC1 = encoderC1
            self.encC2 = encoderC2
            self.encA = encoderA
            self.critic = Critic('mainC', self.encC1, self.encC2, self.env.action_space)
            self.brain = Agent('mainA', self.encA, self.critic, self.env.action_space)
            self.encC10 = encoderC10
            self.encC20 = encoderC20
            self.encA0 = encoderA0
            self.critic0 = Critic('targetC', self.encC10, self.encC20, self.env.action_space)
            self.brain0 = Agent('targetA', self.encA0, self.critic0, self.env.action_space)
            #self.gen = Generator('mainG', self.enc)
            
            # Action value from current policy/freezed policy
            # We use FREEZED policy in exploration

            # Track variance of reward
            self.rewardTrack = rewardTracker(0)
            self.raw_rewardTrack = rewardTracker(0.95)

            self.Idat, self.dat = [], []

            if 'image' in env.observation_space:
                self.Idat.append(self.Iimg0)
                self.dat.append(self.img0)
            else:
                self.Idat.append(None)
                self.dat.append(None)
            if 'audio' in env.observation_space:
                self.Idat.append(self.Iwav0)
                self.dat.append(self.wav0)
            else:
                self.Idat.append(None)
                self.dat.append(None)
            if 'obj' in env.observation_space:
                self.Idat.append(self.Iobj)
                self.dat.append(self.obj)
            else:
                self.Idat.append(None)
                self.dat.append(None)
            if 'touch' in env.observation_space:
                self.Idat.append(self.Itouch)
                self.dat.append(self.touch)
            else:
                self.Idat.append(None)
                self.dat.append(None)

            self.inferA = self.brain0.forward(self.Idat)
            self.inferAR = self.inferA[0] + self.inferA[1] * tf.random.normal(tf.shape(self.inferA[0]))
            self.inferV0 = self.critic0.forward(self.Idat, self.inferAR)
            #print(self.inferA.shape, self.inferAR.shape, self.inferV0.shape)

            self.oldA = self.brain0.forward(self.dat)
            self.A = self.brain.forward(self.dat)

            # Agent Loss
            if env.mode == 'CONT':
                self.oldAR = tf.tanh(self.oldA[0] + self.oldA[1] * tf.random.normal(tf.shape(self.oldA[0])))
                self.loss_Agent, self.entropy, self.approxkl, self.pg_loss, self.ratio, self.obs = self.brain.get_loss(self.dat, self.myu, self.sigma, self.ent_coef)
            if env.mode == 'DISC':
                self.oldAR = gumbel_softmax(tf.log(self.oldA))
                self.loss_Agent, self.entropy, self.approxkl, self.pg_loss, self.ratio, self.obs = self.brain.get_loss(self.dat, self.prob, self.ent_coef)
            
            self.V0 = self.critic0.forward(self.dat, self.oldAR)
            self.V = self.critic.forward(self.dat, self.oldAR)

            # Critic Loss
            self.loss_Critic = self.critic.get_loss(self.dat, self.actionReal, self.Vtarget)
            self.loss = self.loss_Agent + 0.5 * self.loss_Critic

            # Regularization Loss
            train_vars = tf.trainable_variables()
            #self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'b' not in v.name]) * 0.0001
            #self.loss = self.ICM.lossI
            #train_varsA = self.brain.get_params()
            train_varsA = self.brain.get_params()
            train_varsC = self.critic.get_params()
            print('A', train_varsA)
            print('C', train_varsC)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                #self.optC = makeOptimizer(self.lr / TRAIN_LOOP, self.loss_Critic, decay = False)
                self.accA, self.init_accA, self.optA, self.gradA = makeOptimizer(self.lrA, self.loss_Agent, decay = False, var_list = train_varsA)
                self.accC, self.init_accC, self.optC, self.gradC = makeOptimizer(self.lrA, self.loss_Critic, decay = False, var_list = train_varsC)
            
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
            #config.log_device_placement = True
            self.sess = sess
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #tf.summary.scalar('inverse_loss', self.lossI)
            #tf.summary.scalar('lr', self.lr)
            summaries = []
            summaries.append(tf.summary.scalar('helper_reward', tf.reduce_mean(self.helper_reward)))
            summaries.append(tf.summary.scalar('agent_loss', self.loss_Agent))
            summaries.append(tf.summary.scalar('V', tf.reduce_mean(self.V)))
            summaries.append(tf.summary.scalar('critic_loss', self.loss_Critic))
            summaries.append(tf.summary.scalar('relative_critic_loss', self.loss_Critic / tf.reduce_mean(self.reward)))
            summaries.append(tf.summary.scalar('targetV', tf.reduce_mean(self.V0)))
            #summaries.append(tf.summary.scalar('clipfrac', self.clipfrac))
            summaries.append(tf.summary.scalar('raw_reward', tf.reduce_mean(self.raw_reward)))
            summaries.append(tf.summary.scalar('reward', tf.reduce_mean(self.reward)))
            summaries.append(tf.summary.scalar('rewardStd', self.rewardStd))
            summaries.append(tf.summary.scalar('lr', self.lrA))
      
            if env.SIM == 'VECA':
                pass
                #summaries.append(tf.summary.image('image0', tf.cast(255*self.img0[:, :, :, 0:IMG_C//2], tf.uint8), max_outputs = 8))
            elif env.SIM == 'ATARI':
                #img = tf.reshape(tf.transpose(self.img0, [0, 3, 1, 2]), [BATCH_SIZE * IMG_C, IMG_H, IMG_W, 1])
                img = tf.reshape(self.img0[:, :, :, 0], [BATCH_SIZE, IMG_H, IMG_W, 1])
                #tf.summary.image('image0', tf.cast(255*self.img0, tf.uint8), max_outputs = 64)
            #tf.summary.image('obj', tf.cast(255*tf.transpose(self.obj[0:1, :, :, :], [3,1,2,0]), tf.uint8), max_outputs = 64)
            #self.enc.summarize_filters()
            #tf.summary.image('filter', self.enc.get_filter())
            #tf.summary.image('bin_edge', self.enc.im0)
            #tf.summary.image('im0', tf.cast(127+127*tf.transpose(self.enc.im0[0:1,:,:,:],[3,1,2,0]), tf.uint8), max_outputs = 32)
            #tf.summary.image('im1', tf.cast(127+127*tf.transpose(self.enc.im1[0:1,:,:,:],[3,1,2,0]), tf.uint8), max_outputs = 32)
            #tf.summary.image('im2', tf.cast(127+127*tf.transpose(self.enc.im2[0:1,:,:,:],[3,1,2,0]), tf.uint8), max_outputs = 32)
            summaries.append(tf.summary.scalar('entropy', self.entropy))
            summaries.append(tf.summary.scalar('approxkl', self.approxkl))
            summaries.append(tf.summary.scalar('cumulative_raw_reward', self.tot_reward))
            #tf.summary.histogram('activation', self.enc.obf)
            #tf.summary.histogram('GAPactivation', self.enc.obp)
            #tf.summary.image('GAPactivation', tf.cast(255*tf.reshape(self.enc.obp, [1, BATCH_SIZE, -1, 1]), tf.uint8))
            if 'obj' in env.observation_space:
                 summaries.append(tf.summary.image('objvec', tf.cast(255*tf.reshape(self.Mobj, [1, NUM_CHUNKS*BATCH_SIZE, NUM_OBJS, 1]), tf.uint8)))
            #tf.summary.scalar('recon_loss', self.loss_recon)
            summaries.append(tf.summary.scalar('loss', self.loss))
            print(tf.trainable_variables())
            train_vars = tf.trainable_variables(scope = name + '/targetA')
            print(train_vars)
            for var in train_vars:
                summaries += variable_summaries(var)
            train_vars = tf.trainable_variables(scope = name + '/targetC')
            print(train_vars)
            for var in train_vars:
                summaries += variable_summaries(var)

            train_vars = []
            train_vars += tf.global_variables(scope = name + '/targetA')
            train_vars += tf.global_variables(scope = name + '/targetC')
            self.loader = tf.train.Saver(train_vars)
            print(train_vars)
     
            self.merge = tf.summary.merge(summaries)
            self.writer = tf.summary.FileWriter('./log/' + env.name + '/', self.sess.graph)
            self.global_step = 0

            #self.saver = tf.train.Saver(train_vars)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            #self.sess.run(tf.global_variables_initializer())

            self.hard_copy_op = []
            self.soft_copy_op = []
            self.isSame = []

            main_varsA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/mainA')
            target_varsA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/targetA')

            print('A')
            for main_var, target_var in zip(main_varsA, target_varsA):
                print(main_var.name, target_var.name)
                self.isSame.append(tf.reduce_sum(tf.abs(main_var - target_var)))
                #print(np.sum(self.sess.run(target_var)))
                self.hard_copy_op.append(target_var.assign(main_var.value()))
                self.soft_copy_op.append(target_var.assign(main_var.value()))
                #self.soft_copy_op.append(target_var.assign(target_var.value() * 0.99 + main_var.value() * 0.01))

            main_varsC = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/mainC')
            target_varsC = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/targetC')
            
            print('C')
            for main_var, target_var in zip(main_varsC, target_varsC):
                print(main_var.name, target_var.name)
                #self.isSame.append(tf.reduce_sum(tf.abs(main_var - target_var)))
                self.hard_copy_op.append(target_var.assign(main_var.value()))
                self.soft_copy_op.append(target_var.assign(target_var.value() * 0.99 + main_var.value() * 0.01))

    def split_dict(self, data, num_chunks):
        dict_chunks = []
        for i in range(num_chunks):
            dict_chunks.append({})
        for key in data.keys():
            value = np.array(data[key])
            if len(value.shape) == 0:
                for i in range(num_chunks):
                    dict_chunks[i][key] = value.copy()
            else:
                batch_size = value.shape[0]
                valueR = np.reshape(value, [num_chunks, batch_size // num_chunks] + list(value.shape[1:]))
                for i in range(num_chunks):
                    dict_chunks[i][key] = valueR[i]
        return dict_chunks

    def get_action(self, data):
        #print(self.env.observation_space)
        dict_all = {}
        if 'image' in self.env.observation_space:
            IMG_C, IMG_H, IMG_W = self.env.observation_space['image']
            img = data['image']
            img = np.transpose(np.reshape(img, [-1, IMG_C, IMG_H, IMG_W]), [0, 2, 3, 1])
            dict_all.update({self.Iimg0: img})
        if 'audio' in self.env.observation_space:
            WAV_C, WAV_LENGTH = self.env.observation_space['audio']
            wav = data['audio']
            wav = np.reshape(wav, [-1, WAV_C * WAV_LENGTH])
            dict_all.update({self.Iwav0: wav})
        if 'obj' in self.env.observation_space:
            VEC_OBJ = self.env.VEC_OBJ
            obj = data['obj']
            if VEC_OBJ is False:
                obj = np.transpose(np.reshape(obj, [-1, NUM_DEGS, IMG_H, IMG_W]), [0, 2, 3, 1])
                obj = obj[:,:,:,4:5]
            dict_all.update({self.Iobj: obj})
        if 'touch' in self.env.observation_space:
            TACTILE_LENGTH = self.env.observation_space['touch']
            touch = data['touch']
            touch = np.reshape(touch, [-1, TACTILE_LENGTH])
            dict_all.update({self.Itouch: touch})

        if self.env.mode == 'CONT':
            myu, sigma = self.sess.run(self.inferA, feed_dict = dict_all)
            action = np.random.normal(myu, sigma)
            return myu, sigma, action
        elif self.env.mode == 'DISC':
            prob = self.sess.run(self.inferA, feed_dict = dict_all)
            action = np.zeros_like(prob)
            for i in range(self.env.num_envs):
                ind = np.random.choice(self.env.action_space, 1, p = prob[i])[0]
                action[i][ind] = 1.
            return prob, action

    def make_batch(self, data, ent_coef, add_merge = False, num_chunks = None, update_gpu = False, lr = None):
        NUM_AGENTS = self.env.num_envs * self.env.agents_per_env
        if 'image' in self.env.observation_space:
            img0 = data['img0']
            img1 = data['img1']
            IMG_C, IMG_H, IMG_W = self.env.observation_space['image']
        if 'audio' in self.env.observation_space:
            wav0 = data['wav0']
            wav1 = data['wav1']
            WAV_C, WAV_LENGTH = self.env.observation_space['audio']
        if 'obj' in self.env.observation_space:
            obj = data['obj']
            if self.env.VEC_OBJ == False:
                NUM_DEGS, _, _ = self.env.observation_space['obj']
                obj = np.transpose(np.reshape(obj, [TIME_STEP, NUM_AGENTS, NUM_DEGS, IMG_H, IMG_W]), [1, 0, 3, 4, 2])
                obj = np.reshape(obj, [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, NUM_DEGS])
                obj = obj[:, :, :, 4:5]
            else:
                NUM_OBJS = self.env.observation_space['obj']
                obj = np.transpose(np.reshape(obj, [TIME_STEP, NUM_AGENTS, NUM_OBJS]), [1, 0, 2])
                obj = np.reshape(obj, [NUM_AGENTS*TIME_STEP, NUM_OBJS])
        if 'touch' in self.env.observation_space:
            touch0 = data['touch0']
            touch1 = data['touch1']
            TACTILE_LENGTH = self.env.observation_space['touch']

        actionReal, helper_reward, raw_reward = data['action'], data['helper_reward'], data['raw_reward']
        done = data['done']
        ACTION_LENGTH = self.env.action_space

        if 'image' in self.env.observation_space:
            img0 = np.transpose(np.reshape(img0, [TIME_STEP, NUM_AGENTS, IMG_C, IMG_H, IMG_W]), [1, 0, 3, 4, 2])
            img1 = np.transpose(np.reshape(img1, [TIME_STEP, NUM_AGENTS, IMG_C, IMG_H, IMG_W]), [1, 0, 3, 4, 2])
        if 'audio' in self.env.observation_space:
            wav0 = np.transpose(np.reshape(wav0, [TIME_STEP, NUM_AGENTS, WAV_C * WAV_LENGTH]), [1, 0, 2])
            wav1 = np.transpose(np.reshape(wav1, [TIME_STEP, NUM_AGENTS, WAV_C * WAV_LENGTH]), [1, 0, 2])
        if 'touch' in self.env.observation_space:
            touch0 = np.transpose(np.reshape(touch0, [TIME_STEP, NUM_AGENTS, TACTILE_LENGTH]), [1, 0, 2])
            touch1 = np.transpose(np.reshape(touch1, [TIME_STEP, NUM_AGENTS, TACTILE_LENGTH]), [1, 0, 2])
        actionReal = np.transpose(np.reshape(actionReal, [TIME_STEP, NUM_AGENTS, ACTION_LENGTH]), [1, 0, 2])
        done = np.transpose(np.reshape(done, [TIME_STEP, NUM_AGENTS]), [1, 0])
        
        actionReal = np.reshape(actionReal, [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
        done = np.reshape(done, [NUM_AGENTS*TIME_STEP, 1])
        helper_reward = np.transpose(np.reshape(helper_reward, [TIME_STEP, NUM_AGENTS]), [1, 0])
        raw_reward = np.transpose(np.reshape(raw_reward, [TIME_STEP, NUM_AGENTS]), [1, 0])

        dict_mem = {}
        if 'image' in self.env.observation_space:
            img0 = np.reshape(img0, [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, IMG_C])
            img1 = np.reshape(img1, [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, IMG_C])
            dict_mem.update({self._img0: img1})
        if 'audio' in self.env.observation_space:
            wav0 = np.reshape(wav0, [NUM_AGENTS*TIME_STEP, WAV_C * WAV_LENGTH])
            wav1 = np.reshape(wav1, [NUM_AGENTS*TIME_STEP, WAV_C * WAV_LENGTH])
            dict_mem.update({self._wav0: wav1})
        if 'obj' in self.env.observation_space:
            dict_mem.update({self._obj: obj})
        if 'touch' in self.env.observation_space:
            touch0 = np.reshape(touch0, [NUM_AGENTS*TIME_STEP, TACTILE_LENGTH])
            touch1 = np.reshape(touch1, [NUM_AGENTS*TIME_STEP, TACTILE_LENGTH])
            dict_mem.update({self._touch: touch1})
        self.sess.run(self.memory_op_small, feed_dict = dict_mem)
       
        if self.env.mode == 'CONT':
            myu, sigma, V0 = [], [], []
            chunk_size = (NUM_AGENTS * TIME_STEP) // num_chunks
            for i in range(num_chunks):
                self.sess.run(self.data_op[i])
                Ai, ARi, V0i = self.sess.run([self.oldA, self.oldAR, self.V0])
                myui, sigmai = Ai
                myu.append(myui)
                sigma.append(sigmai)
                V0.append(V0i - ent_coef * np.mean(log_prob(Ai, ARi), axis = 1, keepdims=True))
                #print(V0i.shape, myui.shape, ARi.shape)
            myu = np.reshape(np.array(myu), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            sigma = np.reshape(np.array(sigma), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            V0 = np.reshape(np.array(V0), [NUM_AGENTS*TIME_STEP, 1]) 
        if self.env.mode == 'DISC':
            prob, V0 = [], []
            chunk_size = (NUM_AGENTS * TIME_STEP) // num_chunks
            for i in range(num_chunks):
                self.sess.run(self.data_op[i])
                Ai, ARi, V0i = self.sess.run([self.oldA, self.oldAR, self.V0])
                probi = Ai
                prob.append(probi)
                #V0.append(V0i - ent_coef * np.sum(np.log(probi + 1e-4) * ARi, axis = 1, keepdims=True))
                V0.append(V0i - ent_coef * np.mean(np.log(probi + 1e-4) * ARi, axis = 1, keepdims=True))

                #print(V0i.shape, probi.shape, ARi.shape)
            prob = np.reshape(np.array(prob), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            V0 = np.reshape(np.array(V0), [NUM_AGENTS*TIME_STEP, 1])
         
        reward = helper_reward + raw_reward
        cum_reward = np.zeros_like(reward)
        done = np.reshape(done, [NUM_AGENTS, TIME_STEP])

        T = TIME_STEP
        for i in range(NUM_AGENTS):
            done[i][-1] = reward[i][-1]
            for t in reversed(range(T-1)):
                if done[i][t]:
                    cum_reward[i][t] = reward[i][t]
                else:
                    cum_reward[i][t] = GAMMA * cum_reward[i][t+1] + reward[i][t]

        for i in range(TIME_STEP):
            self.rewardTrack.update(cum_reward[:, i])

        for i in range(NUM_AGENTS):
            done[i][-1] = raw_reward[i][-1]
            for t in reversed(range(T-1)):
                if done[i][t]:
                    cum_reward[i][t] = raw_reward[i][t]
                else:
                    cum_reward[i][t] = GAMMA * cum_reward[i][t+1] + raw_reward[i][t]

        for i in range(TIME_STEP):
            self.raw_rewardTrack.update(cum_reward[:, i])

        rewardStd = self.rewardTrack.get_std()
        tot_reward = self.raw_rewardTrack.X0.mean()
        reward /= rewardStd
        #print('std :', rewardStd)
        V0 = np.reshape(V0, [NUM_AGENTS, TIME_STEP])
        Vtarget = np.where(done, reward, reward + GAMMA * V0)
        #Vtarget = np.zeros_like(V0)
        #reward *= (1 - GAMMA)
        '''
        for i in range(NUM_AGENTS):
            if done[i][-1]:
                Vtarget[i][-1] = reward[i][-1]
            else:
                Vtarget[i][-1] = reward[i][-1] + GAMMA * V[i]
            for t in reversed(range(T - 1)):
                if done[i][t]:
                    Vtarget[i][t] = reward[i][t]
                else:
                    Vtarget[i][t] = reward[i][t] + GAMMA * V0[i][t+1]
        '''
        print(np.mean(V0 ** 2))
        print(np.mean(Vtarget ** 2))
        print(np.mean((Vtarget - V0) ** 2))
        Vtarget = np.reshape(Vtarget, [NUM_AGENTS * TIME_STEP, 1])
        #V0 = np.reshape(V0, [NUM_AGENTS * TIME_STEP, 1])
        reward = np.reshape(reward, [NUM_AGENTS * TIME_STEP, 1])
        helper_reward = np.reshape(helper_reward, [NUM_AGENTS * TIME_STEP, 1])
        raw_reward = np.reshape(raw_reward, [NUM_AGENTS * TIME_STEP, 1])
        actionReal = np.reshape(actionReal, [NUM_AGENTS * TIME_STEP, ACTION_LENGTH])
        
        #advs = helper_reward.copy()
        
        #advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        '''
        for i in range(NUM_AGENTS * TIME_STEP):
            if advs[i][0] < 0:
                advs[i][0] = -advs[i][0]
                actionReal[i] = 2 * myu[i] - actionReal[i]
        '''
        #real_acc = np.sum(np.sum(obj * prob, axis = 1) * mask) / (np.sum(mask) + 1e-8)

        dict_mem = {}
        if 'image' in self.env.observation_space:
            dict_mem.update({self._img0: img0})
        if 'audio' in self.env.observation_space:
            dict_mem.update({self._wav0: wav0})
        if 'obj' in self.env.observation_space:
            dict_mem.update({self._obj: obj})
        if 'touch' in self.env.observation_space:
            dict_mem.update({self._touch: touch0})
        self.sess.run(self.memory_op_small, feed_dict = dict_mem)
       
        if self.env.mode == 'CONT':
            myu, sigma = [], []
            chunk_size = (NUM_AGENTS * TIME_STEP) // num_chunks
            for i in range(num_chunks):
                self.sess.run(self.data_op[i])
                Ai = self.sess.run(self.oldA)
                myui, sigmai = Ai
                myu.append(myui)
                sigma.append(sigmai)
                #print(V0i.shape, myui.shape, ARi.shape)
            myu = np.reshape(np.array(myu), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            sigma = np.reshape(np.array(sigma), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
        if self.env.mode == 'DISC':
            prob = []
            chunk_size = (NUM_AGENTS * TIME_STEP) // num_chunks
            for i in range(num_chunks):
                self.sess.run(self.data_op[i])
                Ai = self.sess.run(self.oldA)
                probi = Ai
                prob.append(probi)
            prob = np.reshape(np.array(prob), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
 
        if update_gpu:
            dict_all = {}
            if 'image' in self.env.observation_space:
                dict_all.update({self._img0:img0})
            if 'audio' in self.env.observation_space:
                dict_all.update({self._wav0:wav0})
            if 'obj' in self.env.observation_space:
                dict_all.update({self._obj:obj})
            if 'touch' in self.env.observation_space:
                dict_all.update({self._touch:touch0})
            if self.env.mode == 'CONT':
                dict_all.update({self._myu:myu, self._sigma:sigma})
            if self.env.mode == 'DISC':
                dict_all.update({self._prob:prob})
            dict_all.update({self._Vtarget: Vtarget, self._actionReal: actionReal})
            self.sess.run(self.memory_op, feed_dict = dict_all)
        if add_merge:
            dict_all = {self.reward: reward, self.helper_reward:helper_reward, self.raw_reward: raw_reward, self.rewardStd: rewardStd, self.tot_reward: tot_reward, self.lrA:lr, self.ent_coef:ent_coef}
            TlA, TlC, Tpg_loss, Tratio, Tloss = 0., 0., 0., 0., 0.
            self.sess.run(self.init_accA)
            for i in range(num_chunks):
                self.sess.run(self.data_op[i])
                lA, lC, pg_loss, loss, ratio, _ = self.sess.run([self.loss_Agent, self.loss_Critic, self.pg_loss, self.loss, self.ratio, self.accA], feed_dict = {self.ent_coef:ent_coef})
                TlA += lA / num_chunks
                TlC += lC / num_chunks
                Tpg_loss += pg_loss / num_chunks
                Tloss += loss / num_chunks
                Tratio = max(np.max(ratio), Tratio)

                '''
                obs = self.sess.run(self.obs)
                _prob1, _prob0, _logP0, _logP = obs
                prob0 = self.sess.run(self.oldA)
                prob1 = self.sess.run(self.A)
                _img0 = self.sess.run(self.img0)
                print(_img0.min(), _img0.max())
                print('normalized?', advs.mean(), advs.std())
                for j in range(chunk_size):
                    with np.printoptions(precision=3, suppress = True):
                        print(prob0[j], prob1[j], actionReal[j+i*chunk_size], advs[j])
                        #print(myu0[j], sigma0[j], myu1[j], sigma1[j])
                        #print(_myu0[j], _sigma0[j], _myu1[j], _sigma1[j])
                        print(_logP0[j], _logP[j], ratio[j])
                        print(np.sum(np.square(_img0[j] - img0[i*chunk_size])), np.sum(np.square(img0[j+i*chunk_size]-img0[i*chunk_size])))
                        #print(np.where(np.all(myu == myu0[j], axis = 1)))
                print("Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f}".format(loss, lA, lC, np.max(ratio), pg_loss))
                '''
                '''        
                obs = self.sess.run(self.obs)
                _myu1, _sigma1, _myu0, _sigma0, _logP0, _logP = obs
                myu0, sigma0 = self.sess.run(self.oldA)
                myu1, sigma1 = self.sess.run(self.A)
                if i > 0:
                    continue
                #print('normalized?', advs.mean(), advs.std())
                for j in range(chunk_size):
                    #if myu0[j][0] != myu1[j][0]:
                    #print(myu0[j], sigma0[j], myu1[j], sigma1[j], advs[j])
                    print(myu0[j], sigma0[j], myu1[j], sigma1[j])
                    print(_myu0[j], _sigma0[j], _myu1[j], _sigma1[j])
                    print(_logP0[j], _logP[j], ratio[j])
                    print(np.where(np.all(myu == myu0[j], axis = 1)))
                print("Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f}".format(loss, lA, lC, np.max(ratio), pg_loss))
                '''
            TgradA = self.sess.run(self.gradA)
            print("Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f} lr {:.5f}".format(Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, lr))

            return Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, dict_all 
            
        #data = [advs, img0, obj, myu, sigma, Vtarget, V0, actionReal]
        #return data

    def debug_merge(self, dict_all):
        summary = self.sess.run(self.merge, feed_dict = dict_all)
        self.writer.add_summary(summary, self.global_step)

    def trainA(self, lr, num_chunks = None, ent_coef = None):
        self.sess.run([self.init_accA, self.init_accC])
        approxkl = 0.
        for i in range(num_chunks):
            self.sess.run(self.data_op[i])
            _, _, Tapproxkl = self.sess.run([self.accA, self.accC, self.approxkl], feed_dict = {self.ent_coef:ent_coef})
            approxkl += Tapproxkl / num_chunks
            if (i+1) % NUM_UPDATE == 0:
                self.sess.run([self.optC, self.optA], feed_dict = {self.lrA:lr, self.ent_coef:ent_coef})
                #self.sess.run(self.optC, feed_dict = {self.lrA:lr, self.ent_coef:ent_coef})
                self.sess.run([self.init_accC, self.init_accA])
                #self.sess.run(self.init_accA)
        return approxkl

    def updateNetwork(self):
        print('updating networks A and C...')
        #print(self.sess.run(self.isSame))
        self.sess.run(self.soft_copy_op)
        #print(self.sess.run(self.isSame))
        print('updated!')

    def save(self, num):
        self.saver.save(self.sess, './model/my-model', global_step = self.global_step)

    def load(self, model_name, global_step):
        self.loader.restore(self.sess, model_name)
        self.global_step = global_step
