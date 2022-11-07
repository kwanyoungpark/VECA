import tensorflow as tf
from utils import *
import numpy as np
import sys
import os
import time
from constants import *
from agents import CriticPPO as Critic
from agents import AgentContinuousPPO as Agent
from munch import Munch

class Model():
    def __init__(self, env, sess, encoder, encoder0, obs_sample, tag,name = None): 
        self.env = env
        self.tag = tag
        self.sess = sess
        if name is None:
            name = env.name
        self.name = name
        self.global_step = 0
        NUM_AGENTS = self.env.num_agents
        ACTION_LENGTH = self.env.action_space
        if NUM_CHUNKS is None:
            BATCH_SIZE = 256
        else:
            BATCH_SIZE = (TIME_STEP * NUM_AGENTS) // NUM_CHUNKS
        '''
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
        '''

        obshape = Munch({k:v.shape for k,v in obs_sample.items()})
        data_struct = Munch({
            "myu": (NUM_AGENTS, ACTION_LENGTH),
            "sigma": (NUM_AGENTS, ACTION_LENGTH),
            "actionReal": (NUM_AGENTS, ACTION_LENGTH),
            "advs": (NUM_AGENTS, 1),
            "Vtarget": (NUM_AGENTS, 1),
            "oldV0": (NUM_AGENTS, 1),
        })
        data_struct.update(obshape)

        
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
            '''
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
            if RNN:
                self.Istate = tf.placeholder(tf.float32, shape = [NUM_AGENTS, STATE_LENGTH])
            '''
            self.data_inf = {k : tf.placeholder(tf.float32, shape = v) for k,v in obshape.items()}

            # Placeholders
            self.data_place = {k : tf.placeholder(tf.float32, shape = (TIME_STEP,) + v) for k,v in data_struct.items()}
            '''
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
            if RNN:
                self._state = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, STATE_LENGTH])

            self._myu = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            self._sigma = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])

            self._actionReal = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
            self._advs = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, 1])
            self._Vtarget = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, 1])
            self._oldV0 = tf.placeholder(tf.float32, shape = [NUM_AGENTS*TIME_STEP, 1])
            '''

            # Data storage(Only used in training)
            self.data_mem = {k : tf.get_variable("mem/" + k,tf.float32, shape = (NUM_CHUNKS, BATCH_SIZE) + v[1:]) for k,v in data_struct.items()}
            '''
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
            if RNN:
                self.Mstate = tf.get_variable("Mstate", [NUM_CHUNKS, BATCH_SIZE, STATE_LENGTH], trainable = False)

            self.Mmyu = tf.get_variable("Mmyu", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.Msigma = tf.get_variable("Msigma", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.MactionReal = tf.get_variable("MactionReal", [NUM_CHUNKS, BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.Madvs = tf.get_variable("Madvs", [NUM_CHUNKS, BATCH_SIZE, 1], trainable = False)
            self.MVtarget = tf.get_variable("MVtarget", [NUM_CHUNKS, BATCH_SIZE, 1], trainable = False)
            self.MoldV0 = tf.get_variable("MoldV0", [NUM_CHUNKS, BATCH_SIZE, 1], trainable = False)
            '''
           
            # Real values 
            self.data_real = {k : tf.get_variable("real/" + k,tf.float32, shape = (BATCH_SIZE,) + v) for k,v in data_struct.items()}
            '''
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
            if RNN:
                self.state = tf.get_variable("RNN", [BATCH_SIZE, STATE_LENGTH], trainable = False)

            self.myu = tf.get_variable("myu", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.sigma = tf.get_variable("sigma", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.actionReal = tf.get_variable("actionReal", [BATCH_SIZE, ACTION_LENGTH], trainable = False)
            self.advs = tf.get_variable("advs", [BATCH_SIZE, 1], trainable = False)
            self.Vtarget = tf.get_variable("Vtarget", [BATCH_SIZE, 1], trainable = False)
            self.oldV0 = tf.get_variable("oldV0", [BATCH_SIZE, 1], trainable = False)
            '''
            '''
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
            if RNN:
                self._data += [self._state]
                self.Mdata += [self.Mstate]
                self.data += [self.state]
            self._data += [self._myu, self._sigma]
            self.Mdata += [self.Mmyu, self.Msigma]
            self.data += [self.myu, self.sigma]
            self._data += [self._actionReal, self._advs, self._Vtarget, self._oldV0]
            self.Mdata += [self.MactionReal, self.Madvs, self.MVtarget, self.MoldV0]
            self.data += [self.actionReal, self.advs, self.Vtarget, self.oldV0]
            '''

            # Define Memory & Data Operation
            self.memory_op_small = [
                tf.assign(self.data_mem[key], tf.reshape(self.data_place[key], self.data_mem[key].shape), validate_shape = True)
                for key in obshape.keys()
            ]
            self.memory_op = [
                tf.assign(self.data_mem[key], tf.reshape(self.data_place[key], self.data_mem[key].shape), validate_shape = True)
                for key in data_struct.keys()
            ]
            self.data_op = [
                [
                    tf.assign(self.data_real[key], self.data_mem[key][i], validate_shape = True) 
                    for key in data_struct.keys()
                ]
                for i in range(NUM_CHUNKS)
            ]
            
            # Model Init 
            self.enc = encoder
            self.brain = Agent('mainA', self.enc, self.env.action_space)
            self.critic = Critic('mainC', self.enc)
            self.enc0 = encoder0
            self.brain0 = Agent('targetA', self.enc0, self.env.action_space)
            self.critic0 = Critic('targetC', self.enc0)
            
            # Action value from current policy/freezed policy
            # We use FREEZED policy in exploration


            # Accumulate different modality in single input
            self.Idat, self.dat = self.data_inf, {k:self.data_real[k] for k in obshape.keys()}
            '''
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
            if RNN:
                self.Idat.append(self.Istate)
                self.dat.append(self.state)
            '''

            # Forward Propagation 
            self.inferA = self.brain0.forward(self.Idat)
            self.inferV0 = self.critic0.forward(self.Idat)
            if RNN:
                self.inferS = self.enc0.forward(self.Idat)
            self.oldA = self.brain0.forward(self.dat)
            self.A = self.brain.forward(self.dat)
            self.V0 = self.critic0.forward(self.dat)
            self.rewardTrack = rewardTracker(0) # Track variance of reward 
            self.raw_rewardTrack = rewardTracker(0.95)

            # Loss definition
            self.loss_Agent, self.clipfrac, self.entropy, self.approxkl, self.pg_loss, self.ratio, self.obs = self.brain.get_loss(self.dat, self.myu, self.sigma, self.actionReal, self.advs, self.ent_coef)
            self.loss_Critic = self.critic.get_loss(self.dat, self.Vtarget, self.oldV0)
            self.loss = self.loss_Agent + 0.5 * self.loss_Critic

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.accA, self.init_accA, self.optA, self.gradA = makeOptimizer(self.lrA, self.loss, decay = False)
            
            # tf.Summary & Writer
            summaries = []
            summaries.append(tf.summary.scalar('helper_reward', tf.reduce_mean(self.helper_reward)))
            summaries.append(tf.summary.scalar('agent_loss', self.loss_Agent))
            summaries.append(tf.summary.scalar('expected_reward(V)', tf.reduce_mean(self.V0)))
            summaries.append(tf.summary.scalar('critic_loss', self.loss_Critic))
            summaries.append(tf.summary.scalar('relative_critic_loss', self.loss_Critic / tf.reduce_mean(self.reward)))
            summaries.append(tf.summary.scalar('V0', tf.reduce_mean(self.V0)))
            summaries.append(tf.summary.scalar('clipfrac', self.clipfrac))
            summaries.append(tf.summary.scalar('raw_reward', tf.reduce_mean(self.raw_reward)))
            summaries.append(tf.summary.scalar('reward', tf.reduce_mean(self.reward)))
            summaries.append(tf.summary.scalar('rewardStd', self.rewardStd))
            summaries.append(tf.summary.scalar('lr', self.lrA))
            summaries.append(tf.summary.scalar('entropy', self.entropy))
            summaries.append(tf.summary.scalar('approxkl', self.approxkl))
            summaries.append(tf.summary.scalar('cumulative_reward', self.tot_reward))
            if 'obj' in env.observation_space:
                 summaries.append(tf.summary.image('objvec', tf.cast(255*tf.reshape(self.Mobj, [1, NUM_CHUNKS*BATCH_SIZE, NUM_OBJS, 1]), tf.uint8)))
            summaries.append(tf.summary.scalar('loss', self.loss))
            print(tf.trainable_variables())
            '''
            train_vars = tf.trainable_variables(scope = name + '/targetA')
            print(train_vars)
            for var in train_vars:
                summaries += variable_summaries(var)
            train_vars = tf.trainable_variables(scope = name + '/targetC')
            print(train_vars)
            for var in train_vars:
                summaries += variable_summaries(var)
            '''
            self.merge = tf.summary.merge(summaries)
            self.writer = tf.summary.FileWriter('./log/' + env.name + f'_Sub_{self.tag}/', self.sess.graph)

            # Loader & Saver
            train_vars = []
            train_vars += tf.global_variables(scope = name + '/targetA')
            train_vars += tf.global_variables(scope = name + '/targetC')
            self.loader = tf.train.Saver(train_vars)
            print(train_vars)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            self.sess.run(tf.global_variables_initializer())

            self.copy_op = []
            self.revert_op = []
            main_varsA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/mainA')
            target_varsA = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/targetA')

            for main_var, target_var in zip(main_varsA, target_varsA):
                #print(np.sum(self.sess.run(target_var)))
                self.copy_op.append(target_var.assign(main_var.value()))
                self.revert_op.append(main_var.assign(target_var.value()))

            main_varsC = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/mainC')
            target_varsC = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/targetC')

            for main_var, target_var in zip(main_varsC, target_varsC):
                #print(np.sum(self.sess.run(target_var)))
                self.copy_op.append(target_var.assign(main_var.value()))
                self.revert_op.append(main_var.assign(target_var.value()))

        self.updateNetwork()

    def get_action(self, data):
        # Reshape, Transpose & Collect obs as dict
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
        if RNN:
            state = data['state']
            state = np.reshape(state, [-1, STATE_LENGTH])
            #print(np.var(state, axis = 1))
            dict_all.update({self.Istate: state})

        # Policy Action Inference
        myu, sigma = self.sess.run(self.inferA, feed_dict = dict_all)
        action = np.random.normal(myu, sigma)
        if RNN:
            state = self.sess.run(self.inferS, feed_dict = dict_all)
            return myu, sigma, action, state
        return myu, sigma, action

    def make_batch(self, data, ent_coef, add_merge = False, num_chunks = None, update_gpu = False, lr = None):
        # Obs Dict to variables
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
        if RNN:
            state0 = data['state0']
            state1 = data['state1']

        

        # Reshape & Transopse & memory_op_small
        dict_mem = {}
        if 'image' in self.env.observation_space:
            img0 = np.transpose(np.reshape(img0, [TIME_STEP, NUM_AGENTS, IMG_C, IMG_H, IMG_W]), [1, 0, 3, 4, 2])
            img1 = np.transpose(np.reshape(img1, [TIME_STEP, NUM_AGENTS, IMG_C, IMG_H, IMG_W]), [1, 0, 3, 4, 2])
        if 'audio' in self.env.observation_space:
            wav0 = np.transpose(np.reshape(wav0, [TIME_STEP, NUM_AGENTS, WAV_C * WAV_LENGTH]), [1, 0, 2])
            wav1 = np.transpose(np.reshape(wav1, [TIME_STEP, NUM_AGENTS, WAV_C * WAV_LENGTH]), [1, 0, 2])
        if 'touch' in self.env.observation_space:
            touch0 = np.transpose(np.reshape(touch0, [TIME_STEP, NUM_AGENTS, TACTILE_LENGTH]), [1, 0, 2])
            touch1 = np.transpose(np.reshape(touch1, [TIME_STEP, NUM_AGENTS, TACTILE_LENGTH]), [1, 0, 2])
        if RNN:
            state0 = np.reshape(state0, [TIME_STEP, NUM_AGENTS, STATE_LENGTH])
            #for i in range(TIME_STEP):
                #print(np.var(state0[i], axis = 1))
            state0 = np.transpose(np.reshape(state0, [TIME_STEP, NUM_AGENTS, STATE_LENGTH]), [1, 0, 2])
            state1 = np.transpose(np.reshape(state1, [TIME_STEP, NUM_AGENTS, STATE_LENGTH]), [1, 0, 2])
        if 'image' in self.env.observation_space:
            img0 = np.reshape(img0, [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, IMG_C])
            img1 = np.reshape(img1, [NUM_AGENTS*TIME_STEP, IMG_H, IMG_W, IMG_C])
            dict_mem.update({self._img0: img0})
        if 'audio' in self.env.observation_space:
            wav0 = np.reshape(wav0, [NUM_AGENTS*TIME_STEP, WAV_C * WAV_LENGTH])
            wav1 = np.reshape(wav1, [NUM_AGENTS*TIME_STEP, WAV_C * WAV_LENGTH])
            dict_mem.update({self._wav0: wav0})
        if 'obj' in self.env.observation_space:
            dict_mem.update({self._obj: obj})
        if 'touch' in self.env.observation_space:
            touch0 = np.reshape(touch0, [NUM_AGENTS*TIME_STEP, TACTILE_LENGTH])
            touch1 = np.reshape(touch1, [NUM_AGENTS*TIME_STEP, TACTILE_LENGTH])
            dict_mem.update({self._touch: touch0})
        if RNN:
            state0 = np.reshape(state0, [NUM_AGENTS*TIME_STEP, STATE_LENGTH])
            state1 = np.reshape(state1, [NUM_AGENTS*TIME_STEP, STATE_LENGTH])
            dict_mem.update({self._state: state0})
        self.sess.run(self.memory_op_small, feed_dict = dict_mem)
        
        # Value inference & memory_op_small 
        T = TIME_STEP
        dict_infer = {}
        if 'image' in self.env.observation_space:
            lastimg = [img1[(i+1)*T-1] for i in range(NUM_AGENTS)]
            dict_infer.update({self.Iimg0:lastimg})
        if 'audio' in self.env.observation_space:
            lastwav = [wav1[(i+1)*T-1] for i in range(NUM_AGENTS)]
            dict_infer.update({self.Iwav0:lastwav})
        if 'obj' in self.env.observation_space:
            lastobj = [obj[(i+1)*T-1] for i in range(NUM_AGENTS)] # We assume that obj doesn't change at last episode
            dict_infer.update({self.Iobj:lastobj})
        if 'touch' in self.env.observation_space:
            lasttouch = [touch1[(i+1)*T-1] for i in range(NUM_AGENTS)]
            dict_infer.update({self.Itouch:lasttouch})
        if RNN:
            laststate = [state1[(i+1)*T-1] for i in range(NUM_AGENTS)]
            dict_infer.update({self.Istate:laststate})
        V = self.sess.run(self.inferV0, feed_dict = dict_infer)
        
        # Policy Computation
        myu, sigma, V0 = [], [], []
        chunk_size = (NUM_AGENTS * TIME_STEP) // num_chunks
        for i in range(num_chunks):
            self.sess.run(self.data_op[i])
            Ai, V0i = self.sess.run([self.oldA, self.V0])
            myui, sigmai = Ai
            myu.append(myui)
            sigma.append(sigmai)
            V0.append(V0i)
        
        helper_reward, raw_reward = data['helper_reward'], data['raw_reward']
        helper_reward = np.transpose(np.reshape(helper_reward, [TIME_STEP, NUM_AGENTS]), [1, 0])
        raw_reward = np.transpose(np.reshape(raw_reward, [TIME_STEP, NUM_AGENTS]), [1, 0])
        reward = helper_reward + raw_reward
        cum_reward = np.zeros_like(reward)

        done = data['done']
        done = np.transpose(np.reshape(done, [TIME_STEP, NUM_AGENTS]), [1, 0])
        done = np.reshape(done, [NUM_AGENTS*TIME_STEP, 1])
        done = np.reshape(done, [NUM_AGENTS, TIME_STEP])

        for i in range(NUM_AGENTS):
            done[i][-1] = reward[i][-1]
            for t in reversed(range(T-1)):
                if done[i][t]:
                    cum_reward[i][t] = reward[i][t]
                else:
                    cum_reward[i][t] = GAMMA * cum_reward[i][t+1] + reward[i][t]

        V0 = np.reshape(V0, [NUM_AGENTS, TIME_STEP])
        advs = np.zeros_like(V0)

        reward /= rewardStd

        for i in range(NUM_AGENTS):
            if done[i][-1]:
                advs[i][-1] = reward[i][-1] - V0[i][-1]
            else:
                advs[i][-1] = reward[i][-1] + GAMMA * V[i] - V0[i][-1]
            for t in reversed(range(T - 1)):
                if done[i][t]:
                    advs[i][t] = reward[i][t] - V0[i][t]
                else:
                    delta = reward[i][t] + GAMMA * V0[i][t+1] - V0[i][t]
                    advs[i][t] = delta + GAMMA * LAMBDA * advs[i][t+1]

        Vtarget = advs + V0

        #Before gpu update
        ACTION_LENGTH = self.env.action_space
        myu = np.reshape(np.array(myu), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
        sigma = np.reshape(np.array(sigma), [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
        V0 = np.reshape(np.array(V0), [NUM_AGENTS*TIME_STEP, 1])
        for i in range(TIME_STEP):
            self.rewardTrack.update(cum_reward[:, i])
        for i in range(TIME_STEP):
            self.raw_rewardTrack.update(cum_reward[:, i])
        rewardStd = self.rewardTrack.get_std()
        tot_reward = self.raw_rewardTrack.X0.mean()
        advs = np.reshape(advs, [NUM_AGENTS * TIME_STEP, 1])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        Vtarget = np.reshape(Vtarget, [NUM_AGENTS * TIME_STEP, 1])
        V0 = np.reshape(V0, [NUM_AGENTS * TIME_STEP, 1])
        reward = np.reshape(reward, [NUM_AGENTS * TIME_STEP, 1])
        helper_reward = np.reshape(helper_reward, [NUM_AGENTS * TIME_STEP, 1])
        raw_reward = np.reshape(raw_reward, [NUM_AGENTS * TIME_STEP, 1])
        actionReal = data['action']
        actionReal = np.transpose(np.reshape(actionReal, [TIME_STEP, NUM_AGENTS, ACTION_LENGTH]), [1, 0, 2])
        actionReal = np.reshape(actionReal, [NUM_AGENTS*TIME_STEP, ACTION_LENGTH])
        actionReal = np.reshape(actionReal, [NUM_AGENTS * TIME_STEP, ACTION_LENGTH])

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
            if RNN:
                dict_all.update({self._state:state0})
            dict_all.update({self._myu:myu, self._sigma:sigma})
            dict_all.update({self._advs: advs, self._Vtarget: Vtarget, self._oldV0: V0, self._actionReal: actionReal})
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

            TgradA = self.sess.run(self.gradA)
            print("Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f} lr {:.5f}".format(Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, lr))

            return Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, dict_all 
            
        #data = [advs, img0, obj, myu, sigma, Vtarget, V0, actionReal]
        #return data

    def debug_merge(self, dict_all, global_step):
        summary = self.sess.run(self.merge, feed_dict = dict_all)
        self.writer.add_summary(summary, global_step)

    def trainA(self, lr, num_chunks = None, ent_coef = None):
        self.sess.run(self.init_accA)
        approxkl = 0.
        for i in range(num_chunks):
            self.sess.run(self.data_op[i])
            _, Tapproxkl = self.sess.run([self.accA, self.approxkl], feed_dict = {self.ent_coef:ent_coef})
            approxkl += Tapproxkl / num_chunks
            if (i+1) % NUM_UPDATE == 0:
                self.sess.run(self.optA, feed_dict = {self.lrA:lr, self.ent_coef:ent_coef})
                self.sess.run(self.init_accA)
        return approxkl

    def updateNetwork(self):
        print('updating networks A and C...')
        self.sess.run(self.copy_op)
        print('updated!')

    def revertNetwork(self):
        print('reverting network A and C...')
        self.sess.run(self.revert_op)
        print('reverted!')
