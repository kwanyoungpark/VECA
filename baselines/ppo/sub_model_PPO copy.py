import tensorflow as tf
from utils import *
import numpy as np
from constants import *
from agents import CriticPPO as Critic
from agents import AgentContinuousPPO as Agent
from munch import Munch

class Model():
    def __init__(self, env, sess, timestep, num_batches, encoder, encoder0, obs_sample, tag,name = None): 
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.tag = tag
        self.sess = sess
        self.timestep = timestep
        self.num_batches = num_batches
        if name is None: name = env.name
        if self.num_batches is None:
            BATCH_SIZE = 256
        else:
            BATCH_SIZE = (self.timestep * self.num_agents) // self.num_batches
        
        obshape = Munch({k:v.shape for k,v in obs_sample.items()})
        outshape = Munch({
            "myu": (self.num_agents, self.action_space),
            "sigma": (self.num_agents, self.action_space),
            "actionReal": (self.num_agents, self.action_space),
            "advs": (self.num_agents, 1),
            "Vtarget": (self.num_agents, 1),
            "oldV0": (self.num_agents, 1),
        })
        tensor_shapes = {**obshape, **outshape}

        
        with tf.variable_scope(name):
            # Plotting placeholders
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.raw_reward = tf.placeholder(tf.float32, [None, 1])
            self.rewardStd = tf.placeholder(tf.float32)
            self.helper_reward = tf.placeholder(tf.float32, [None, 1])
            self.tot_reward = tf.placeholder(tf.float32)
            self.lrA = tf.placeholder(tf.float32)
            self.ent_coef = tf.placeholder(tf.float32)

            self.placeholders_inf = {k : tf.placeholder(tf.float32, shape = v) for k,v in obshape.items()}

            self.placeholders = {k : tf.placeholder(tf.float32, shape = (self.timestep,) + v) for k,v in tensor_shapes.items()}

            self.memory = {k : tf.get_variable("mem/" + k,tf.float32, shape = (self.num_batches, BATCH_SIZE) + v[1:]) for k,v in tensor_shapes.items()}
           
            self.data_real = {k : tf.get_variable("real/" + k,tf.float32, shape = (BATCH_SIZE,) + v[1:]) for k,v in tensor_shapes.items()}

            # Define Memory & Data Operation
            self.store_obs_op = [
                tf.assign(self.memory[key], tf.reshape(self.placeholders[key], self.memory[key].shape), validate_shape = True)
                for key in obshape.keys()
            ]
            self.store_op = [
                tf.assign(self.memory[key], tf.reshape(self.placeholders[key], self.memory[key].shape), validate_shape = True)
                for key in tensor_shapes.keys()
            ]
            self.load_op = [
                [
                    tf.assign(self.data_real[key], self.memory[key][i], validate_shape = True) 
                    for key in tensor_shapes.keys()
                ]
                for i in range(self.num_batches)
            ]
            
            # Model Init 
            self.brain = Agent('mainA', encoder, self.env.action_space)
            self.critic = Critic('mainC', encoder)
            self.brain0 = Agent('targetA', encoder0, self.env.action_space)
            self.critic0 = Critic('targetC', encoder0)

            # Forward Propagation 
            self.dat = {k:self.data_real[k] for k in obshape.keys()}
            self.inferA = self.brain0.forward(self.placeholders_inf)
            self.inferV0 = self.critic0.forward(self.placeholders_inf)
            if RNN: self.inferS = encoder0.forward(self.placeholders_inf)
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
            
            self.merge, self.writer = self._register_summaries()

            self.copy_op, self.revert_op = [], []
            for mainV, targetV in (('/mainA', '/targetA'), ('/mainC','/targetC')):
                main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+mainV)
                target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+targetV)
                for main_var, target_var in zip(main_vars, target_vars):
                    self.copy_op.append(target_var.assign(main_var.value()))
                    self.revert_op.append(main_var.assign(target_var.value()))

        self.sess.run([self.copy_op])

    def get_action(self, data):
        dict_inf = {self.placeholders_inf[k]:data[k] for k in self.placeholders_inf.keys()}
        myu, sigma = self.sess.run(self.inferA, feed_dict = dict_inf)
        action = np.random.normal(myu, sigma)
        if not RNN:
            return myu, sigma, action
        else:
            state = self.sess.run(self.inferS, feed_dict = dict_inf)
            return myu, sigma, action, state

    def make_batch(self, data, ent_coef, add_merge = False, num_batches = None, update_gpu = False, lr = None):
        
        obs_infer = {self.placeholders_inf[k]:data[k][-1]  for k in data.keys() if "cur/" in k }
        V = self.sess.run(self.inferV0, feed_dict = obs_infer)
        
        obs_mem = {self.placeholders[k]:data[k] for k in data.keys() if "prev/" in k }
        self.sess.run(self.store_obs_op, feed_dict = obs_mem)
        
        myu, sigma, V0 = self._batch_collate(self.sess, num_batches, self.load_op, 
            ops = [self.oldA, self.V0], feed_dict= None ) # N, B, ACT / N, B, 1
    
        helper_reward, raw_reward, done = data['helper_reward'], data['raw_reward'], data['done']
        reward = helper_reward + raw_reward # TS, NA

        reward, rewardStd, tot_reward = self._reward(self.timestep, self.num_agents, done, reward,self.rewardTrack, self.raw_rewardTrack)
        advs = self._advantage(self.timestep, self.num_agents, done, reward, V0, V)
        Vtarget = advs + V0 # TS, NA

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actionReal = data['action'] # TS, NA, ACT

        assert update_gpu
        dict_all = {self.placeholders[k]:data[k] for k in self.placeholders.keys() if "prev/" in k}
        dict_all.update({self._myu:myu, self._sigma:sigma, self._advs: advs, 
            self._Vtarget: Vtarget, self._oldV0: V0, self._actionReal: actionReal})
        self.sess.run(self.store_op, feed_dict = dict_all)
        if add_merge:
            dict_all = {
                self.reward: reward, self.helper_reward:helper_reward, self.raw_reward: raw_reward, 
                self.rewardStd: rewardStd, self.tot_reward: tot_reward, 
                self.lrA:lr, self.ent_coef:ent_coef
            }
            self.sess.run(self.init_accA)
            TlA, TlC, Tpg_loss, Tloss, Tratio, _ = self._batch_collate(self.sess, num_batches, self.load_op, 
                ops = [self.loss_Agent, self.loss_Critic, self.pg_loss, self.loss, self.ratio, self.accA],
                feed_dict= {self.ent_coef:ent_coef}) # N, B, ACT / N, B, 1
            TgradA = self.sess.run(self.gradA)
            return Tloss.mean(), TlA.mean(), TlC.mean(), Tratio.max(), Tpg_loss.mean(), TgradA, dict_all 

    def trainA(self, lr, num_batches = None, ent_coef = None):
        assert NUM_UPDATE == 1
        self.sess.run(self.init_accA)
        approxkl = 0.
        for i in range(num_batches):
            self.sess.run(self.load_op[i])
            _, Tapproxkl = self.sess.run([self.accA, self.approxkl], feed_dict = {self.ent_coef:ent_coef})
            approxkl += Tapproxkl / num_batches
            self.sess.run(self.optA, feed_dict = {self.lrA:lr, self.ent_coef:ent_coef})
            self.sess.run(self.init_accA)
        return approxkl

    def _register_summaries(self):
        # tf.Summary & Writer
        summaries = {
                'helper_reward': tf.reduce_mean(self.helper_reward),
                'agent_loss': self.loss_Agent,
                'expected_reward(V)': tf.reduce_mean(self.V0),
                'critic_loss': self.loss_Critic,
                'relative_critic_loss': self.loss_Critic / tf.reduce_mean(self.reward),
                'V0': tf.reduce_mean(self.V0),
                'clipfrac': self.clipfrac,
                'raw_reward': tf.reduce_mean(self.raw_reward),
                'reward': tf.reduce_mean(self.reward),
                'rewardStd': self.rewardStd,
                'lr': self.lrA,
                'entropy': self.entropy,
                'approxkl': self.approxkl,
                'cumulative_reward': self.tot_reward,
                'loss': self.loss,
            }
        summaries = [tf.summary.scalar(k, v) for k,v in summaries]
        print(tf.trainable_variables())
        merge = tf.summary.merge(summaries)
        writer = tf.summary.FileWriter('./log/' + f'_Sub_{self.tag}/', self.sess.graph)
        return merge, writer

    def _advantage(self, timestep, num_agents, done, reward, V0, V):
        advs = np.zeros_like(V0) # N, CHK
        for i in range(num_agents):
            if done[-1,i]:
                advs[-1, i] = reward[-1, i] - V0[-1, i]
            else:
                advs[-1, i] = reward[-1, i] + GAMMA * V[i] - V0[-1, i]
            for t in reversed(range(timestep - 1)):
                if done[t, i]:
                    advs[t,i] = reward[t,i] - V0[t,i]
                else:
                    delta = reward[t,i] + GAMMA * V0[t+1,i] - V0[t,i]
                    advs[t,i] = delta + GAMMA * LAMBDA * advs[t+1,i]
        return advs
    def _reward(self, timestep, num_agents, done, reward, rewardTrack,raw_rewardTrack ):
        cum_reward = np.zeros_like(reward) # TS, NA 
        for i in range(num_agents):
            done[-1, i] = reward[-1, i] # THIS IS WEIRD
            for t in reversed(range(timestep-1)):
                if done[t,i]:
                    cum_reward[t,i] = reward[t,i]
                else:
                    cum_reward[t,i] = GAMMA * cum_reward[t+1, i] + reward[t,i]
        for t in range(timestep):
            rewardTrack.update(cum_reward[t, :])
            raw_rewardTrack.update(cum_reward[t, :])
        rewardStd = rewardTrack.get_std()
        tot_reward = raw_rewardTrack.X0.mean()
        reward /= rewardStd
        return reward, rewardStd, tot_reward
    def _batch_collate(self,sess, num_batches, load_op, ops, feed_dict):
        collate = []
        for i in range(num_batches):
            sess.run(load_op[i])
            collate.append(sess.run(ops, feed_dict = feed_dict))
        return [np.stack(x) for x in zip(*collate)] # N, B, ACT / N, B, 1
