import tensorflow as tf
import numpy as np
from .agents import CriticPPO as Critic
from .agents import AgentContinuousPPO as Agent
from .utils import ParamBackup, rewardTracker, makeOptimizer

class Model():
    def __init__(self, env, sess, encoder, encoder0, obs_sample, gamma, lambda_coeff, timestep, num_batches, use_rnn,tag): 
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.sess = sess
        self.timestep = timestep
        self.gamma = gamma 
        self.lambda_coeff = lambda_coeff
        self.use_rnn = use_rnn

        name = env.task
        BATCH_SIZE = (self.timestep * self.num_agents) // num_batches
        
        obshape_cur = {k:v.shape for k,v in obs_sample.items() if "cur/" in k}
        obshape_prev = {k:v.shape for k,v in obs_sample.items() if "prev/" in k}
        outshape = {
            "myu": (self.action_space,), "sigma": (self.action_space,), "oldV0": (1,),
            "advs": (1,), "Vtarget": (1,),
            "actionReal": (self.action_space,),
        }
        outshape = {k: (self.timestep, self.num_agents) + v for k, v in outshape.items()} 
        tensor_shapes = {**obshape_prev, **outshape}
        print({k:v.shape for k,v in obs_sample.items() })
        print(obshape_cur)
        print(obshape_prev)
        print(tensor_shapes)

        
        with tf.variable_scope(name):
            self.placeholders_inf = {k : tf.placeholder(tf.float32, shape = (1,) + v[1:]) for k,v in obshape_cur.items()} # Cur

            self.placeholders = {k : tf.placeholder(tf.float32, shape = v) for k,v in tensor_shapes.items()} # Prev

            self.memory = {k : tf.get_variable("mem/" + k,shape = (num_batches, BATCH_SIZE) + v[1:],dtype = tf.float32,  trainable = False) for k,v in tensor_shapes.items()}
           
            self.batch = {k : tf.get_variable("real/" + k,shape = (BATCH_SIZE,) + v[1:],dtype = tf.float32,  trainable = False) for k,v in tensor_shapes.items()}

            # Define Memory & Data Operation
            self.store_obs_op = [
                tf.assign(self.memory[key], tf.reshape(self.placeholders[key], self.memory[key].shape), validate_shape = True)
                for key in obshape_prev.keys()
            ]
            self.store_op = [
                tf.assign(self.memory[key], tf.reshape(self.placeholders[key], self.memory[key].shape), validate_shape = True)
                for key in tensor_shapes.keys()
            ]
            self.load_op = [
                [
                    tf.assign(self.batch[key], self.memory[key][i], validate_shape = True) 
                    for key in tensor_shapes.keys()
                ]
                for i in range(num_batches)
            ]
            
            # Model Init 
            self.brain = Agent('mainA', encoder, self.action_space)
            self.critic = Critic('mainC', encoder)
            self.brain0 = Agent('targetA', encoder0, self.action_space)
            self.critic0 = Critic('targetC', encoder0)

            # Forward Propagation 
            self.inferA = self.brain0.forward(self.placeholders_inf)
            self.inferV0 = self.critic0.forward(self.placeholders_inf)
            if self.use_rnn: self.inferS = encoder0.forward(self.placeholders_inf)

            self.dat = {k:self.batch[k] for k in obshape_prev.keys()}
            self.oldA_myu, self.oldA_sigma = self.brain0.forward(self.dat)
            self.A_myu, self.A_sigma = self.brain.forward(self.dat)
            self.V0 = self.critic0.forward(self.dat)
            self.rewardTrack = rewardTracker(0) # Track variance of reward 
            self.raw_rewardTrack = rewardTracker(0.95)

            # Loss definition
            self.ent_coef = tf.placeholder(tf.float32)
            self.loss_Agent, self.clipfrac, self.entropy, self.approxkl, self.pg_loss, self.ratio, self.obs = self.brain.get_loss(
                self.dat, self.batch["myu"], self.batch["sigma"], self.batch["actionReal"], self.batch["advs"], self.ent_coef)
            self.loss_Critic = self.critic.get_loss(self.dat, self.batch["Vtarget"], self.batch["oldV0"])
            self.loss = self.loss_Agent + 0.5 * self.loss_Critic

            self.lrA = tf.placeholder(tf.float32)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.backward_op, self.zero_grad, self.optimizer_step_op, self.grad_norm = makeOptimizer(self.lrA, self.loss, num_batches, decay = False)
            
            self.merge, self.writer = self._register_summaries(name,tag)

            self.backup = ParamBackup(self.sess, [name + '/mainA', name + '/mainC' ], [name + '/targetA', name + '/targetC'])


    def get_action(self, data):
        dict_inf = {self.placeholders_inf[k]:np.expand_dims(data["/".join(k.split("/")[1:])], axis = 0) for k in self.placeholders_inf.keys()}
        myu, sigma = self.sess.run(self.inferA, feed_dict = dict_inf)
        action = np.random.normal(myu, sigma)
        if not self.use_rnn:
            return myu, sigma, action
        else:
            state = self.sess.run(self.inferS, feed_dict = dict_inf)
            return myu, sigma, action, state

    def feed_batch(self, data):
        
        obs_infer = {self.placeholders_inf[key]:np.expand_dims(data[key][-1], axis = 0)  for key in self.placeholders_inf.keys()}
        V = self.sess.run(self.inferV0, feed_dict = obs_infer)
        
        obs_mem = {self.placeholders[key]:data[key] for key in data.keys() if key in self.placeholders.keys() }
        self.sess.run(self.store_obs_op, feed_dict = obs_mem)
        
        myu, sigma, V0 = self._batch_process(self.sess, self.load_op, ops = [self.oldA_myu, self.oldA_sigma,self.V0], feed_dict= None )

        NB, BS, A = myu.shape; myu = np.reshape(myu, [NB*BS, self.num_agents, A])
        NB, BS, A = sigma.shape; sigma = np.reshape(sigma, [NB*BS, self.num_agents, A])
        NB, BS, A = V0.shape; V0 = np.reshape(V0, [NB*BS, A])
    
        helper_reward, raw_reward, done = data['helper_reward'], data['raw_reward'], data['done']
        reward = helper_reward + raw_reward # TS, NA

        reward, rewardStd, tot_reward = self._reward(done, reward, self.gamma, self.rewardTrack, self.raw_rewardTrack, self.timestep, self.num_agents)
        advs = self._advantage( done, reward, V0, V, self.gamma, self.lambda_coeff, self.timestep, self.num_agents)
        Vtarget = advs + V0 # TS, NA

        dict_all = {self.placeholders[key]:data[key] for key in data.keys() if key in self.placeholders.keys()}
        dict_all.update({
            self.placeholders["myu"]:myu, self.placeholders["sigma"]:sigma, self.placeholders["oldV0"]: np.expand_dims(V0,axis = 1), 
            self.placeholders["advs"]: np.expand_dims((advs - advs.mean()) / (advs.std() + 1e-8) ,axis = 1), 
            self.placeholders["Vtarget"]: np.expand_dims(Vtarget ,axis = 1), 
            self.placeholders["actionReal"]: data['action']})
        self.sess.run(self.store_op, feed_dict = dict_all)
        return {
                self.reward: reward, self.helper_reward:helper_reward, self.raw_reward: raw_reward, 
                self.rewardStd: rewardStd, self.tot_reward: tot_reward, self.lrA:1e-4, self.ent_coef:0.01
            }

    def forward(self, ent_coef):
        self.sess.run(self.zero_grad)
        TlA, TlC, Tpg_loss, Tloss, Tratio, _ = self._batch_process(self.sess, self.load_op, 
            ops = [self.loss_Agent, self.loss_Critic, self.pg_loss, self.loss, self.ratio, self.backward_op],
            feed_dict= {self.ent_coef:ent_coef}) # N, B, ACT / N, B, 1
        TgradA = self.sess.run(self.grad_norm)
        return Tloss.mean(), TlA.mean(), TlC.mean(), Tratio.max(), Tpg_loss.mean(), TgradA

    def optimize_step(self, lr, ent_coef):
        approx_kls = []
        for load_op_e in self.load_op:
            self.sess.run(self.zero_grad)
            self.sess.run(load_op_e)
            _, approx_kl = self.sess.run([self.backward_op, self.approxkl], feed_dict = {self.ent_coef:ent_coef})
            approx_kls.append(approx_kl)
            self.sess.run(self.optimizer_step_op, feed_dict = {self.lrA:lr, self.ent_coef:ent_coef})
        return  sum(approx_kls) / len(approx_kls)

    def _advantage(self, done, reward, V0, V, gamma, lambda_coeff, timestep, num_agents):
        assert reward.shape == (timestep, num_agents)
        assert V0.shape == (timestep, num_agents)
        assert done.shape == (timestep, num_agents)
        advs = np.zeros_like(V0) # N, CHK
        for i in range(num_agents):
            if done[-1,i]:
                advs[-1, i] = reward[-1, i] - V0[-1, i]
            else:
                advs[-1, i] = reward[-1, i] + gamma * V[i] - V0[-1, i]
            for t in reversed(range(timestep - 1)):
                if done[t, i]:
                    advs[t,i] = reward[t,i] - V0[t,i]
                else:
                    delta = reward[t,i] + gamma * V0[t+1,i] - V0[t,i]
                    advs[t,i] = delta + gamma * lambda_coeff * advs[t+1,i]
        return advs

    def _reward(self, done, reward, gamma, rewardTrack,raw_rewardTrack, timestep, num_agents ):
        assert reward.shape == (timestep, num_agents)
        assert done.shape == (timestep, num_agents)
        cum_reward = np.zeros_like(reward) # TS, NA 
        for i in range(num_agents):
            done[-1, i] = reward[-1, i] # THIS IS WEIRD
            for t in reversed(range(timestep-1)):
                if done[t,i]:
                    cum_reward[t,i] = reward[t,i]
                else:
                    cum_reward[t,i] = gamma * cum_reward[t+1, i] + reward[t,i]
        for t in range(timestep):
            rewardTrack.update(cum_reward[t, :])
            raw_rewardTrack.update(cum_reward[t, :])
        rewardStd = rewardTrack.get_std()
        tot_reward = raw_rewardTrack.X0.mean()
        reward /= rewardStd
        return reward, rewardStd, tot_reward

    def _batch_process(self,sess, load_ops, ops, feed_dict):
        collate = []
        for load_op in load_ops:
            sess.run(load_op)
            collate.append(sess.run(ops, feed_dict = feed_dict)) # B, ACT / B, 1
        return [np.stack(x) for x in zip(*collate)] # N, B, ACT / N, B, 1

    def _register_summaries(self, name, tag):
        # tf.Summary & Writer
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.raw_reward = tf.placeholder(tf.float32, [None, 1])
        self.rewardStd = tf.placeholder(tf.float32)
        self.helper_reward = tf.placeholder(tf.float32, [None, 1])
        self.tot_reward = tf.placeholder(tf.float32)
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
                'entropy': self.entropy,
                'approxkl': self.approxkl,
                'cumulative_reward': self.tot_reward,
                'loss': self.loss,
            }
        summaries = [tf.summary.scalar(k, v) for k,v in summaries.items()]
        print(tf.trainable_variables())
        merge = tf.summary.merge(summaries)
        writer = tf.summary.FileWriter('./log/' + f'submodel_{name}_{tag}/', self.sess.graph)
        return merge, writer
