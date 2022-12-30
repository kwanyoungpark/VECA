import tensorflow as tf
from utils import *
import numpy as np
from constants import *
from sub_model_SAC import Model as SubModel
from agents import UniversalEncoder as Encoder

class Model():
    def __init__(self, envs): 
        self.envs = envs
        self.heads = []
        self.models = [] 

        self.encoderC1 = Encoder('mainEC1', SIM)
        self.encoderC10 = Encoder('targetEC1', SIM)
        self.encoderC2 = Encoder('mainEC2', SIM)
        self.encoderC20 = Encoder('targetEC2', SIM)
        self.encoderA = Encoder('mainEA', SIM)
        self.encoderA0 = Encoder('targetEA', SIM)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()

        for env in envs:
            if env.SIM == 'VECA':
                if env.name == 'Navigation':
                    from tasks.navigation.headquarter import HeadQuarter
                elif env.name == 'KickTheBall':
                    from tasks.kicktheball.headquarter import HeadQuarter
                elif env.name == 'MANavigation':
                    from tasks.MANavigation.headquarter import HeadQuarter
                elif env.name == 'Transport':
                    from tasks.transport.headquarter import HeadQuarter
                elif env.name == 'RunBaby':
                    from tasks.runBaby.headquarter import HeadQuarter
                elif env.name == 'COGNIANav':
                    from tasks.COGNIANav.headquarter import HeadQuarter
            elif env.SIM == 'ATARI':
                from tasks.atari.headquarter import HeadQuarter
            elif env.SIM == 'CartPole':
                from tasks.cartpole.headquarter import HeadQuarter
            elif env.SIM == 'Pendulum':
                from tasks.pendulum.headquarter import HeadQuarter
            model = SubModel(env, self.sess, self.encoderC1, self.encoderC10, self.encoderC2, self.encoderC20, self.encoderA, self.encoderA0)
            self.models.append(model)
            self.heads.append(HeadQuarter(env, model))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        summaries = []
        train_vars = tf.trainable_variables(scope = 'targetEC')
        print(train_vars)
        for var in train_vars:
            summaries.append(variable_summaries(var))

        train_vars = []
        train_vars = tf.global_variables(scope = 'targetEC')
        print(train_vars)
        
        train_vars = tf.trainable_variables()
        self.loader = tf.train.Saver(train_vars)
        print(train_vars)
 
        self.merge = tf.summary.merge(summaries)
        self.writer = tf.summary.FileWriter('./log/', self.sess.graph)
        self.global_step = 0
        self.sess.run(tf.global_variables_initializer())

        self.hard_copy_op = []
        self.soft_copy_op = []

        self.isSame = []
        main_varsEC = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mainEC')
        target_varsEC = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetEC')

        main_varsEA = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mainEA')
        target_varsEA = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetEA')

        print('C')
        for main_var, target_var in zip(main_varsEC, target_varsEC):
            print(main_var.name, target_var.name)
            #print(np.sum(self.sess.run(target_var)))
            #self.isSame.append(tf.reduce_sum(tf.abs(main_var - target_var)))
            self.hard_copy_op.append(target_var.assign(main_var.value()))
            self.soft_copy_op.append(target_var.assign(target_var.value() * 0.99 + main_var.value() * 0.01))

        print('A')
        for main_var, target_var in zip(main_varsEA, target_varsEA):
            print(main_var.name, target_var.name)
            #print(np.sum(self.sess.run(target_var)))
            self.isSame.append(tf.reduce_sum(tf.abs(main_var - target_var)))
            self.hard_copy_op.append(target_var.assign(main_var.value()))
            self.soft_copy_op.append(target_var.assign(main_var.value()))
            #self.soft_copy_op.append(target_var.assign(target_var.value() * 0.99 + main_var.value() * 0.01))

        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.hard_copy_op)
        for model in self.models:
            self.sess.run(model.hard_copy_op)
        self.updateNetwork()

    def get_action(self, data):
        action = []
        for i, model in enumerate(self.models):
            action.append(model.get_action(data[i]))
        return action

    def make_batch(self, ent_coef, add_merge = False, num_chunks = None, update_gpu = False, lr = None):
        if add_merge:
            TlA, TlC, Tpg_loss, Tratio, Tloss, TgradA, Tdict_all = 0., 0., 0., 0., 0., 0., []
            for i, model in enumerate(self.models):
                if METHOD == 'PPO':
                    batch = self.heads[i].replayBuffer.get_batch(-1)
                else:
                    batch = self.heads[i].replayBuffer.get_batch(TIME_STEP)
                loss, lA, lC, ratio, pg_loss, grad, dict_all = model.make_batch(batch, ent_coef, add_merge, num_chunks, update_gpu, lr)
                TlA += lA 
                TlC += lC 
                Tpg_loss += pg_loss 
                Tloss += loss
                TgradA += grad
                Tratio = max(np.max(ratio), Tratio)
                Tdict_all.append(dict_all)
            print("Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f} lr {:.5f}".format(Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, lr))

            return Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, Tdict_all 
        else:
            for i, model in enumerate(self.models):
                if METHOD == 'PPO':
                    batch = self.heads[i].replayBuffer.get_batch(-1)
                if METHOD == 'SAC':
                    batch = self.heads[i].replayBuffer.get_batch(TIME_STEP)
                model.make_batch(batch, ent_coef, add_merge, num_chunks, update_gpu, lr)
            
        #data = [advs, img0, obj, myu, sigma, Vtarget, V0, actionReal]
        #return data

    def debug_merge(self, dict_all):
        summary = self.sess.run(self.merge)
        self.writer.add_summary(summary, self.global_step)
        for i, model in enumerate(self.models):
            model.debug_merge(dict_all[i])

    def trainA(self, lr, num_chunks = None, ent_coef = None):
        Tapproxkl = 0.
        for model in self.models:
            approxkl = model.trainA(lr, num_chunks, ent_coef)
            Tapproxkl = max(Tapproxkl, approxkl)
        return Tapproxkl

    def updateNetwork(self):
        print('updating networks A and C...')
        #print(self.sess.run(self.isSame))
        self.sess.run(self.soft_copy_op)
        for model in self.models:
            model.updateNetwork()
        #print(self.sess.run(self.isSame))
        print('updated!')

    def restart(self):
        for head in self.heads:
            head.restart()

    def step(self):
        self.global_step += NUM_ENVS 
        for model in self.models:
            model.global_step += NUM_ENVS
        if SIM == 'VECA':
            #start_time = time.time()
            for head in self.heads:
                head.send_action()
            #print(time.time() - start_time)
            #start_time = time.time()
            for head in self.heads:
                head.collect_observations()
            #print(time.time() - start_time)
        else:
            for head in self.heads:
                head.step()

    def clear(self):
        for head in self.heads:
            head.replayBuffer.clear()

    def save(self):
        name = './model/' + METHOD
        for env in self.envs:
            name += '-' + env.name
        self.saver.save(self.sess, name, global_step = self.global_step)

    def load(self, model_name, global_step = None):
        self.loader.restore(self.sess, model_name)
        #self.saver.restore(self.sess, model_name)
        if global_step is not None:
            self.global_step = global_step
