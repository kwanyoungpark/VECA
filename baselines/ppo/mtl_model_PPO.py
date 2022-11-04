import tensorflow as tf
from utils import *
import numpy as np
import sys
import os
import cv2 
import time
from constants import *
from sub_model_PPO import Model as SubModel
if RNN:
    from agents import UniversalRNNEncoder as Encoder
else:
    from agents import UniversalEncoder as Encoder

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as printCheck

class Model():
    def __init__(self, envs,tag,curriculum ): 
        self.envs = envs
        self.heads = []
        self.models = [] 
        self.tag = tag

        self.encoder = Encoder('mainE', SIM)
        self.encoder0 = Encoder('targetE', SIM)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()

        for env in envs:
            if env.SIM == 'VECA':
                if env.name == 'Navigation':
                    from tasks.navigation.headquarter import HeadQuarter
                elif env.name == 'NavigationCogSci':
                    from tasks.navigationCogSci.headquarter import HeadQuarter
                elif env.name == 'KickTheBall':
                    from tasks.kicktheball.headquarter import HeadQuarter
                elif env.name == 'MANavigation':
                    from tasks.MANavigation.headquarter import HeadQuarter
                elif env.name == 'Transport':
                    from tasks.transport.headquarter import HeadQuarter
                elif env.name == 'MazeNav':
                    from tasks.MazeNav.headquarter import HeadQuarter
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
            model = SubModel(env, self.sess, self.encoder, self.encoder0, self.tag)
            self.models.append(model)
            self.heads.append(HeadQuarter(env, model,curriculum))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        summaries = []
        train_vars = tf.trainable_variables(scope = 'targetE')
        print(train_vars)
        for var in train_vars:
            summaries.append(variable_summaries(var))

        train_vars = []
        train_vars += tf.global_variables(scope = 'targetE')
        print(train_vars)

        train_vars = tf.trainable_variables()
        self.loader = tf.train.Saver(train_vars)
        print(train_vars)
 
        self.merge = tf.summary.merge(summaries)
        self.writer = tf.summary.FileWriter('./log/' + f'MTL_{self.tag}', self.sess.graph)
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        self.sess.run(tf.global_variables_initializer())

        self.copy_op = []
        self.revert_op = []
        main_varsE = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mainE')
        target_varsE = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetE')

        for main_var, target_var in zip(main_varsE, target_varsE):
            print(main_var.name, target_var.name)
            #print(np.sum(self.sess.run(target_var)))
            self.copy_op.append(target_var.assign(main_var.value()))
            self.revert_op.append(main_var.assign(target_var.value()))

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
                batch = self.heads[i].replayBuffer.get_batch(-1)
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
                batch = self.heads[i].replayBuffer.get_batch(-1)
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
        self.sess.run(self.copy_op)
        for model in self.models:
            model.updateNetwork()
        print('updated!')

    def revertNetwork(self):
        print('reverting network A and C...')
        self.sess.run(self.revert_op)
        for model in self.models:
            model.revertNetwork()
        print('reverted!')

    def restart(self):
        for head in self.heads:
            head.restart()

    def step(self):
        self.global_step += NUM_ENVS
        for model in self.models:
            model.global_step += NUM_ENVS
        #start_time = time.time()
        if SIM == 'VECA':
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
        name = f'./model/{METHOD}_mtl_model_{self.tag}/'
        #for env in self.envs:
        #    name += '-' + env.name
        self.saver.save(self.sess, name, global_step = self.global_step)

    def load(self, model_name, global_step = None):
        print("DEBUG")
        print(model_name)
        ckp = tf.train.latest_checkpoint(model_name)
        printCheck(ckp, all_tensors=True, tensor_name='')
        self.loader.restore(self.sess, model_name)
        if global_step is not None:
            self.global_step = global_step
            for model in self.models:
                model.global_step = global_step
