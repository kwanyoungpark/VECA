import tensorflow as tf
from utils import *
import numpy as np
import sys
import os
import time
from constants import *
from sub_model_PPO import Model as SubModel
if RNN:
    from agents import UniversalRNNEncoder as Encoder
else:
    from agents import UniversalEncoder as Encoder

class Model():
    def __init__(self, envs,sess, tag): 
        self.envs = envs
        self.models = [] 
        self.tag = tag
        self.sess = sess

        # Multi-task Model Initialization
        self.encoder = Encoder('mainE', SIM)
        self.encoder0 = Encoder('targetE', SIM)
        for env in envs:
            model = SubModel(env, self.sess, self.encoder, self.encoder0, self.tag)
            self.models.append(model)

        # Model Loader and Saver Init
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.loader = tf.train.Saver(tf.trainable_variables())

        # Log Writer Init
        summaries = [variable_summaries(var) for var in tf.trainable_variables(scope = 'targetE')]
        self.merge = tf.summary.merge(summaries)
        self.writer = tf.summary.FileWriter('./log/' + f'MTL_{self.tag}', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # Trainable Variables Early Stopping & Parameter Revert init
        main_varsE = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mainE')
        target_varsE = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetE')
        self.copy_op = [target_var.assign(main_var.value()) for main_var, target_var in zip(main_varsE, target_varsE)]
        self.revert_op = [main_var.assign(target_var.value()) for main_var, target_var in zip(main_varsE, target_varsE)]

        self.updateNetwork()

    def get_action(self, data):
        action = []
        for i, model in enumerate(self.models):
            action.append(model.get_action(data[i]))
        return action

    def make_batch(self, heads, ent_coef, add_merge = False, num_chunks = None, update_gpu = False, lr = None):
        assert add_merge
        TlA, TlC, Tpg_loss, Tratio, Tloss, TgradA, Tdict_all = 0., 0., 0., 0., 0., 0., []
        for i, model in enumerate(self.models):
            batch = heads[i].replayBuffer.get_batch()
            loss, lA, lC, ratio, pg_loss, grad, dict_all = model.make_batch(batch, ent_coef, add_merge, num_chunks, update_gpu, lr)
            TlA += lA; TlC += lC; Tpg_loss += pg_loss; Tloss += loss; TgradA += grad; Tratio = max(np.max(ratio), Tratio)
            Tdict_all.append(dict_all)
        print("Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f} lr {:.5f}".format(Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, lr))

        return Tloss, TlA, TlC, Tratio, Tpg_loss, TgradA, Tdict_all 

    def debug_merge(self, dict_all, global_step):
        summary = self.sess.run(self.merge)
        self.writer.add_summary(summary, global_step)
        for i, model in enumerate(self.models):
            model.debug_merge(dict_all[i], global_step)

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

    def save(self, name, global_step):
        self.saver.save(self.sess, name, global_step = global_step)

    def load(self, model_name, global_step = None):
        print("LOAD")
        ckp = tf.train.latest_checkpoint(model_name)
        self.loader.restore(self.sess, model_name)
