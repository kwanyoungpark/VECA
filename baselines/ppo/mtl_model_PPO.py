import tensorflow as tf
from .utils import *
import numpy as np
from .constants import *
from .sub_model_PPO import Model as SubModel
if False:
    from agents import UniversalRNNEncoder as Encoder
else:
    from .agents import UniversalEncoder as Encoder
from .util2 import ParamBackup


class Model():
    def __init__(self, envs,sess, obs_sample, tag): 
        self.sess = sess

        # Multi-task Model Initialization
        self.encoder = Encoder('mainE', SIM)
        self.encoder0 = Encoder('targetE', SIM)
        self.models = [] 
        for env in envs:
            model = SubModel(env, self.sess, gamma = GAMMA, lambda_coeff = LAMBDA, timestep = TIME_STEP, num_batches = NUM_CHUNKS,
                encoder = self.encoder, encoder0 = self.encoder0, obs_sample = obs_sample, tag = tag)
            self.models.append(model)

        # Model Loader and Saver Init
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.loader = tf.train.Saver(tf.trainable_variables())

        # Log Writer Init
        self.summaries = {'lr': tf.placeholder(tf.float32),'ent_coeff': tf.placeholder(tf.float32),}
        summaries = [variable_summaries(var) for var in tf.trainable_variables(scope = 'targetE')]
        summaries.append([tf.summary.scalar(k, v) for k,v in self.summaries.items()])
        self.merge = tf.summary.merge(summaries)
        self.writer = tf.summary.FileWriter('./log/' + f'MultiTask_{tag}', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # Trainable Variables Early Stopping & Parameter Revert 
        self.backup = ParamBackup(self.sess, ['mainE'], ['targetE'], backup_subsets = [submodel.backup for submodel in self.models])
    def get_action(self, data):
        return [model.get_action(elem)[2] for elem, model in zip(data,self.models)]

    def make_batch(self, batches):
        summarys = []
        for batch, model in zip(batches,self.models):
            summarys.append(model.make_batch(batch))
        return summarys

    def forward(self, ent_coef):
        collate = [model.forward(ent_coef) for model in self.models]
        loss, lA, lC, ratio, pg_loss, grad = zip(*collate)
        loss, lA, lC, ratio, pg_loss, grad= sum(loss), sum(lA), sum(lC), max(ratio), sum(pg_loss), sum(grad)

        return loss, lA, lC, ratio, pg_loss, grad


    def optimize_step(self, lr, ent_coef):
        approx_kls = [model.optimize_step(lr, ent_coef) for model in self.models]
        return max(approx_kls)

    # Tensorboard Logging
    def log(self, summarys, lr, ent_coeff, global_step):
        summary = self.sess.run(self.merge, feed_dict = {self.summaries["lr"]:lr, self.summaries["ent_coeff"]:ent_coeff})
        self.writer.add_summary(summary, global_step)
        for summary_dict, model in zip(summarys,self.models):
            summary = self.sess.run(model.merge, feed_dict = summary_dict)
            model.writer.add_summary(summary, global_step)

    # Saver & Loader
    def save(self, name, global_step):
        self.saver.save(self.sess, name, global_step = global_step)

    def load(self, model_name, global_step = None):
        print("LOAD")
        ckp = tf.train.latest_checkpoint(model_name)
        self.loader.restore(self.sess, model_name)
