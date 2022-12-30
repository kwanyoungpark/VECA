import tensorflow as tf
import math
import numpy as np
import os
class ParamBackup:
    def __init__(self, sess, src_scopes, dst_scopes, backup_subsets = []):
        assert len(src_scopes) == len(dst_scopes)
        self.copy_op, self.revert_op = [], []
        self.sess = sess
        self.backup_subsets = backup_subsets
        for src, dst in zip(src_scopes, dst_scopes):
            main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=src)
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst)
            self.copy_op.append([target_var.assign(main_var.value()) for main_var, target_var in zip(main_vars, target_vars)])
            self.revert_op.append([main_var.assign(target_var.value()) for main_var, target_var in zip(main_vars, target_vars)])

    def commit(self):
        print("Commited")
        self.sess.run(self.copy_op)
        for backup in self.backup_subsets:
            backup.commit()

    def revert(self):
        print("Reverted")
        self.sess.run(self.revert_op)
        for backup in self.backup_subsets:
            backup.revert()
        
class AdaptiveLR:
    def __init__(self, schedule):
        self.schedule = schedule 
        if self.schedule:
            self.lrAM, self.lrA0, self.lrA = 1e-3, 1e-4, 1e-4
        else:
            self.lrA = 2.5e-4
    def step(self, backup, approxkl, loss, lossP):
        if self.schedule:
            if (approxkl > 0.01) or (loss < lossP):
                self.lrA /= 2
                self.lrA0 *= 0.95
            else:
                self.lrA = min(self.lrA0, self.lrA * 2)
                self.lrA0 *= 1.02
            if (approxkl > 0.015) or (loss < lossP):
                backup.revert()
            else:
                backup.commit()
            self.lrA0 = min(self.lrA0, self.lrAM) 
        else:
            backup.commit()

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

def excludeNone(grad, var):
    gradients, variables = [], []
    for i, grad in enumerate(grad):
        if grad is None:
            continue
        gradients.append(grad)
        variables.append(var[i])
    return gradients, variables

def makeOptimizer(lr, loss, num_batches, decay = False, var_list = None):
    if decay:
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(lr, global_step, 1000, 0.96, staircase = False)
    opt = tf.train.AdamOptimizer(lr, epsilon = 1e-4)
    if var_list == None:
        gradients, variables = zip(*opt.compute_gradients(loss))
    else:
        gradients, variables = zip(*opt.compute_gradients(loss, var_list = var_list))
    gradients, variables = excludeNone(gradients, variables)

    accum_grads = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in variables]
    backward_op= [accum_grads[i].assign_add(gradients / num_batches) for i, gradients in enumerate(gradients)]

    accum_gradsC, _ = tf.clip_by_global_norm(accum_grads, 5.0)
    optimizer_step_op = opt.apply_gradients([(accum_gradsC[i], var) for i, var in enumerate(variables)])
    
    zero_grads = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]
    
    return backward_op, zero_grads, optimizer_step_op, tf.global_norm(accum_grads)

def variable_summaries(var):
    summaries = []
    name = "_"+var.name
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        summaries.append(tf.summary.scalar('mean'+name, mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            summaries.append(tf.summary.scalar('stddev'+name, stddev))
        summaries.append(tf.summary.scalar('max'+name, tf.reduce_max(var)))
        summaries.append(tf.summary.scalar('min'+name, tf.reduce_min(var)))
        summaries.append(tf.summary.histogram('histogram'+name, var))
    return summaries
    
def lr_schedule(lr, step, decay_step, ratio):
    if step < decay_step:
        return lr * step / decay_step
    else:
        return lr * np.power(ratio, (step / decay_step - 1))

class Saver:
    def __init__(self, sess, max_to_keep = 5):
        self.sess = sess
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        self.loader = tf.train.Saver(tf.trainable_variables())
        self.filename = "model"

    def save(self, ckpt_dir, global_step):
        os.makedirs(ckpt_dir, exist_ok = True)
        self.saver.save(self.sess, os.path.join(ckpt_dir, self.filename), global_step = global_step)
        print("Saved model to path:", ckpt_dir)

    def load_if_exists(self, ckpt_dir):
        steps = 0
        if os.path.exists(ckpt_dir):
            ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if ckpt is not None:
                print("Loading model from path:", ckpt_dir)
                self.loader.restore(self.sess, ckpt)
                steps = int(os.path.basename(os.path.splitext(ckpt)[0]).split("-")[-1])
        else:
            print("Given checkpoint directory does not exist")
        return steps