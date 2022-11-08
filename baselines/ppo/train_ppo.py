import tensorflow as tf
import numpy as np
import os
from constants import *
from utils import *
from mtl_model_PPO import Model as MTLModel
import time

tag = "Moveobj_112233_level1"
PORT = 10000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tasks.COGNIANav.COGNIADemoEnv import Environment
from tasks.COGNIANav.headquarter import HeadQuarter


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session()

envs = [Environment(task = TASK, num_envs = NUM_ENVS, port = PORT+ i) for i, TASK in enumerate(TASKS)]

model = MTLModel(envs, sess,obs_sample, tag)
heads = [HeadQuarter(env, bufferlength = BUFFER_LENGTH, timestep = TIME_STEP) for env in envs]

TRAIN_STEP = 2000000
SAVE_STEP = 80000 / NUM_ENVS
REC_STEP = 100000
NUM_CHUNKS = 4
NUM_UPDATE = 1
TRAIN_LOOP = 4
TIME_STEP = 128
BUFFER_LENGTH = 2500

SCHEDULE = True

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


if __name__ == "__main__":
    lr_scheduler = AdaptiveLR(schedule = SCHEDULE)
    frac = 1.
    entropy_coeff = 0.01 * frac
    

    for step in range(TRAIN_STEP):

        [head.step() for head in heads]

        if (step+1) % TIME_STEP == 0:
            print("Training Actor & Critic...")

            summarys = model.make_batch(heads)
            loss, _, _, _, _, _ = model.forward(ent_coef = entropy_coeff)

            for idx in range(TRAIN_LOOP):
                approxkl = model.optimize_step(lr = lr_scheduler.lrA / (TRAIN_LOOP), ent_coef = entropy_coeff)
                if approxkl > 0.01: break

            if approxkl <= 0.01:
                print("KLD {:.3f}, updated full gradient step.".format(approxkl))
            else:
                print("KLD {:.3f}, early stopping.".format(approxkl))
            
            lossP, _, _, _, _, _ = model.forward(ent_coef = entropy_coeff)

            lr_scheduler.step(model.backup, approxkl, loss, lossP)

            model.log(summarys, lr_scheduler.lrA, entropy_coeff, step )

            [head.replayBuffer.clear() for head in heads]
        if (step+1) % 1280 == 0:
            [head.restart() for head in heads]
        if (step + 1) % SAVE_STEP == 0:
            model.save(name = f'./model/{METHOD}_mtl_model_{tag}/')


