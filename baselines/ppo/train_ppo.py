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

envs = [Environment(task = TASK, num_envs = NUM_ENVS, port = PORT+ i) for i, TASK in enumerate(TASKS)]

model = MTLModel(envs, tag)
heads = [HeadQuarter(env, bufferlength = BUFFER_LENGTH, timestep = TIME_STEP) for env in envs]

TRAIN_STEP = 2000000
SAVE_STEP = 80000 / NUM_ENVS
REC_STEP = 100000

SCHEDULE = True

class AdaptiveLR:
    def __init__(self, schedule):
        self.schedule = schedule 
        if self.schedule:
            self.lrAM, self.lrA0, self.lrA = 1e-3, 1e-4, 1e-4
        else:
            self.lrA = 2.5e-4
    def step(self, model, approxkl, loss, lossP):
        if self.schedule:
            if (approxkl > 0.01) or (loss < lossP):
                self.lrA /= 2
                self.lrA0 *= 0.95
            else:
                self.lrA = min(self.lrA0, self.lrA * 2)
                self.lrA0 *= 1.02
            if (approxkl > 0.015) or (loss < lossP):
                model.revertNetwork()
            else:
                model.updateNetwork()
            self.lrA0 = min(self.lrA0, self.lrAM) 
        else:
            model.updateNetwork()

lr_scheduler = AdaptiveLR(schedule = SCHEDULE)
frac = 1.
for step in range(TRAIN_STEP):
    [head.step() for head in heads]
    if (step+1) % 1280 == 0:
        [head.restart() for head in heads]
    if (step + 1) % SAVE_STEP == 0:
        model.save(name = f'./model/{METHOD}_mtl_model_{tag}/')
    if (step+1) % TIME_STEP == 0:
        print("Training Actor & Critic...")
        #model.make_batch(ent_coef = 0.01 * frac, num_chunks = NUM_CHUNKS, update_gpu = True)
        loss, _, _, _, _, _, _ = model.make_batch(heads, ent_coef = 0.01 * frac, add_merge = True, update_gpu = True, num_chunks = NUM_CHUNKS, lr = lr_scheduler.lrA)
        for substep in range(TRAIN_LOOP):
            approxkl = model.trainA(lr_scheduler.lrA / (TRAIN_LOOP), num_chunks = NUM_CHUNKS, ent_coef = 0.01 * frac)
            if approxkl > 0.01:
                print("KL-divergence {:.3f}, early stopping at iter {:d}.".format(approxkl, substep))
                break
        if approxkl <= 0.01:
            print("KL-divergence {:.3f}, updated full gradient step.".format(approxkl))
        
        lossP, _, _, _, _, _, dictA = model.make_batch(ent_coef = 0.01 * frac, add_merge = True, num_chunks = NUM_CHUNKS, lr = lr_scheduler.lrA)
        lr_scheduler.step(model, approxkl, loss, lossP)
        model.debug_merge(dictA)
        [head.replayBuffer.clear() for head in heads]


