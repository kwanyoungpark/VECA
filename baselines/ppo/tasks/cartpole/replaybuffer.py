import tensorflow as tf
#from model import Model
#from PPOmodel.easy_model import Model
import numpy as np
import sys
import os
import cv2 
import time
import copy
from PIL import Image
from constants import *

class ReplayBuffer():
    def __init__(self, BUFFER_LENGTH, env):
        self.BUFFER_LENGTH = BUFFER_LENGTH
        NUM_AGENTS = env.num_envs
        NUM_TIME = 1
        TACTILE_LENGTH = env.observation_space['touch']
        ACTION_LENGTH = env.action_space

        self.data = {
            'touch0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, TACTILE_LENGTH]),
            'touch1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, TACTILE_LENGTH]),
            'done': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS], dtype=bool),
            'action': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, ACTION_LENGTH]),
            'helper_reward': np.zeros([BUFFER_LENGTH, NUM_AGENTS]),
            'raw_reward': np.zeros([BUFFER_LENGTH, NUM_AGENTS])
        }
        if RNN:
            self.data['state0'] = np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, STATE_LENGTH])
            self.data['state1'] = np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, STATE_LENGTH])
        self.replayBufSize = 0
        #self.clear()

    def clear(self):
        self.replayBufSize = 0
        for key in self.data.keys():
            self.data[key].fill(0)
        print('cleared!')

    def add_replay(self, touch0, touch1, action, helper_reward, raw_reward, done, state0 = None, state1 = None):
        if self.replayBufSize < self.BUFFER_LENGTH:
            ind = self.replayBufSize
            self.replayBufSize += 1
        else:
            ind = np.random.randint(self.replayBufSize)
        self.data['touch0'][ind] = touch0.copy()
        self.data['touch1'][ind] = touch1.copy()
        self.data['action'][ind] = action.copy()
        self.data['helper_reward'][ind] = helper_reward.copy()
        self.data['raw_reward'][ind] = raw_reward.copy()
        #self.data['myu'][ind] = myu.copy()
        #self.data['sigma'][ind] = sigma.copy()
        self.data['done'][ind] = done.copy()
        if RNN:
            self.data['state0'][ind] = state0.copy()
            self.data['state1'][ind] = state1.copy()

    def get_batch(self, batch_per_agent = -1):
        if batch_per_agent == -1:
            return self.data.copy()
        ind = np.random.choice(self.replayBufSize, batch_per_agent)
        batch = {}
        for key in self.data:
            dat = self.data[key][ind].copy()
            batch[key] = np.reshape(dat, dat.shape[:2] + (-1,))
        return batch
