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
        IMG_CHANNEL, IMG_H, IMG_W = env.observation_space['image']
        WAV_CHANNEL, WAV_LENGTH = env.observation_space['audio']
        _, RAW_WAV_LENGTH = env.observation_space['real_audio']
        NUM_TACTILE = env.observation_space['touch']
        NUM_REAL_TACTILE = env.observation_space['real_touch']
        ACTION_LENGTH = env.action_space

        self.data = {
            'img0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W]),
            'img1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W]),
            'wav0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, WAV_CHANNEL, WAV_LENGTH]),
            'wav1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, WAV_CHANNEL, WAV_LENGTH]),
            'Rwav0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, WAV_CHANNEL, RAW_WAV_LENGTH]),
            'Rwav1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, WAV_CHANNEL, RAW_WAV_LENGTH]),
            'touch0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, NUM_TACTILE]),
            'touch1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, NUM_TACTILE]),
            'Rtouch0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, NUM_REAL_TACTILE]),
            'Rtouch1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, NUM_TIME, NUM_REAL_TACTILE]),
            'pos0': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, 3]),
            'pos1': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, 3]),
            'objimg': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, 2]),
            'objwav': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS]),
            'objtouch': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS]),
            #'myu': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, ACTION_LENGTH]),
            #'sigma': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, ACTION_LENGTH]),
            #'pos': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, ACTION_LENGTH]),
            'done': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS], dtype=bool),
            'action': np.zeros([self.BUFFER_LENGTH, NUM_AGENTS, ACTION_LENGTH]),
            'helper_reward': np.zeros([BUFFER_LENGTH, NUM_AGENTS]),
            'raw_reward': np.zeros([BUFFER_LENGTH, NUM_AGENTS])
        }
        self.replayBufSize = 0
        #self.clear()

    def clear(self):
        self.replayBufSize = 0
        for key in self.data.keys():
            self.data[key].fill(0)
        print('cleared!')

    def add_replay(self, img0, img1, wav0, wav1, Rwav0, Rwav1, touch0, touch1, Rtouch0, Rtouch1, action, helper_reward, raw_reward, pos0, pos1, done, objimg, objwav, objtouch):
        if self.replayBufSize < self.BUFFER_LENGTH:
            ind = self.replayBufSize
            self.replayBufSize += 1
        else:
            ind = np.random.randint(self.replayBufSize)
        self.data['img0'][ind] = img0.copy()
        self.data['img1'][ind] = img1.copy()
        self.data['wav0'][ind] = wav0.copy()
        self.data['wav1'][ind] = wav1.copy()
        self.data['Rwav0'][ind] = Rwav0.copy()
        self.data['Rwav1'][ind] = Rwav1.copy()
        self.data['touch0'][ind] = touch0.copy()
        self.data['touch1'][ind] = touch1.copy()
        self.data['Rtouch0'][ind] = Rtouch0.copy()
        self.data['Rtouch1'][ind] = Rtouch1.copy()
        self.data['action'][ind] = action.copy()
        self.data['helper_reward'][ind] = helper_reward.copy()
        self.data['raw_reward'][ind] = raw_reward.copy()
        #self.data['myu'][ind] = myu.copy()
        #self.data['sigma'][ind] = sigma.copy()
        self.data['done'][ind] = done.copy()
        self.data['pos0'][ind] = pos0.copy()
        self.data['pos1'][ind] = pos1.copy()
        self.data['objimg'][ind] = objimg.copy()
        self.data['objwav'][ind] = objwav.copy()
        self.data['objtouch'][ind] = objtouch.copy()

    def get_batch(self, batch_per_agent = -1):
        if batch_per_agent == -1:
            return self.data.copy()
        ind = np.random.choice(self.replayBufSize, batch_per_agent)
        batch = {}
        for key in self.data:
            dat = self.data[key][ind].copy()
            batch[key] = np.reshape(dat, dat.shape[:2] + (-1,))
        return batch
