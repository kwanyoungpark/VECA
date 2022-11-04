import tensorflow as tf
from tasks.atari.utils import *
#from model import Model
#from PPOmodel.easy_model import Model
import numpy as np
import sys
import os
import cv2 
import time
import copy
from PIL import Image
from constants import BUFFER_LENGTH
from tasks.atari.constants import FRAME_SKIP
from tasks.atari.replaybuffer import ReplayBuffer
import pickle

class HeadQuarter():
    def __init__(self, env, model):
        NUM_AGENTS = env.num_envs
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = env.observation_space['image']
        ACTION_LENGTH = env.action_space

        self.img0 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.img1 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.action = np.zeros([NUM_AGENTS, ACTION_LENGTH])
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        self.done = np.zeros([NUM_AGENTS], dtype=bool)

        self.replayBuffer = ReplayBuffer(BUFFER_LENGTH = BUFFER_LENGTH, env = env)
        self.env = env
        self.model = model
        self.restart()

    def restart(self, mask = None):
        NUM_TIME = 1
        if mask is None:
            mask = np.ones(self.env.num_envs, dtype = np.bool)
        obs = self.env.reset(mask)
        imgs = obs
        for i in range(self.env.num_envs):
            if mask[i] == False:
                continue
            img = imgs[i]
            self.img1[i] = np.array([img] * NUM_TIME)
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
    
    def add_replay_buf(self, buf):
        buf.add_replay(self.img0, self.img1, self.action, self.helper_reward, self.raw_reward, self.done)
        
    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def step(self, add_replay = True):
        NUM_AGENTS = self.env.num_envs
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = self.env.observation_space['image']
        ACTION_LENGTH = self.env.action_space

        data = {'image': self.img1}
        if self.env.mode == 'CONT':
            _, _, self.action = self.model.get_action(data)
        elif self.env.mode == 'DISC':
            _, self.action = self.model.get_action(data)

        self.img0 = self.img1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        imgs1 = np.zeros([NUM_AGENTS, IMG_CHANNEL, IMG_H, IMG_W])
        action = self.action
        for _ in range(FRAME_SKIP):
            obs, rewards, done, infos = self.env.step(action)
            imgs = obs['image']
            imgs1 = imgs.copy()
            for j in range(NUM_AGENTS):
                self.helper_reward[j] = 0.
                self.raw_reward[j] += rewards[j]
                self.done[j] = done[j]

        for i in range(NUM_AGENTS):
            self.img1[i] = np.append(self.img1[i][1:], [imgs1[i]], axis=0)
        if add_replay:
            self.add_replay()
        self.restart(done)
        return done
