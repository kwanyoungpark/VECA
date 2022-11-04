import tensorflow as tf
from tasks.pendulum.utils import *
#from model import Model
#from PPOmodel.easy_model import Model
import numpy as np
import sys
import os
import cv2 
import time
import copy
from PIL import Image
from constants import BUFFER_LENGTH, RNN, STATE_LENGTH
from tasks.pendulum.constants import FRAME_SKIP
from tasks.pendulum.replaybuffer import ReplayBuffer
import pickle

class HeadQuarter():
    def __init__(self, env, model):
        NUM_AGENTS = env.num_envs
        NUM_TIME = 1
        TACTILE_LENGTH = env.observation_space['touch']
        ACTION_LENGTH = env.action_space

        self.touch0 = np.zeros([NUM_AGENTS, NUM_TIME, TACTILE_LENGTH])
        self.touch1 = np.zeros([NUM_AGENTS, NUM_TIME, TACTILE_LENGTH])

        if RNN:
            self.state0 = np.zeros([NUM_AGENTS, NUM_TIME, STATE_LENGTH])
            self.state1 = np.zeros([NUM_AGENTS, NUM_TIME, STATE_LENGTH])

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
        touchs = obs
        for i in range(self.env.num_envs):
            if mask[i] == False:
                continue
            touch = touchs[i]
            self.touch1[i] = np.array([touch] * NUM_TIME)
            if RNN:
                self.state1[i] = np.zeros(STATE_LENGTH)
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
    
    def add_replay_buf(self, buf):
        if RNN:
            buf.add_replay(self.touch0, self.touch1, self.action, self.helper_reward, self.raw_reward, self.done, self.state0, self.state1)
        else:
            buf.add_replay(self.touch0, self.touch1, self.action, self.helper_reward, self.raw_reward, self.done)
        
    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def step(self, add_replay = True):
        NUM_AGENTS = self.env.num_envs
        NUM_TIME = 1
        TACTILE_LENGTH = self.env.observation_space['touch']
        ACTION_LENGTH = self.env.action_space

        data = {'touch': self.touch1}
        if RNN:
            data.update({'state':self.state0.copy()})
        if self.env.mode == 'CONT':
            if RNN:
                _, _, self.action, self.state1 = self.model.get_action(data)
                self.state1 = np.reshape(self.state1, [NUM_AGENTS, NUM_TIME, STATE_LENGTH])
            else:
                _, _, self.action = self.model.get_action(data)
        elif self.env.mode == 'DISC':
            if RNN:
                _, self.action, self.state1 = self.model.get_action(data)
                self.state1 = np.reshape(self.state1, [NUM_AGENTS, NUM_TIME, STATE_LENGTH])
            else:
                _, self.action = self.model.get_action(data)

        self.touch0 = self.touch1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        touchs1 = np.zeros([NUM_AGENTS, TACTILE_LENGTH])
        action = self.action
        for _ in range(FRAME_SKIP):
            obs, rewards, done, infos = self.env.step(action)
            touchs = obs['touch']
            touchs1 = touchs.copy()
            for j in range(NUM_AGENTS):
                self.helper_reward[j] = 0.
                self.raw_reward[j] += rewards[j]
                self.done[j] = done[j]

        #print(self.touch1.shape)
        for i in range(NUM_AGENTS):
            self.touch1[i] = np.append(self.touch1[i][1:], [touchs1[i]], axis=0)
        if add_replay:
            self.add_replay()
        if RNN:
            self.state0 = self.state1.copy()
        self.restart(done)
        return done
