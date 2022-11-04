import tensorflow as tf
from tasks.kicktheball.utils import *
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
from tasks.kicktheball.constants import FRAME_SKIP
from tasks.kicktheball.replaybuffer import ReplayBuffer
import pickle

class HeadQuarter():
    def __init__(self, env, model):
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = env.observation_space['image']
        WAV_CHANNEL, WAV_LENGTH = env.observation_space['audio']
        NUM_AGENTS = env.num_envs
        ACTION_LENGTH = env.action_space

        self.img0 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.img1 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.wav0 = np.zeros([NUM_AGENTS, NUM_TIME, WAV_CHANNEL, WAV_LENGTH])
        self.wav1 = np.zeros([NUM_AGENTS, NUM_TIME, WAV_CHANNEL, WAV_LENGTH])
        self.pos0 = np.zeros([NUM_AGENTS, 3])
        self.pos1 = np.zeros([NUM_AGENTS, 3])
        self.action = np.zeros([NUM_AGENTS, ACTION_LENGTH])
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        self.done = np.zeros([NUM_AGENTS], dtype=bool)

        self.replayBuffer = ReplayBuffer(BUFFER_LENGTH = BUFFER_LENGTH, env = env)
        self.env = env
        self.model = model
        self.restart()

    def restart(self):
        NUM_TIME = 1
        NUM_AGENTS, ACTION_LENGTH = self.env.num_agents, self.env.action_space
        self.env.reset()
        obs, rewards, done, infos = self.env.step(np.zeros([NUM_AGENTS, ACTION_LENGTH]))
        imgs, wavs = obs['image'], obs['audio']
        #print(others.shape)
        print('first img', imgs.mean())
        for i in range(NUM_AGENTS):
            img, wav = imgs[i], wavs[i]
            self.img1[i] = np.array([img] * NUM_TIME)
            self.wav1[i] = np.array([wav] * NUM_TIME)
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
        self.pos1 = infos
    
    def add_replay_buf(self, buf):
        buf.add_replay(self.img0, self.img1, self.wav0, self.wav1, self.action, self.helper_reward, self.raw_reward, self.pos0, self.pos1, self.done)
        
    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def send_action(self):
        data = {'image':self.img1, 'audio':self.wav1}
        if self.env.mode == 'DISC':
            _, self.action = self.model.get_action(data)
        else:
            _, _, self.action = self.model.get_action(data)
        if self.env.mode == 'DISC':
            action = np.zeros([NUM_AGENTS, 2])
            for i in range(NUM_AGENTS):
                if self.action[i][0] == 1:
                    action[i][0] = 1
                if self.action[i][1] == 1:
                    action[i][1] = -1
                if self.action[i][2] == 1:
                    action[i][1] = 1
        else:
            action = np.tanh(self.action.copy())
            action[:, 0] = action[:, 0] * 0.8
            #action[:, 0] = np.maximum(action[:, 0], 0)
            action[:, 0] = (action[:, 0] + 1) / 2
            #print(action)
        action[:, 1] = action[:, 1] * 3
        self.env.send_action(action)

    def collect_observations(self, add_replay = True):
        NUM_AGENTS = self.env.num_envs
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = self.env.observation_space['image']
        WAV_CHANNEL, WAV_LENGTH = self.env.observation_space['audio']
        ACTION_LENGTH = self.env.action_space

        #print(self.pos0)
        #self.myu, self.sigma, self.action = self.model.get_action(self.pos1)
        self.img0 = self.img1.copy()
        self.wav0 = self.wav1.copy()
        self.pos0 = self.pos1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        imgs1 = np.zeros([NUM_AGENTS, IMG_CHANNEL, IMG_H, IMG_W])
        wavs1 = np.zeros([NUM_AGENTS, WAV_CHANNEL, WAV_LENGTH])
        
        obs, rewards, done, infos = self.env.collect_observations()
        imgs, wavs = obs['image'], obs['audio']
        imgs1 = imgs.copy()
        wavs1 = wavs.copy()
        self.pos1 = infos
        #print("Reward : ")
        if self.env.mode == 'DISC':
            action = np.zeros([NUM_AGENTS, 2])
            for i in range(NUM_AGENTS):
                if self.action[i][0] == 1:
                    action[i][0] = 1
                if self.action[i][1] == 1:
                    action[i][1] = -1
                if self.action[i][2] == 1:
                    action[i][1] = 1
        else:
            action = np.tanh(self.action.copy())
            action[:, 0] = action[:, 0] * 0.8
            #action[:, 0] = np.maximum(action[:, 0], 0)
            action[:, 0] = (action[:, 0] + 1) / 2
            #print(action)
        action[:, 1] = action[:, 1] * 3
        for j in range(NUM_AGENTS):
            #self.helper_reward[j] = 0.01*(np.linalg.norm(self.posA0[j]) - np.linalg.norm(self.posA1[j]) - 0.05)
            #if self.helper_reward[j] < -0.01:
            #    self.helper_reward[j] = 0
            self.helper_reward[j] += getRewardFromPos(self.pos0[j], action[j], wavs[j])
            #self.helper_reward[j] -= 0.3*getRewardFromFakeCamPos(self.posF0[j], action[j], cheat = False)
            self.raw_reward[j] += rewards[j]
            self.done[j] = done[j]
            #print(self.helper_reward[j], self.raw_reward[j])

        for i in range(NUM_AGENTS):
            #print(self.img[i].mean(), self.img[i].var())
            self.img1[i] = np.append(self.img1[i][1:], [imgs1[i]], axis=0)
            self.wav1[i] = np.append(self.wav1[i][1:], [wavs1[i]], axis=0)
    
        if add_replay:
            self.add_replay()
        return done

    def step(self, add_replay = True):
        self.send_action()
        return self.collect_observations(add_replay)
