import tensorflow as tf
from tasks.MazeNav.utils import *
#from model import Model
#from PPOmodel.easy_model import Model
import numpy as np
import sys
import os
import cv2 
import time
import copy
from PIL import Image
from constants import BUFFER_LENGTH, STATE_LENGTH, RNN
from tasks.MazeNav.constants import FRAME_SKIP, NUM_OBJS
from tasks.MazeNav.replaybuffer import ReplayBuffer
import pickle

class HeadQuarter():
    def __init__(self, env, model):
        NUM_AGENTS = env.num_agents
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = env.observation_space['image']
        ACTION_LENGTH = env.action_space
        VEC_OBJ = env.VEC_OBJ

        self.img0 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.img1 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])

        self.obj0 = np.zeros([NUM_AGENTS, NUM_OBJS])
        self.obj1 = np.zeros([NUM_AGENTS, NUM_OBJS])

        if RNN:
            self.state0 = np.zeros([NUM_AGENTS, STATE_LENGTH])
            self.state1 = np.zeros([NUM_AGENTS, STATE_LENGTH])

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
        self.env.reset()
        obs, rewards, done, infos = self.env.step(np.zeros([self.env.num_agents, self.env.action_space]))
        imgs, objs = obs['image'], obs['obj']
        #print(others.shape)
        print('first img', imgs.mean())
        for i in range(self.env.num_envs):
            img = imgs[i]
            self.img1[i] = np.array([img] * NUM_TIME)
            if RNN:
                self.state1[i] = np.zeros(STATE_LENGTH)
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
            self.obj1[i] = np.array(objs[i])
        #_, _, _, _ = infos
    
    def add_replay_buf(self, buf):
        if RNN:
            buf.add_replay(self.img0, self.img1, self.action, self.helper_reward, self.raw_reward, self.obj0, self.done, self.state0, self.state1)
        else:
            buf.add_replay(self.img0, self.img1, self.action, self.helper_reward, self.raw_reward, self.obj0, self.done)
        
    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def send_action(self):
        data = {'image':self.img1, 'obj':self.obj1}
        if RNN:
            data.update({'state':self.state0.copy()})
        if self.env.mode == 'DISC':
            if RNN:
                _, self.action, self.state1 = self.model.get_action(data)
            else:
                _, self.action = self.model.get_action(data)
        else:
            if RNN:
                _, _, self.action, self.state1 = self.model.get_action(data)
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
        NUM_AGENTS = self.env.num_agents
        IMG_CHANNEL, IMG_H, IMG_W = self.env.observation_space['image']
        
        #print(self.pos0)
        #self.myu, self.sigma, self.action = self.model.get_action(self.pos1)
        self.img0 = self.img1.copy()
        self.obj0 = self.obj1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        imgs1 = np.zeros([NUM_AGENTS, IMG_CHANNEL, IMG_H, IMG_W])
        
        obs, rewards, done, infos = self.env.collect_observations()
        _, _, _, Hreward = infos
        imgs = obs['image']
        objs = obs['obj']
        imgs1 = imgs.copy()
        self.obj1 = objs.copy()
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
            self.helper_reward[j] += Hreward[j]#getRewardFromCamPos(self.pos0[j], action[j], cheat = False)
            #self.helper_reward[j] -= 0.3*getRewardFromFakeCamPos(self.posF0[j], action[j], cheat = False)
            self.raw_reward[j] += rewards[j]
            self.done[j] = done[j]
            #print(self.helper_reward[j], self.raw_reward[j])

        for i in range(NUM_AGENTS):
            #print(self.img[i].mean(), self.img[i].var())
            self.img1[i] = np.append(self.img1[i][1:], [imgs1[i]], axis=0)
        if add_replay:
            self.add_replay()
        if RNN:
            self.state0 = self.state1.copy()
        return done
       
    def step(self, add_replay = True):
        self.send_action()
        return self.collect_observations(add_replay)

