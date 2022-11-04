import tensorflow as tf
from tasks.transport.utils import *
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
from tasks.transport.replaybuffer import ReplayBuffer
from munch import Munch
import pickle

class HeadQuarter():
    def __init__(self, env, model):
        NUM_AGENTS = env.num_agents
        NUM_TIME = 1
        IMG_C, IMG_H, IMG_W = env.observation_space['image']
        TACTILE_LENGTH = env.observation_space['touch']
        ACTION_LENGTH = env.action_space

        self.img0 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_C, IMG_H, IMG_W])
        self.img1 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_C, IMG_H, IMG_W])
        self.touch0 = np.zeros([NUM_AGENTS, NUM_TIME, TACTILE_LENGTH])
        self.touch1 = np.zeros([NUM_AGENTS, NUM_TIME, TACTILE_LENGTH])

        self.posHL0 = np.zeros([NUM_AGENTS, 3])
        self.posHL1 = np.zeros([NUM_AGENTS, 3])
        self.posHR0 = np.zeros([NUM_AGENTS, 3])
        self.posHR1 = np.zeros([NUM_AGENTS, 3])
        self.posO0 = np.zeros([NUM_AGENTS, 3])
        self.posO1 = np.zeros([NUM_AGENTS, 3])
        self.delta0 = np.zeros([NUM_AGENTS, 3])
        self.delta1 = np.zeros([NUM_AGENTS, 3])
    
        self.action = np.zeros([NUM_AGENTS, ACTION_LENGTH])
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        self.done = np.zeros([NUM_AGENTS], dtype=bool)
        self.timeover = np.zeros([NUM_AGENTS], dtype=bool)
        self.replayBuffer = ReplayBuffer(BUFFER_LENGTH = BUFFER_LENGTH, env = env)
        self.env = env
        self.model = model
        self.restart()
        
        self.num_steps = 0

    def restart(self):
        NUM_TIME = 1
        self.env.reset()
        obs, rewards, done, infos = self.env.step(np.zeros([self.env.num_agents, self.env.action_space]))
        imgs, touchs = obs['image'], obs['touch']
        print('first img', imgs.mean())
        for i in range(self.env.num_envs):
            img = imgs[i]
            touch = touchs[i]
            self.img1[i] = np.array([img] * NUM_TIME)
            self.touch1[i] = np.array([touch] * NUM_TIME)
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
        self.posHL1, self.posHR1, self.posO1, self.delta1, self.timeover = infos
    
    def add_replay_buf(self, buf):
        buf.add_replay(self.img0, self.img1, self.touch0, self.touch1, self.action, self.helper_reward, self.raw_reward, self.done)
        
    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def send_action(self):
        data = {'image': self.img1, 'touch': self.touch1}
        if self.env.mode == 'DISC':
            _, self.action = self.model.get_action(data)
        else:
            _, _, self.action = self.model.get_action(data)
        action = np.tanh(self.action)
        #print(action[:, 60:])
        self.env.send_action(action)

    def collect_observations(self, add_replay = True):
        NUM_AGENTS = self.env.num_agents
        IMG_C, IMG_H, IMG_W = self.env.observation_space['image']
        TACTILE_LENGTH = self.env.observation_space['touch']
        
        self.img0 = self.img1.copy()
        self.touch0 = self.touch1.copy()
        self.posHL0 = self.posHL1.copy()
        self.posHR0 = self.posHR1.copy()
        self.posO0 = self.posO1.copy()
        self.delta0 = self.delta1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        
        obs, rewards, done, infos = self.env.collect_observations()
        imgs, touchs = obs['image'], obs['touch']
        imgs1, touchs1 = imgs.copy(), touchs.copy()
        self.posHL1, self.posHR1, self.posO1, self.delta1, self.timeover = infos

        for j in range(NUM_AGENTS):
            # Insert your Curriculum 
            agent_info = Munch({
                'posHL0':self.posHL0[j], 
                'posHR0':self.posHR0[j],
                'posO0': self.posO0[j],
                'posHL1': self.posHL1[j],
                'posHR1': self.posHR1[j],
                'posO1': self.posO1[j],
                'delta1': self.delta1[j],
                'action': self.action[j],
                'timeover': self.timeover[j]
            })
            #self.helper_reward[j] += getRewardFromPos(self.posHL0[j], self.posHR0[j], self.posO0[j], self.posHL1[j], self.posHR1[j], self.posO1[j])
            #self.helper_reward[j] -= 4e-3 * np.mean(self.action[j]*self.action[j])
            self.helper_reward[j] = self.curriculum(self.num_steps,agent_info)
            
            self.raw_reward[j] += 10 * rewards[j] 
            self.done[j] = done[j]

        for i in range(NUM_AGENTS):
            self.img1[i] = np.append(self.img1[i][1:], [imgs1[i]], axis = 0)
            self.touch1[i] = np.append(self.touch1[i][1:], [touchs1[i]], axis=0)
    
        if add_replay:
            self.add_replay()
        return done
       
    def step(self, add_replay = True):
        self.send_action()
        self.num_steps += 1
        return self.collect_observations(add_replay)
    
    def curriculum(self, num_iteration, info):

        reward_distance_from_hand = getRewardFromPos(info.posHL0, info.posHR0, info.posO0, info.posHL1, info.posHR1, info.posO1)
        reward_distance_from_target = 4e-4 * np.tanh(-np.linalg.norm(info.delta1))
        reward_entropy = 4e-3 * np.mean(info.action*info.action)
        
        helper_reward = reward_distance_from_hand + reward_distance_from_target + negative_reward
        
        if self.num_steps % 5000:
            print("Agent global steps:", global_steps)
        #helper_reward = reward_distance_from_hand + reward_distance_from_target + reward_entropy
        return helper_reward

