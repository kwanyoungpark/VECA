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
from tasks.runBaby.replaybuffer import ReplayBuffer
from munch import Munch
import pickle

class HeadQuarter():
    def __init__(self, env, model, curriculum):
        self.CURRICULUM = curriculum
        NUM_AGENTS = env.num_agents
        NUM_TIME = 1
        TACTILE_LENGTH = env.observation_space['touch']
        ACTION_LENGTH = env.action_space

        self.touch0 = np.zeros([NUM_AGENTS, NUM_TIME, TACTILE_LENGTH])
        self.touch1 = np.zeros([NUM_AGENTS, NUM_TIME, TACTILE_LENGTH])
    
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
        for _ in range(5):
            self.env.reset()
            for _ in range(3):
                self.env.step(np.zeros([self.env.num_agents, self.env.action_space]))
        for _ in range(32):
            obs, rewards, done, infos = self.env.step(np.zeros([self.env.num_agents, self.env.action_space]))
        touchs = obs['touch']
        print('first img', touchs.mean())
        for i in range(self.env.num_envs):
            touch = touchs[i]
            self.touch1[i] = np.array([touch] * NUM_TIME)
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
    
    def add_replay_buf(self, buf):
        buf.add_replay(self.touch0, self.touch1, self.action, self.helper_reward, self.raw_reward, self.done)
        
    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def send_action(self):
        data = {'touch': self.touch1}
        if self.env.mode == 'DISC':
            _, self.action = self.model.get_action(data)
        else:
            _, _, self.action = self.model.get_action(data)
        action = np.tanh(self.action)
        #print(action[:, 60:])
        self.env.send_action(action)

    def collect_observations(self, add_replay = True):
        NUM_AGENTS = self.env.num_agents
        TACTILE_LENGTH = self.env.observation_space['touch']
        
        self.touch0 = self.touch1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        
        obs, rewards, done, infos = self.env.collect_observations()
        touchs = obs['touch']
        touchs1 = touchs.copy()
        #self.posHL1, self.posHR1, self.posO1 = infos
        self.pos, self.vel, self.compos, self.comvel = infos

        for j in range(NUM_AGENTS):
            agent_info = Munch({
                'pos':self.pos[j],
                'vel':self.vel[j],
                'compos':self.compos[j],
                'comvel':self.comvel[j],
                'action':self.action[j],
            })
            #self.helper_reward[j] += getRewardFromPos(self.posHL0[j], self.posHR0[j], self.posO0[j], self.posHL1[j], self.posHR1[j], self.posO1[j])
            #self.helper_reward[j] -= 4e-3 * np.mean(self.action[j]*self.action[j])
            self.helper_reward[j] = self.curriculum(self.model.global_step,agent_info)
            

            self.raw_reward[j] += rewards[j] 
            #self.helper_reward[j] -= 1e-2 * np.mean(self.action[j] * self.action[j])
            self.done[j] = done[j]

        for i in range(NUM_AGENTS):
            self.touch1[i] = np.append(self.touch1[i][1:], [touchs1[i]], axis=0)
    
        if add_replay:
            self.add_replay()
        return done
       
    def step(self, add_replay = True):
        self.send_action()
        return self.collect_observations(add_replay)
 
    def curriculum(self, num_iteration, info):

        #Curriculum learning research code
        #reward_distance_from_hand = getRewardFromPos(info.posHL0, info.posHR0, info.posO0, info.posHL1, info.posHR1, info.posO1)
        #reward_distance_from_target = 4e-4 * np.tanh(-np.linalg.norm(info.delta1))
        reward_velocity = 1e-2 * info.comvel[0]
        reward_entropy = 1e-2 * np.mean(info.action*info.action) 
        helper_reward = reward_entropy + reward_velocity
        if global_steps % 5000:
            print("Agent global steps:", global_steps)
        return helper_reward

