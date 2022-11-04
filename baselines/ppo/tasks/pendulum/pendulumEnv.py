#from model import *
import numpy as np
import socket
import struct
import os
import time
import random
import gym
from PIL import Image

class Environment():
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.agents_per_env = 1
        self.num_agents = self.num_envs * self.agents_per_env
        self.envs = []
        for _ in range(self.num_envs):
            #env = gym.make(env_name + 'Deterministic-v4')
            env = gym.make('Pendulum-v0')
            #wrap_env = Wrapper.wrap(env)
            self.envs.append(env)
        self.name = 'Pendulum'
        self.SIM = 'Pendulum'
        self.mode = 'CONT'
        self.action_space = 1
        print(env.action_space)
        print(self.action_space)
        self.RGB = False
        self.observation_space = {
            'touch': 3
        }
    
    def reset(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype = np.bool)
        obs = []
        for i in range(self.num_envs):
            if mask[i]:
                ob = self.envs[i].reset()
                obs.append(np.array(ob))
            else:
                obs.append(None)
        return obs

    def step(self, action):
        TACTILE_LENGTH = self.observation_space['touch']
        rewards, done, info = [], [], []
        touchs = []
        for i in range(self.num_envs):
            env = self.envs[i]
            act = action[i] * 2
            obs, reward, doneA, _ = env.step(act)
            touch = np.array(obs)
            touchs.append(touch)
            rewards.append(reward)
            if doneA: done.append(True)
            else: done.append(False)
        touchs = np.array(touchs)
        obs = {'touch': touchs}
        #print(img.shape, np.max(obs))
        return (obs, rewards, done, info)
