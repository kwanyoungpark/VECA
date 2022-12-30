#from model import *
import numpy as np
import socket
import struct
import os
import time
import random
import gym
from PIL import Image
from tasks.atari.wrapper import AtariGymWrapper as Wrapper

class Environment():
    def __init__(self, num_envs, env_name):
        self.num_envs = num_envs
        self.agents_per_env = 1
        self.num_agents = self.num_envs * self.agents_per_env
        self.envs = []
        for _ in range(self.num_envs):
            #env = gym.make(env_name + 'Deterministic-v4')
            env = gym.make(env_name + 'NoFrameskip-v4')
            assert 'NoFrameskip' in env.spec.id
            wrap_env = Wrapper.wrap(env)
            self.envs.append(wrap_env)
        self.name = env_name
        self.SIM = 'ATARI'
        self.mode = 'DISC'
        self.action_space = env.action_space.n
        print(env.action_space)
        print(self.action_space)
        self.RGB = False
        self.observation_space = {
            'image': (4, 84, 84)
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
        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        rewards, done, info = [], [], []
        imgs = []
        for i in range(self.num_envs):
            env = self.envs[i]
            act = np.argmax(action[i])
            obs, reward, doneA, _ = env.step(act)
            #print('got observation':)
            #img = list(reversed(readBytes(data[:IMG_DATAL], 'uint8')))
            #wav = readBytes(data[IMG_DATAL:IMG_DATAL + 2 * WAV_DATAL], 'int16')
            img = np.array(obs)
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            imgs.append(img)
            #others.append(other)
            rewards.append(reward)
            if doneA: done.append(True)
            else: done.append(False)
        imgs = np.array(imgs)
        obs = {'image': imgs}
        #print(img.shape, np.max(obs))
        return (obs, rewards, done, info)
