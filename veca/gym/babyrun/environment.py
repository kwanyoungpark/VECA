import numpy as np
import struct
import os
import time
import random
from PIL import Image
from veca.gym.core.environment import EnvModule
import pickle

VISION = True
AUDIO = False
SIM = 'VECA'
MODE = 'CONT'
VEC_OBJ = True
if VEC_OBJ:
    OBJ_NAME = ['ball', 'toypig', 'pyramid']
    NUM_OBJS = len(OBJ_NAME)
NUM_DEGS = 15
#RGB = False
#IMG_CHANNEL = 6 if RGB else 2
IMG_H, IMG_W = 84, 84
NUM_TIME = 1
FRAME_SKIP = 1

class Environment(EnvModule):
    def __init__(self, num_envs,ip,port,  args):
        EnvModule.__init__(self, num_envs,ip,port, args)
        self.name = 'BabyRun'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.stepnum = np.zeros(num_envs)
        self.episode = np.zeros(num_envs)
        self.observation_space = {
            'image': (6, 84, 84),
            'obj': (NUM_OBJS)
        }
        self.VEC_OBJ = VEC_OBJ
        self.NUM_OBJS = NUM_OBJS
        self.action_length = self.action_space - 1
        self.prev_dist = np.zeros(num_envs)
        #self.action_space = 2

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, done, info, objs = [], [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        
        #print(data.keys())
        #print(self.obj_data.keys())
        mask = np.zeros(self.num_envs, dtype = np.bool)
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            imgs.append(img)
            rewards.append(reward)
            if doneA:
                done.append(True)
            else:
                done.append(False)
        self.reset(mask)
        imgs = np.array(imgs)
        info = {}
        obs = {'image': imgs,}
        return (obs, rewards, done, info)

    def send_action(self, action):
        super().send_action(action)
