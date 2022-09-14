import numpy as np
import struct
import os
import time
import random
#from PIL import Image
from veca.gym.core.environment import EnvModule
import pickle

VISION = False
AUDIO = False
TACTILE = True

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
    def __init__(self, task,num_envs,ip,port,  args):
        EnvModule.__init__(self,task, num_envs,ip,port, args, 
            exec_path_win = "veca/env_manager/bin/babyrun/VECA-BS.exe",
            download_link_win = "https://drive.google.com/uc?export=download&id=1fbpQffo30ULbInX21NqP6U5nuEycKtW0",
            exec_path_linux = ,
            download_link_linux = 
            )
        self.name = 'RunBaby'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        
        self.observation_space = {
            'image': (6, 84, 84),
            'touch': (5 * 82 + 9888)
        }
        if TACTILE is False:
            self.observation_space['touch'] = (5 * 82)

        #self.tracker = momentsTracker(self.observation_space['touch'], 0.)

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, dones, info, touchs = [], [], [], []
        #posLs, posRs, posOs = [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        TACTILE_L = self.observation_space['touch']
        
        #print(data.keys())
        #print(self.obj_data.keys())
        for i in range(self.num_envs):
            if VISION:
                img = list(reversed(data['img'][i]))
                img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
                imgs.append(img)
            obsP = data['obs'][i]
            if TACTILE:
                touch = data['tactile'][i]
                touchs.append(np.concatenate([touch, obsP], axis = 0))
            else:
                touchs.append(obsP)

            #done = data['done'][i][0]
            done = False
            reward = data['reward'][i][0]
            if done: dones.append(True)
            else: dones.append(False)
            rewards.append(reward)
        info = None
        touchs = np.array(touchs)
        if VISION:
            imgs = np.array(imgs)
            obs = {'touch':touchs, 'image': imgs,}
        else:
            obs = {'touch':touchs,}
        return (obs, rewards, dones, info)

