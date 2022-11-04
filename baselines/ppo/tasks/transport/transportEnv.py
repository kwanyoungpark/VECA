import numpy as np
import socket
import struct
import os
import time
import random
from PIL import Image
from tasks.transport.utils import getRewardFromPos
from tasks.transport.constants import *
from tasks.transport.VECA.environment import GeneralEnvironment
import pickle

class Environment(GeneralEnvironment):
    def __init__(self, num_envs, port = 8870):
        GeneralEnvironment.__init__(self, num_envs, port)
        self.name = 'Transport'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
            #'touch': (1182 + 2 * 66) # GrabObject Before
            'touch': (1182 + 2 * 58)
        }
        if TACTILE is False:
            self.observation_space['touch'] = (2 * 66)

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, dones, info, touchs = [], [], [], []
        posLs, posRs, posOs, deltas, timeover = [], [], [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        TACTILE_L = 1182#self.observation_space['tactile']
        
        #print(data.keys())
        #print(self.obj_data.keys())
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            if img.shape[0] == 3 * IMG_C: #RGB -> G
                #print("Input is RGB, but using grayscale in setting. Converting...")
                temp_imgs = []
                for i in range(IMG_C):
                    temp_imgs.append(0.299 * img[3*i] + 0.587 * img[3*i+1] + 0.114 * img[3*i+2])
                img = np.stack(temp_imgs, axis = 0)
            imgs.append(img)

            angle = data['angle'][i]
            vel = data['vel'][i]
            if TACTILE:
                touch = data['tactile'][i]
                #print("touch", len(touch), "angle", len(angle),"vel",len(vel))
                touchs.append(np.concatenate([touch, angle, vel], axis = 0))
            else:
                touchs.append(np.concatenate([angle, vel], axis = 0))

            done = data['done'][i][0]
            reward = data['reward'][i][0]
            if done: dones.append(True)
            else: dones.append(False)
            rewards.append(reward)

            posL = data['handposL'][i]
            posR = data['handposR'][i]
            posO = data['objpos'][i]
            delta = data['delta'][i]
            _timeover = data['timeover'][i]

            posOs.append(posO)
            posRs.append(posR)
            posLs.append(posL)
            deltas.append(delta)
            timeover.append(_timeover)

        #info = (data['pos'], data['campos'])
        posRs = np.array(posRs)
        posLs = np.array(posLs)
        posOs = np.array(posOs)
        deltas = np.array(deltas)
        timeover = np.array(timeover)

        #info = (posA, posL, posAL, posF, posAF, timeover)

        #print(np.linalg.norm(posAL, axis = 1))
        info = (posLs, posRs, posOs,deltas, timeover)
        imgs = np.array(imgs)
        obs = {'image':imgs, 'touch':touchs}
        return (obs, rewards, dones, info)

    def step(self, action, ignore_agent_dim = True):
        super().send_action(action)
        return self.collect_observations(ignore_agent_dim)
