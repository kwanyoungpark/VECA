import numpy as np
import socket
import struct
import os
import time
import random
from PIL import Image
from tasks.navigation.utils import getRewardFromPos, wav2freq
from tasks.navigation.constants import *
from VECA.environment import GeneralEnvironment
import pickle

class Environment(GeneralEnvironment):
    def __init__(self, num_envs, port = 8870):
        GeneralEnvironment.__init__(self, num_envs, port)
        self.name = 'Navigation'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
            'obj': (NUM_OBJS)
        }
        self.VEC_OBJ = VEC_OBJ
        self.NUM_OBJS = NUM_OBJS
        if VEC_OBJ == False:
            with open('navigation/data/data.pkl', 'rb') as F:
                self.obj_data = pickle.load(F)

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, done, info, objs = [], [], [], []
        posAL, posL, posF = [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        
        #print(data.keys())
        #print(self.obj_data.keys())
        for i in range(self.num_envs):
            if 'img' in data:
                img = list(reversed(data['img'][i]))
            else:
                img = np.zeros([IMG_C, IMG_H, IMG_W])
                print('NO IMAGE')
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            posA = data['pos'][i]
            pos = list(data['campos'][i])
            pos[0], pos[1], pos[2] = np.tanh(pos[0] - 0.5), pos[1], np.tanh(pos[2])
            posf = list(data['fakecampos'][i])
            posf[0], posf[1], posf[2] = np.tanh(posf[0] - 0.5), posf[1], np.tanh(posf[2])
            obj = str(data['obj'][i])
            if VEC_OBJ:
                obj_oh = np.zeros([NUM_OBJS])
                obj_oh[OBJ_NAME.index(obj)] = 1.
                objs.append(obj_oh)
            else:
                objs.append(np.reshape(self.obj_data[obj], [NUM_DEGS, IMG_H, IMG_W]))
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            if img.shape[0] == 3 * IMG_C: #RGB -> G
                #print("Input is RGB, but using grayscale in setting. Converting...")
                temp_imgs = []
                for i in range(IMG_C):
                    temp_imgs.append(0.299 * img[3*i] + 0.587 * img[3*i+1] + 0.114 * img[3*i+2])
                img = np.stack(temp_imgs, axis = 0)
            imgs.append(img)
            rewards.append(reward)
            posF.append(posf)
            posAL.append(posA)
            posL.append(pos)
            if doneA: done.append(True)
            else: done.append(False)
        #info = (data['pos'], data['campos'])
        posAL = np.array(posAL)
        posL = np.array(posL)
        posF = np.array(posF)
        #print(np.linalg.norm(posAL, axis = 1))
        info = (posL, posAL, posF)
        imgs = np.array(imgs)
        obs = {'image': imgs, 'obj': objs}
        return (obs, rewards, done, info)

    def step(self, action, ignore_agent_dim = True):
        super().send_action(action)
        return self.collect_observations(ignore_agent_dim)
