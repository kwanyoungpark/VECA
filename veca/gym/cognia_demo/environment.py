import numpy as np
import socket
import struct
import os
import time
import random
from PIL import Image
from veca.gym.core.environment import EnvModule
import pickle

class Environment(EnvModule):
    def __init__(self, num_envs, port = 8870):
        EnvModule.__init__(self, num_envs, ip, port, args)
        self.name = 'COGNIADemo'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
            'audio': (2, 2940)
        }
        #self.action_space = 2
        '''
        if VEC_OBJ == False:
            with open('navigation/data/data.pkl', 'rb') as F:
                self.obj_data = pickle.load(F)
        '''

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, done, info, Gobjs, Bobjs = [], [], [], [], []
        GposAL, BposAL, GposL, BposL = [], [], [], []
        imgs, wavs = [], []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        WAV_C, _ = self.observation_space['audio']
        
        #print(data.keys())
        mask = np.zeros(self.num_envs, dtype = np.bool)
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            wav = data['wav'][i]
            #print(np.array(wav).shape)
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            GposA, BposA = data['goodpos'][i], data['badpos'][i]
            Gpos, Bpos = list(data['goodcampos'][i]), list(data['badcampos'][i])
            Gpos[0], Gpos[1], Gpos[2] = np.tanh(Gpos[0] - 0.5), Gpos[1], np.tanh(Gpos[2])
            Bpos[0], Bpos[1], Bpos[2] = np.tanh(Bpos[0] - 0.5), Bpos[1], np.tanh(Bpos[2])
            Gobj, Bobj = str(data['goodobj'][i]), str(data['badobj'][i])
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            imgs.append(img)
            wav = np.reshape(np.array(wav), [WAV_C, -1]) / 32768.0
            wavs.append(wav)
            rewards.append(reward)
            GposAL.append(GposA)
            BposAL.append(BposA)
            GposL.append(Gpos)
            BposL.append(Bpos)
            Gobjs.append(Gobj)
            Bobjs.append(Bobj)
            if doneA: done.append(True)
            else: done.append(False)
        self.reset(done)
        #info = (data['pos'], data['campos'])
        GposAL = np.array(GposAL)
        BposAL = np.array(BposAL)
        posAL = np.stack([GposAL, BposAL], axis = 1)
        GposL = np.array(GposL)
        BposL = np.array(BposL)
        posL = np.stack([GposL, BposL], axis = 1)
        objs = [Gobjs, Bobjs]
        info = (posL, posAL, objs)

        imgs = np.array(imgs)
        wavs = np.array(wavs)
        obs = {'image': imgs, 'audio': wavs}

        rewards = np.array(rewards)
        done = np.array(done)

        return (obs, rewards, done, info)
