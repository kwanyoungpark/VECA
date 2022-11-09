import numpy as np
import socket
import struct
import os
import time
import random
from PIL import Image
from tasks.kicktheball.utils import getRewardFromPos, wav2freq
from VECA.environment import GeneralEnvironment

class Environment(GeneralEnvironment):
    def __init__(self, num_envs, port = 8970):
        GeneralEnvironment.__init__(self, num_envs, port)
        self.name = 'KickTheBall'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
            'audio': (2, 250)
        }
    
    def step(self, action, ignore_agent_dim = True):
        super().send_action(action)
        return self.collect_observations(ignore_agent_dim)

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim)
        rewards, done, info = [], [], []
        imgs, wavs = [], []
        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        WAV_C, _ = self.observation_space['audio']
        for i in range(self.num_envs):
            #print('got observation':)
            #img = list(reversed(readBytes(data[:IMG_DATAL], 'uint8')))
            #wav = readBytes(data[IMG_DATAL:IMG_DATAL + 2 * WAV_DATAL], 'int16')
            img = list(reversed(data['img'][i]))
            wav = data['wav'][i]
            doneA = data['done'][i][0]
            #other = data['other'][i][0]
            reward = data['reward'][i][0]
            pos = data['pos'][i]
            #pos = readBytes(data[IMG_DATAL + 2 * WAV_DATAL + 6:], 'float')
            #print('pos', pos)
            #print(data[-50:])
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            wav = np.reshape(np.array(wav), [WAV_C, -1]) / 32768.0 
            wav = wav2freq(wav)
            #other = np.reshape(np.array(other).astype(np.float32), [1])
            #print(np.min(wav), np.max(wav))
            imgs.append(img)
            wavs.append(wav)
            #others.append(other)
            rewards.append(reward)
            if doneA: done.append(True)
            else: done.append(False)
            info.append(pos)
        imgs, wavs = np.array(imgs), np.array(wavs)
        obs = {'image':imgs, 'audio':wavs}
        return (obs, rewards, done, info)
