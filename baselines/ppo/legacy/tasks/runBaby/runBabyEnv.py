import numpy as np
import socket
import struct
import os
import time
import random
from PIL import Image
from tasks.runBaby.constants import *
from tasks.runBaby.VECA.environment import GeneralEnvironment
from tasks.runBaby.utils import * 
import pickle

class Environment(GeneralEnvironment):
    def __init__(self, num_envs, port = 8870):
        GeneralEnvironment.__init__(self, num_envs, port)
        self.name = 'RunBaby'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        
        self.observation_space = {
            'touch': (5 * 82 + 9888)
        }
        if TACTILE is False:
            self.observation_space['touch'] = (5 * 82)

        self.tracker = momentsTracker(self.observation_space['touch'], 0.)

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, dones, info, touchs = [], [], [], []
        poss, vels, composs, comvels = [], [], [], []
        #posLs, posRs, posOs = [], [], []
        imgs = []

        TACTILE_L = self.observation_space['touch']
        
        #print(data.keys())
        #print(self.obj_data.keys())
        for i in range(self.num_envs):
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
            pos = data['pos'][i]
            vel = data['vel'][i]
            compos = data['compos'][i]
            comvel = data['comvel'][i]

            poss.append(pos)
            vels.append(vel)
            composs.append(compos)
            comvels.append(comvel)
        poss = np.array(poss)
        vels = np.array(vels)
        composs = np.array(composs)
        comvels = np.array(comvels)
        info = (poss, vels, composs, comvels)

        touchs = np.array(touchs)
        for i in range(self.num_envs):
            self.tracker.update(touchs[i])
        touchs = self.tracker.normalize(touchs, clip = True)
        #print(np.mean(touchs), np.var(touchs), np.min(touchs), np.max(touchs))
        obs = {'touch':touchs}
        return (obs, rewards, dones, info)

    def step(self, action, ignore_agent_dim = True):
        super().send_action(action)
        return self.collect_observations(ignore_agent_dim)
