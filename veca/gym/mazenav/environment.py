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
    def __init__(self,task, num_envs,ip,port,  args):
        EnvModule.__init__(self, task, num_envs,ip,port, args)
        self.name = 'MazeNav'
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
        #self.action_length = self.action_space - 1
        self.prev_dist = np.zeros(num_envs)
        #self.action_space = self.action_space - 1
        '''
        if VEC_OBJ == False:
            with open('navigation/data/data.pkl', 'rb') as F:
                self.obj_data = pickle.load(F)
        '''

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, done, info, objs = [], [], [], []
        posAL, posL, dist, Hrewards = [], [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        
        print(data.keys())
        #print(self.obj_data.keys())
        mask = np.zeros(self.num_envs, dtype = np.bool)
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            posA = data['pos'][i]
            pos = list(data['campos'][i])
            dis = data['dist'][i][0]
            pos[0], pos[1], pos[2] = np.tanh(pos[0] - 0.5), pos[1], np.tanh(pos[2])
            self.stepnum[i] = data['step'][i][0]
            #posf = list(data['fakecampos'][i])
            #posf[0], posf[1], posf[2] = np.tanh(posf[0] - 0.5), posf[1], np.tanh(posf[2])
            obj = str(data['obj'][i])
            if VEC_OBJ:
                if obj is None or obj == 'None':
                    objs.append(obj)
                else:
                    obj_oh = np.zeros([NUM_OBJS])
                    obj_oh[OBJ_NAME.index(obj)] = 1.
                    objs.append(obj_oh)
            else:
                objs.append(np.reshape(self.obj_data[obj], [NUM_DEGS, IMG_H, IMG_W]))
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            imgs.append(img)
            rewards.append(reward)
            #posF.append(posf)
            posAL.append(posA)
            posL.append(pos)
            dist.append(dis)
            #print('reward')
            #print(reward)
            if doneA:
                Hrewards.append(0.)
                #print("done")
                #print(self.episode[i]) 
                self.episode[i] += 1
                if self.episode[i] == 5:
                    #print("restart")
                    done.append(True)
                    self.episode[i] = 0
                    mask[i] = True
            else:
                Hrewards.append(-0.01 * (dis - self.prev_dist[i]))
                done.append(False)
            self.prev_dist[i] = dis
        self.reset(mask)
        #info = (data['pos'], data['campos'])
        posAL = np.array(posAL)
        posL = np.array(posL)
        dist = np.array(dist)
        Hrewards = np.array(Hrewards)
        #posF = np.array(posF)
        #print(np.linalg.norm(posAL, axis = 1))
        #info = (posL, posAL, posF)
        info = (posL, posAL, dist, Hrewards)
        imgs = np.array(imgs)
        obs = {'image': imgs, 'obj': objs}
        return (obs, rewards, done, info)

    def send_action(self, action):
        
        AR = np.zeros((self.num_envs, 1))
        for i in range(self.num_envs):
            if self.stepnum[i] == 256:
                AR[i][0] = 1
        action[:,-1] = AR #np.concatenate([action, AR], axis = 1)
        super().send_action(action)
