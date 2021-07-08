import tensorflow as tf
from tasks.COGNIANav.utils import *
#from model import Model
#from PPOmodel.easy_model import Model
import numpy as np
import sys
import os
import cv2 
import time
import copy
from PIL import Image
from constants import BUFFER_LENGTH, RNN, HANDCRAFT
from tasks.COGNIANav.constants import FRAME_SKIP
from tasks.COGNIANav.replaybuffer import ReplayBuffer
import pickle

class HeadQuarter():
    def __init__(self, env, model):
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = env.observation_space['image']
        WAV_CHANNEL, WAV_LENGTH = env.observation_space['audio']
        _, RAW_WAV_LENGTH = env.observation_space['real_audio']
        NUM_TOUCH = env.observation_space['real_touch']

        NUM_AGENTS = env.num_envs
        ACTION_LENGTH = env.action_space
        print(ACTION_LENGTH)

        self.img0 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.img1 = np.zeros([NUM_AGENTS, NUM_TIME, IMG_CHANNEL, IMG_H, IMG_W])
        self.wav0 = np.zeros([NUM_AGENTS, NUM_TIME, WAV_CHANNEL, WAV_LENGTH])
        self.wav1 = np.zeros([NUM_AGENTS, NUM_TIME, WAV_CHANNEL, WAV_LENGTH])
        self.Rwav0 = np.zeros([NUM_AGENTS, NUM_TIME, WAV_CHANNEL, RAW_WAV_LENGTH])
        self.Rwav1 = np.zeros([NUM_AGENTS, NUM_TIME, WAV_CHANNEL, RAW_WAV_LENGTH])
        self.touch0 = np.zeros([NUM_AGENTS, NUM_TIME, 1])
        self.touch1 = np.zeros([NUM_AGENTS, NUM_TIME, 1])
        self.Rtouch0 = np.zeros([NUM_AGENTS, NUM_TIME, NUM_TOUCH])
        self.Rtouch1 = np.zeros([NUM_AGENTS, NUM_TIME, NUM_TOUCH])
        self.pos0 = np.zeros([NUM_AGENTS, 2, 3])
        self.pos1 = np.zeros([NUM_AGENTS, 2, 3])
        self.action = np.zeros([NUM_AGENTS, ACTION_LENGTH])
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        self.done = np.zeros([NUM_AGENTS], dtype=bool)
        self.objimg = np.zeros([NUM_AGENTS, 2])
        self.objwav = np.zeros([NUM_AGENTS, 3])
        self.objtouch = np.zeros([NUM_AGENTS])

        if RNN:
            from constants import STATE_LENGTH
            self.state0 = np.zeros([NUM_AGENTS, NUM_TIME, STATE_LENGTH])
            self.state1 = np.zeros([NUM_AGENTS, NUM_TEIM, STATE_LENGTH])

        self.replayBuffer = ReplayBuffer(BUFFER_LENGTH = BUFFER_LENGTH, env = env)
        self.env = env
        self.model = model
        self.restart()

    def restart(self):
        NUM_TIME = 1
        NUM_AGENTS, ACTION_LENGTH = self.env.num_agents, self.env.action_space
        self.env.reset()
        obs, rewards, done, infos = self.env.step(np.zeros([NUM_AGENTS, ACTION_LENGTH]))
        imgs, wavs, touchs, real_wavs, real_touchs = obs['image'], obs['audio'], obs['touch'], obs['real_audio'], obs['real_touch']
        self.objimg, self.objwav, self.objtouch = infos['objimg'], infos['objwav'], infos['objtouch']
        #print(others.shape)
        print('first img', imgs.mean())
        for i in range(NUM_AGENTS):
            img, wav, touch, Rwav, Rtouch = imgs[i], wavs[i], touchs[i], real_wavs[i], real_touchs[i]
            self.img1[i] = np.array([img] * NUM_TIME)
            self.wav1[i] = np.array([wav] * NUM_TIME)
            self.Rwav1[i] = np.array([Rwav] * NUM_TIME)
            self.touch1[i] = np.array([touch] * NUM_TIME)
            self.Rtouch1[i] = np.array([Rtouch] * NUM_TIME)
            if RNN:
                self.state1[i] = np.zeros([NUM_TIME, STATE_LENGTH])
            self.raw_reward[i] = 0
            self.helper_reward[i] = 0
        self.pos1 = infos['pos']
    
    def add_replay_buf(self, buf):
        if RNN:
            buf.add_replay(self.img0, self.img1, self.wav0, self.wav1, self.Rwav0, self.Rwav1, self.touch0, self.touch1, self.Rtouch0, self.Rtouch1, self.action, self.helper_reward, self.raw_reward, self.pos0, self.pos1, self.done, self.objimg, self.objwav, self.objtouch, self.state0, self.state1)
        else:
            buf.add_replay(self.img0, self.img1, self.wav0, self.wav1, self.Rwav0, self.Rwav1, self.touch0, self.touch1, self.Rtouch0, self.Rtouch1, self.action, self.helper_reward, self.raw_reward, self.pos0, self.pos1, self.done, self.objimg, self.objwav, self.objtouch)

    def add_replay(self):
        self.add_replay_buf(self.replayBuffer)
    
    def send_action(self):
        data = {'image':self.img1, 'audio':self.wav1, 'touch': self.touch1, 'obj': self.objwav}
        if RNN:
            data.update({'state': self.state0.copy()})
            self.state0 = self.state1.copy()
        if self.env.mode == 'DISC':
            if RNN:
                _, self.action, self.state1 = self.model.get_action(data)
            else:
                _, self.action = self.model.get_action(data)
        else:
            if RNN:
               _,  _, self.action, self.state1 = self.model.get_action(data)
            else:
                _, _, self.action = self.model.get_action(data)

        NUM_AGENTS = self.env.num_agents

        if self.env.mode == 'DISC':
            action = np.zeros([NUM_AGENTS, 2])
            for i in range(NUM_AGENTS):
                if self.action[i][0] == 1:
                    action[i][0] = 1
                if self.action[i][1] == 1:
                    action[i][1] = -1
                if self.action[i][2] == 1:
                    action[i][1] = 1
        else:
            action = np.tanh(self.action.copy())
            #action[:, 0] = action[:, 0] * 0.8
            #action[:, 0] = np.maximum(action[:, 0], 0)
            action[:, 0] = (action[:, 0] + 1) / 2
            action[:, 0] = action[:, 0] * 3
            #print(action)

        if HANDCRAFT:
            for i in range(NUM_AGENTS):
                action[i] = getActionFromObj(self.pos0[i], self.objimg[i], self.objwav[i])
                print("action", action[i])
        action[:, 1] = action[:, 1] * 3
        '''
        for i in range(NUM_AGENTS):
            action[i] = get_action_from_imgwav(self.objimg[i], self.objwav[i])
        action += np.reshape(np.random.normal(0, 0.2, 2*NUM_AGENTS), [NUM_AGENTS, 2])
        '''
        #print(self.objimg, self.objwav, action)
        self.env.send_action(action)

    def collect_observations(self, add_replay = True):
        NUM_AGENTS = self.env.num_envs
        NUM_TIME = 1
        IMG_CHANNEL, IMG_H, IMG_W = self.env.observation_space['image']
        WAV_CHANNEL, WAV_LENGTH = self.env.observation_space['audio']
        NUM_TOUCH = self.env.observation_space['touch']
        ACTION_LENGTH = self.env.action_space

        #print(self.pos0)
        #self.myu, self.sigma, self.action = self.model.get_action(self.pos1)
        self.img0 = self.img1.copy()
        self.wav0 = self.wav1.copy()
        self.Rwav0 = self.Rwav1.copy()
        self.touch0 = self.touch1.copy()
        self.Rtouch0 = self.Rtouch1.copy()
        self.pos0 = self.pos1.copy()
        self.helper_reward = np.zeros([NUM_AGENTS])
        self.raw_reward = np.zeros([NUM_AGENTS])
        imgs1 = np.zeros([NUM_AGENTS, IMG_CHANNEL, IMG_H, IMG_W])
        wavs1 = np.zeros([NUM_AGENTS, WAV_CHANNEL, WAV_LENGTH])
        
        obs, rewards, done, infos = self.env.collect_observations()
        imgs, wavs, touchs, Rwavs, Rtouchs = obs['image'], obs['audio'], obs['touch'], obs['real_audio'], obs['real_touch']
        imgs1 = imgs.copy()
        wavs1 = wavs.copy()
        Rwavs1 = Rwavs.copy()
        touchs1 = touchs.copy()
        Rtouchs1 = Rtouchs.copy()
        self.pos1, self.objimg, self.objwav, self.objtactile = infos['pos'], infos['objimg'], infos['objwav'], infos['objtouch']
        #print("Reward : ")
        if self.env.mode == 'DISC':
            action = np.zeros([NUM_AGENTS, 2])
            for i in range(NUM_AGENTS):
                if self.action[i][0] == 1:
                    action[i][0] = 1
                if self.action[i][1] == 1:
                    action[i][1] = -1
                if self.action[i][2] == 1:
                    action[i][1] = 1
        else:
            action = np.tanh(self.action.copy())
            action[:, 0] = action[:, 0] * 0.8
            #action[:, 0] = np.maximum(action[:, 0], 0)
            action[:, 0] = (action[:, 0] + 1) / 2
            #print(action)
        action[:, 1] = action[:, 1] * 3
                
        #print(self.objwav)
        for j in range(NUM_AGENTS):
            #self.helper_reward[j] = 0.01*(np.linalg.norm(self.posA0[j]) - np.linalg.norm(self.posA1[j]) - 0.05)
            #if self.helper_reward[j] < -0.01:
            #    self.helper_reward[j] = 0
            #self.helper_reward[j] += getRewardFromPos(self.pos0[j], action[j], self.objwav[j])
            self.helper_reward[j] += getRewardFromAction(self.action[j], self.objimg[j], self.objwav[j])
            #self.helper_reward[j] -= 0.3*getRewardFromFakeCamPos(self.posF0[j], action[j], cheat = False)
            self.raw_reward[j] += rewards[j]
            self.done[j] = done[j]
            #print(self.helper_reward[j], self.raw_reward[j])

        for i in range(NUM_AGENTS):
            #print(self.img[i].mean(), self.img[i].var())
            self.img1[i] = np.append(self.img1[i][1:], [imgs1[i]], axis=0)
            self.wav1[i] = np.append(self.wav1[i][1:], [wavs1[i]], axis=0)
            self.Rwav1[i] = np.append(self.Rwav1[i][1:], [Rwavs1[i]], axis = 0)
            self.touch1[i] = np.append(self.touch1[i][1:], [touchs1[i]], axis = 0)
            self.Rtouch1[i] = np.append(self.Rtouch1[i][1:], [Rtouchs1[i]], axis = 0)
    
        if add_replay:
            self.add_replay()
        return done

    def step(self, add_replay = True):
        self.send_action()
        return self.collect_observations(add_replay)
