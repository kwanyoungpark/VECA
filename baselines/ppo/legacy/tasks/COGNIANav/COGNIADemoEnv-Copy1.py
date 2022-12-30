import numpy as np
import socket
import struct
import os
import time
import random
from PIL import Image
#from COGNIADemo.constants import *
from VECA.environment import GeneralEnvironment
from tasks.COGNIANav.utils import *
import pickle
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
WAV_K = 10
class Environment(GeneralEnvironment):
    def __init__(self, num_envs, port = 8870):
        GeneralEnvironment.__init__(self, num_envs, port)
        self.name = 'COGNIANav'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
            'audio': (2, 66 * 13),
            'touch': 1, # Simple tactile
            'real_audio': (2, 2940),
            'real_touch': 16480
        }
        self.wav_buffer = np.zeros([WAV_K] + list(self.observation_space['real_audio']))
        if RECORD:
            self.imgsRec, self.wavsRec = [], []
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
        objimgs, objwavs, objtouchs = [], [], []
        imgs, wavs, real_audios, touchs, real_touchs = [], [], [], [], []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        WAV_C, _ = self.observation_space['audio']
        
        if RECORD:
            imgRec = np.reshape(np.array(list(reversed(data['Recimg'][i]))), self.observation_space['Recimage'])
            wavRec = np.reshape(np.array(data['Recwav'][i]), [WAV_C, -1])
            self.imgsRec.append(imgRec)
            self.wavsRec.append(wavRec)

        #print(data.keys())
        mask = np.zeros(self.num_envs, dtype = np.bool)
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            if 'wav' in data.keys():
                wav = data['wav'][i]
            else:
                wav = np.zeros(self.observation_space['real_audio'])
            touch = data['touch'][i]
            if 'real_touch' in data.keys():
                real_touch = data['real_touch'][i]
            else:
                real_touch = np.zeros([self.observation_space['real_touch']])
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            GposA, BposA = data['goodpos'][i], data['badpos'][i]
            Gpos, Bpos = list(data['goodcampos'][i]), list(data['badcampos'][i])
            Gpos[0], Gpos[1], Gpos[2] = np.tanh(Gpos[0] - 0.5), Gpos[1], np.tanh(Gpos[2])
            Bpos[0], Bpos[1], Bpos[2] = np.tanh(Bpos[0] - 0.5), Bpos[1], np.tanh(Bpos[2])
            Gobj, Bobj = str(data['goodobj'][i]), str(data['badobj'][i])
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            wav = np.reshape(np.array(wav), [WAV_C, -1]) / 32768.0
            self.wav_buffer = np.concatenate([self.wav_buffer[1:], [wav]], axis = 0)
            objimg, objwav, objtouch = data['objimg'][i], data['objwav'][i], data['objtouch'][i]
            imgs.append(img)
            wavs.append(wav2freqSTFT(self.wav_buffer))
            real_audios.append(wav)
            touchs.append(touch)
            real_touchs.append(real_touch)
            rewards.append(reward)
            GposAL.append(GposA)
            BposAL.append(BposA)
            GposL.append(Gpos)
            BposL.append(Bpos)
            Gobjs.append(Gobj)
            Bobjs.append(Bobj)
            objimgs.append(objimg)
            objwavs.append(objwav)
            objtouchs.append(objtouch)
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
        objimgs = np.reshape(np.array(objimgs), [self.num_envs, 2])
        objwavs = np.reshape(np.array(objwavs), [self.num_envs])
        objtouchs = np.reshape(np.array(objtouchs), [self.num_envs])
        #info = (posL, posAL, objs)

        imgs = np.array(imgs)
        wavs = np.array(wavs)
        touchs = np.array(touchs)
        real_touchs = np.array(real_touchs)
        obs = {'image': imgs, 'audio': wavs, 'touch': touchs, 'real_audio': real_audios, 'real_touch': real_touchs}
        infos = {'pos': GposL, 'objimg': objimgs, 'objwav': objwavs, 'objtouch': objtouchs}

        rewards = np.array(rewards)
        done = np.array(done)

        return (obs, rewards, done, infos)

    def reset(self, mask = None):
        self.wav_buffer = np.zeros_like(self.wav_buffer)
        super().reset(mask)
        return 

