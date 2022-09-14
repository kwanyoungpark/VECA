import numpy as np
import socket
import struct
import os
import time
import random
#from PIL import Image
#from moviepy.editor import *
from veca.gym.core import EnvModule
import pickle

class Environment(EnvModule):
    '''
    Output of step(action) function:
    
    image: (NUM_ENVS, 6, 84, 84) np.float32     Channel : RGBRGB (Left/Right eye respectively)
    pos  : (NUM_ENVS, 6, 3)      np.float32     LR/LB/MR/MB/SR/SB(SizeColor) disk 3d position in agent coordinate 
    grab : (NUM_ENVS, 6)         np.bool        True if LR/LB/MR/MB/SR/SB(SizeColor) disk is grabbed by the agent
    done : (NUM_ENVS)            np.bool        True if the episode is over
    reward : (NUM_ENVS)          np.float32     0~5, calculated as number of successfully stacked disks
    '''
    def __init__(self, task, num_envs, ip, port, args):
        EnvModule.__init__(self, task, num_envs, ip, port, args,
            exec_path_win = "veca\\env_manager\\bin\\disktower\\VECA_latest.exe",
            download_link_win = "https://drive.google.com/uc?export=download&id=1jf4aWG9BR20HVj4sNArTEzqbK6SpSS6P",
            exec_path_linux = "./veca/env_manager/bin/disktower/disktower.x86_64",
            download_link_linux = "https://drive.google.com/uc?export=download&id=1xJ1jjqX9MkoiM0o3_00_LNlFpzyIzjXg" 
            )
        self.name = 'DiskTower'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
        }
        self.record = ("-record" in args)
        if self.record:
            self.imgsRec = []
        #self.action_space = 5
        
    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, done, info, pos, grab = [], [], [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']

        if self.record:
            imgRec = np.reshape(np.array(list(reversed(data['Recimg'][0]))), (3, 224, 224)).astype(np.uint8)
            imgRec = np.transpose(imgRec, (1, 2, 0))
            self.imgsRec.append(imgRec)
            
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            reward = data['reward'][i][0]
            doneA = (reward == 5.)
            posA = np.reshape(data['pos'][i], (6, 3))
            grabA = (np.reshape(data['grab'][i], 6) != 0)
            
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            imgs.append(img)
            rewards.append(reward)
            done.append(doneA)
            pos.append(posA)
            grab.append(grabA)
                
            
        self.reset(done)
        
        imgs = np.array(imgs)
        obs = {'image': imgs}
        
        rewards = np.array(rewards)
        
        done = np.array(done)
        
        pos = np.array(pos)
        grab = np.array(grab)
        info = {'pos': pos, 'grab':grab}

        return (obs, rewards, done, info)

    def write_record(self):
        #image_clip = ImageSequenceClip(self.imgsRec, fps=15) 
        print(self.imgsRec[0].shape, self.imgsRec[0].dtype, self.imgsRec[0].max())
        '''
        audios = np.concatenate(self.wavsRec, axis = 1)
        audios = (np.transpose(audios) / 32768.).astype(np.float32)
        audioclip = AudioArrayClip(audios, fps=44100)
        print(audios.shape, audios.dtype, audios.max())
        print(len(self.imgsRec) / 15, audios.shape[0] / 44100)
        video_clip = image_clip.set_audio(audioclip)
        video_clip.write_videofile("result.mp4", fps=15, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
        '''
        
        #image_clip.write_videofile("result.mp4", fps=15,remove_temp=True, codec="libx264", audio_codec="aac")
