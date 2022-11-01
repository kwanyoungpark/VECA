import numpy as np
import socket
import struct
import os
import time
import random
#from moviepy.editor import *
from veca.gym.core import EnvModule
#import pickle

class Environment(EnvModule):
    def __init__(self, task, num_envs,args, seeds,
            remote_env, port):
        EnvModule.__init__(self, task, num_envs, args, seeds,
            remote_env, port,
            exec_path_win = "veca\\env_manager\\bin\\disktower_multiagent\\VECAUnityApp.exe",
            download_link_win = "https://drive.google.com/uc?export=download&id=1uoBXn8AvKcc1wQpFj3vI0qYssfSjYGEb",
            exec_path_linux ="./veca/env_manager/bin/disktower_multiagent/disktowermultiagent.x86_64",
            download_link_linux = "https://drive.google.com/uc?export=download&id=1VKYVhrhliChZyOq23-9Nd2cqmrB5UwJc"
            )
        self.name = 'DiskTowerMultiAgent'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
        }
        self.record =  ("-record" in args)
        if self.record:
            self.imgsRec = []
        #self.action_space = 5

    def collect_observations(self, ignore_agent_dim = False):
        data = super().collect_observations(ignore_agent_dim = False)
        rewards, done, info, pos, grab = [], [], [], [], []
        imgs = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']

        if self.record:
            imgRec = np.reshape(np.array(list(reversed(data['Recimg'][0][0]))), (3, 224, 224)).astype(np.uint8)
            imgRec = np.transpose(imgRec, (1, 2, 0))
            self.imgsRec.append(imgRec)
            
        for i in range(self.num_envs):
            rewards_env, done_env, info_env, pos_env, grab_env = [], [], [], [], []
            imgs_env = []
            for j in range(self.agents_per_env):
                img = list(reversed(data['img'][i][j]))
                reward = data['reward'][i][j][0]
                doneA = (reward == 5.)
                posA = np.reshape(data['pos'][i][j], (6, 3))
                grabA = (np.reshape(data['grab'][i][j], 6) != 0)
                
                img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
                imgs_env.append(img)
                rewards_env.append(reward)
                done_env.append(doneA)
                pos_env.append(posA)
                grab_env.append(grabA)
            imgs.append(imgs_env)
            rewards.append(rewards_env)
            done.append(any(done_env))
            pos.append(pos_env)
            grab.append(grab_env)
                
            
        self.reset(done)
        
        imgs = np.array(imgs)
        obs = {'image': imgs}
        
        rewards = np.array(rewards)
        
        done = np.array(done)
        
        pos = np.array(pos)
        grab = np.array(grab)
        info = {'pos': pos, 'grab':grab}

        return (obs, rewards, done, info)

    def send_action(self, action):
        super().send_action(action)

    def step(self, action, ignore_agent_dim = False):
        self.send_action(action)
        return self.collect_observations(ignore_agent_dim)

    def write_record(self):
        return
        '''
        image_clip = ImageSequenceClip(self.imgsRec, fps=15) 
        print(self.imgsRec[0].shape, self.imgsRec[0].dtype, self.imgsRec[0].max())
        audios = np.concatenate(self.wavsRec, axis = 1)
        audios = (np.transpose(audios) / 32768.).astype(np.float32)
        audioclip = AudioArrayClip(audios, fps=44100)
        print(audios.shape, audios.dtype, audios.max())
        print(len(self.imgsRec) / 15, audios.shape[0] / 44100)
        video_clip = image_clip.set_audio(audioclip)
        video_clip.write_videofile("result.mp4", fps=15, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
        
        image_clip.write_videofile("result.mp4", fps=15,remove_temp=True, codec="libx264", audio_codec="aac")
        '''
