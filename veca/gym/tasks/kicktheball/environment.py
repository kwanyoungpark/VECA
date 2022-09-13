import numpy as np
from veca.gym.core.environment import EnvModule

IMG_H, IMG_W = 84, 84

def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    #print(np.max(wav0), np.max(wav1))
    return wav

class Environment(EnvModule):
    def __init__(self, task, num_envs, ip,port,args):
        EnvModule.__init__(self, task, num_envs, ip,port,args,
            exec_path = "./veca/env_manager/bin/kicktheball/VECA-BS.exe",
            download_link = "https://drive.google.com/uc?export=download&id=1Qq9SuDMB_0yim05mB_fJwDHriT4amAQZ")
        self.num_envs = num_envs
    
    def step(self, action):
        data = super().step(action)
        rewards, done, info = [], [], []
        imgs, wavs = [], []
        for i in range(self.num_envs):
            img = list(reversed(data['img'][i]))
            wav = data['wav'][i]
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            pos = data['pos'][i]
            img = np.reshape(np.array(img), [6, IMG_H, IMG_W]) / 255.0
            wav = np.reshape(np.array(wav), [2, -1]) / 32768.0 
            wav = wav2freq(wav)
            imgs.append(img)
            wavs.append(wav)
            rewards.append(reward)
            if doneA: done.append(True)
            else: done.append(False)
            info.append(pos)
        imgs, wavs = np.array(imgs), np.array(wavs)
        obs = (imgs, wavs)
        return (obs, rewards, done, info)


