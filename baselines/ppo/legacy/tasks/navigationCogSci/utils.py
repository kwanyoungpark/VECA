import tensorflow as tf
import cv2
import numpy as np
import random
import tensorflow.contrib.slim as slim 
import math

def unzip_obs(obs, IMG_H, IMG_W, RAW_WAV_LENGTH):
    imgs, wavs = [], []
    for i in range(NUM_AGENTS):
        img, wav = obs['img'][i], obs['wav'][i]
        img = np.reshape(img, [2, IMG_H, IMG_W])
        wav = np.reshape(wav, [2, RAW_WAV_LENGTH])
        #wav = abs(np.fft.rfft(wav))[:int(MAX_FREQ/FREQ_STEP)]
        #print(np.min(wav), np.max(wav))
        wav0, wav1 = wav[0], wav[1]
        wav0 = abs(np.fft.rfft(wav0))[:250]
        wav0 = np.log10(wav0 + 1e-8)
        wav1 = abs(np.fft.rfft(wav1))[:250]
        wav1 = np.log10(wav1 + 1e-8)
        wav = np.array([wav0, wav1])
        #wav = np.reshape(wav0 - wav1, [2, 250])
        #print(np.min(wav), np.max(wav))
        #print(np.min(wav0), np.max(wav0), np.min(wav1), np.max(wav1))
        #print(wav0)
        #print(img.mean(), img.var())
        print(np.max(wav0), np.max(wav1))
        imgs.append(img), wavs.append(wav)
    obs['img'] = np.array(imgs)
    obs['wav'] = np.array(wavs)

def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    print(np.max(wav0), np.max(wav1))
    return wav

def getRewardFromFakeCamPos(pos, action, cheat = False):
    x, y, z = pos[0], pos[1], pos[2]
    #print(x, y, z)
    if cheat:
        if x < 0:
            if z > 0:
                return -0.05*action[1] + 0.03*action[0]
            else:
                return 0.05*action[1] - 0.03*action[0]
        else:
            if z > 0:
                return 0.05*action[1] + 0.03*action[0]
            else:
                return -0.05*action[1] - 0.03*action[0]
    reward = 0
    if z>0:
        if -np.tanh(0.5) < x and x < 0:
            reward -= action[1] * 0.02
        if 0 < x and x < np.tanh(0.5):
            reward += action[1] * 0.02
        if abs(x) < np.tanh(0.3):
            reward += 0.03 * action[0]
    return reward
 
def getRewardFromCamPos(pos, action, cheat = False):
    x, y, z = pos[0], pos[1], pos[2]
    #print(x, y, z)
    if cheat:
        if x < 0:
            if z > 0:
                return -0.05*action[1] + 0.03*action[0]
            else:
                return 0.05*action[1] - 0.03*action[0]
        else:
            if z > 0:
                return 0.05*action[1] + 0.03*action[0]
            else:
                return -0.05*action[1] - 0.03*action[0]
    reward = 0
    if z>0:
        if -np.tanh(0.5) < x and x < 0:
            reward -= action[1] * 0.05# * abs(x)
        if 0 < x and x < np.tanh(0.5):
            reward += action[1] * 0.05# * abs(x)
        if abs(x) < np.tanh(0.5):
            reward += 0.03 * action[0]# * (np.tanh(0.5) - abs(x))
        else:
            reward -= 0.03 * action[0]# * (np.tanh(0.5) - abs(x))
            #reward += 0.01 * abs(action[1])
    if z<0:
        reward = -0.03*action[0]# + 0.01 * abs(action[1])
        #reward += 0.01 * abs(action[1])
    return reward
    

def getRewardFromPos(pos, action):
    #wav = wav.flatten()
    #wavL, wavR = np.max(wav[:250]), np.max(wav[250:])
    #wav = wavL - wavR
    cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    sindeg = pos[0] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    '''
    if pos[0] * wav > 0:
        print("HOLYSHIT", wavL, wavR, pos)
    else:
        print("GOOD", wavL, wavR, pos)
    '''
    res = 0
    if action[1] * sindeg >= 0:
        res += abs(action[1]) * 0.03# * abs(sindeg)
    else:
        res -= abs(action[1]) * 0.03
    #res += max(0, action[0]) * (0.01 * (1 + cosdeg) * (1 + cosdeg))
    res += action[0] * (0.1 * cosdeg - 0.07)
    #print(action, sindeg, cosdeg, res)
    return res
    '''
    if cosdeg >= 0.9:
        if action[2] == 1:
            return 0.03
        else:
            return 0
    elif pos[2] < 0:
        if action[0] == 1:
            return 0.03
        else:
            return 0
    else:
        if action[1] == 1:
            return 0.03
        else:
            return 0
    '''

def getActionFromPos(pos, wav):
    wav = wav.flatten()
    wavL, wavR = np.max(wav[:250]), np.max(wav[250:])
    wav = wavL - wavR
    cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    if cosdeg > 0.8:
        return np.array([0, 0, 1])
    elif wav > 0:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])
