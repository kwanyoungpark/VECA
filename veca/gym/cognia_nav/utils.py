import cv2
import numpy as np
import random
import math
import scipy
import scipy.signal
import python_speech_features as sfeat

def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    return wav

def wav2freqSTFT(wav):
    wav0, wav1 = wav[:, 0, :], wav[:, 1, :]
    wav0, wav1 = wav0.flatten(), wav1.flatten()
    #print(wav0.shape, wav1.shape)
    #wav0, wav1 = librosa.resample(wav0, 44100, 16000), librosa.resample(wav0, 44100, 16000)
    wav0 = scipy.signal.resample(wav0, int(wav0.shape[0]*16000/44100))
    #print(wav0.shape, wav0.dtype)
    #_, _, wav0 = np.abs(scipy.signal.stft(wav0, fs=16000))
    wav0 = sfeat.mfcc(wav0) / 36.
    #print(wav0.shape, wav0.dtype)
    #wav0 = np.log10(wav0 + 1e-4)
    wav1 = scipy.signal.resample(wav1, int(wav1.shape[0]*16000/44100))
    #_, _, wav1 = np.abs(scipy.signal.stft(wav1, fs=16000))
    wav1 = sfeat.mfcc(wav1) / 36.
    #wav1 = np.log10(wav1 + 1e-4)
    #print(wav0.shape, wav1.shape, wav0.min(), wav0.max(), wav1.min(), wav1.max())
    wav0, wav1 = wav0.flatten(), wav1.flatten()
    wav = np.array([wav0, wav1])
    return wav 

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
    
def getRewardFromPos(pos, action, objwav):
    pos = pos[0]
    #print(objwav)
    objwav = objwav[0] - objwav[1]
    #print(objwav)
    cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    sindeg = pos[0] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    res = 0
    if action[1] * sindeg >= 0:
        res += abs(action[1]) * 0.03# * abs(sindeg)
    else:
        res -= abs(action[1]) * 0.03
    #res += max(0, action[0]) * (0.01 * (1 + cosdeg) * (1 + cosdeg))
    res += action[0] * (0.1 * cosdeg - 0.07)
    res += 0.03 * objwav * action[0]
    #print(action, sindeg, cosdeg, res)
    #print("Reward", res)
    return 0.5 * res

def getRewardFromAction(action, objimg, objwav):
    reward = 0.
    if objwav[1] == 1:
        reward += 0.1 * (action[1] - action[0])
    if objimg[0] == 1 or objimg[1] == 1:
        reward += 0.3 * (action[0] - abs(action[1]))
    else:
        reward += 0.03 * (action[1] - action[0])
    return reward 

def getActionFromObj(pos, objimg, objwav):
    print(objimg, objwav, pos)
    if objwav[1] == 1:
        return np.array([0, 1])
    elif objimg[0] == 1 or objimg[1] == 1:
        if objimg[0] == 1:
            pos = pos[0]
        else:
            pos = pos[1]
        sindeg = pos[0] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2]) 
        if abs(sindeg) < 0.2:
            return np.array([1, 0])
        elif sindeg < 0.2:
            return np.array([1, -1])
        else:
            return np.array([1, 1])
    else:
        return np.array([0, 1])

