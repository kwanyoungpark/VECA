#import cv2
import numpy as np
import random
import math

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
    cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    sindeg = pos[0] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    res = 0
    if action[1] * sindeg >= 0:
        res += abs(action[1]) * 0.03# * abs(sindeg)
    else:
        res -= abs(action[1]) * 0.03
    #res += max(0, action[0]) * (0.01 * (1 + cosdeg) * (1 + cosdeg))
    res += action[0] * (0.1 * cosdeg - 0.07)
    #print(action, sindeg, cosdeg, res)
    return res
