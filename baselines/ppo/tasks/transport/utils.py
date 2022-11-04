import tensorflow as tf
import cv2
import numpy as np
import random
import tensorflow.contrib.slim as slim 
import math

def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    print(np.max(wav0), np.max(wav1))
    return wav

def getRewardFromPos(posHL0, posHR0, posO0, posHL1, posHR1, posO1):
    distL0, distR0 = np.linalg.norm(posHL0 - posO0), np.linalg.norm(posHR0 - posO0)
    distL1, distR1 = np.linalg.norm(posHL1 - posO1), np.linalg.norm(posHR1 - posO1)
    dist = (distR1 - distR0) + (distL1 - distL0)
    return -dist
