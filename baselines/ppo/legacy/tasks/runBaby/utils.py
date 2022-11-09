import tensorflow as tf
import cv2
import numpy as np
import random
import tensorflow.contrib.slim as slim 
import math

class momentsTracker():
    def __init__(self, num_dims, GAMMA):
        self.mean = np.zeros(num_dims)
        self.N = 0
        self.SSE = np.zeros(num_dims)
        self.X0 = None
        self.GAMMA = GAMMA
    
    def update(self, x0):
        if self.X0 is None:
            self.X0 = x0
        else:
            self.X0 = self.X0 * self.GAMMA + x0 * (1 - self.GAMMA)
        self.N += 1
        error = self.X0 - self.mean
        self.mean += (error / self.N)
        self.SSE += error * (self.X0 - self.mean)

    def moments(self):
        return self.mean, np.sqrt(self.SSE / self.N)

    def normalize(self, x, clip = False):
        mean, std = self.moments()
        res = (x - mean) / (std + 1e-8)
        if clip:
            res = np.clip(res, -3, 3)
        return res
