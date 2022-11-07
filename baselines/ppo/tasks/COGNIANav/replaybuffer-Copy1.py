import numpy as np
import random

class ReplayBuffer():
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.storage = [] 
        self.size = 0
        print('cleared!')
        
    def add(self, datadict):
        if len(self.storage) >= self.capacity:
            idx = np.random.randint(len(self.storage))
            self.storage[idx] = datadict
        else:
            self.storage.append(datadict)
       

    def get(self, num = -1): # list[timestep] ((NUM_AGENTS, *obs_shape))
        if num < 0:
            return self.storage
        else:
            return [self.storage[i] for i in random.sample(range(self.size), k = num)]
