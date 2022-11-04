import numpy as np
from collections import defaultdict

class ReplayBuffer():
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.storage = defaultdict(list)
        self.size = 0
        print('cleared!')
        
    def add(self, datadict):
        for k, v in datadict.items():
            if self.size >= self.capacity:
                idx = np.random.randint(self.size)
                self.storage[k] = v
            else:
                self.storage[k].append(v)
                self.size += 1
       

    def get(self, num = -1):
        if num < 0:
            return self.storage
        else:
            return {self.storage[key][np.random.choice(self.size, num)] for key in self.storage.keys()}
