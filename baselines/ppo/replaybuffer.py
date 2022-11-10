import numpy as np
import random

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.storage = [] 
        print('cleared!')
        
    def _add_dict(self, datadict):
        if len(self.storage) >= self.capacity:
            idx = np.random.randint(len(self.storage))
            self.storage[idx] = datadict
        else:
            self.storage = [datadict] * self.capacity

    def sample(self, num = -1): # list[timestep] ((NUM_AGENTS, *obs_shape))
        if num < 0:
            return self._listdict_to_dictlist(self.storage)
        else:
            choices = [self.storage[i] for i in random.sample(range(len(self.storage)), k = num)]
            return self._listdict_to_dictlist(choices)

    def _listdict_to_dictlist(self,listdict):
        out = {}
        for key in listdict[0].keys():
            out[key] = np.stack([data[key] for data in listdict])
        return out

    def add(self, obs,reward,done, action,clear):
        obs_cur = {"cur/" + k:v for k,v in obs.items()}

        if clear: self.obs_prev = obs
        obs_prev = {"prev/" + k:v for k,v in self.obs_prev.items()}

        ddict = self._merge_dicts(obs_cur, obs_prev, {"helper_reward": np.zeros_like(reward)},{"raw_reward":reward}, {"done":done}, {"action":action})

        self._add_dict(ddict)
        self.obs_prev = obs

    def _merge_dicts(self,*args):
        acc = {}
        for d in args:
            assert isinstance(d, dict)
            acc.update(d)
        return acc
    
class MultiTaskReplayBuffer:
    def __init__(self, num_tasks, buffer_length, timestep):
        self.timestep = timestep
        self.replayBuffers = []
        for i in range(num_tasks):
            self.replayBuffers.append(ReplayBuffer(capacity= buffer_length))
    def __len__(self):
        return len(self.replayBuffers)
    def sample_batch(self):
        collate = []
        for buffer in self.replayBuffers:
            collate.append(buffer.sample(self.timestep))
        return collate
    def clear(self):
        for replay_buffer in self.replayBuffers:
            replay_buffer.clear()
