import numpy as np
from tasks.COGNIANav.replaybuffer import ReplayBuffer
from collections import defaultdict

class HeadQuarter():
    def __init__(self, env, bufferlength, timestep):
        self.cache = defaultdict(list)

        self.replayBuffer = ReplayBuffer(BUFFER_LENGTH = bufferlength, env = env)
        self.env = env
        self.timestep = timestep
        self.restart()

    def restart(self):
        self.env.reset()
        obs,infos,reward,done = self.env.step(np.zeros([self.env.num_agents, self.env.action_space]))
        self.add_replay(obs,reward,done)
    
    def add_replay(self, obs,reward,done):
        ddict = self.merge_dicts(obs, {"reward":reward}, {"done":done})
        for k, v in ddict.items():
            if k not in self.cache:
                self.cache[k] =[v] * self.timestep
            else:
                self.cache[k] = self.cache[k][1:] + [v]
        self.replayBuffer.add(self.cache)
    
    def filter_action(self, action):
        action = np.tanh(self.action.copy())
        action[:, 0] = action[:, 0] * 0.8
        #action[:, 0] = np.maximum(action[:, 0], 0)
        action[:, 0] = (action[:, 0] + 1) / 2
        #print(action)
        action[:, 1] = action[:, 1] * 3
        return action

    def merge_dicts(self,*args):
        acc = {}
        for d in args:
            assert isinstance(d, dict)
            acc.update(d)
        return acc

    def step(self, action):
        obs,infos,reward,done = self.env.step(self.filter_action(action))
        self.add_replay(obs,reward,done)
        return done
