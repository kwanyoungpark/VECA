import numpy as np
from .replaybuffer import ReplayBuffer
from collections import defaultdict

class HeadQuarter(): #TODO Make it as a generator (DataLoader pov)
    def __init__(self, env, bufferlength, timestep):
        self.cache = defaultdict(list)

        self.replayBuffer = ReplayBuffer(capacity= bufferlength)
        self.env = env
        self.timestep = timestep
        self.restart()

    def restart(self):
        self.env.reset()
        action = np.zeros([self.env.num_agents, self.env.action_space])
        obs,reward,done,infos = self.env.step(action)
        self._add_replay(obs,reward,done,action, clear = True)

    def step(self, action):
        obs,reward,done, infos = self.env.step(self._filter_action(action))
        self._add_replay(obs,reward,done, action, clear = False)
        return obs,infos,reward,done

    def get_batch(self, num = -1):
        data_list = self.replayBuffer.get(num)
        out = {}
        for key in data_list[0].keys():
            out[key] = np.stack([data[key] for data in data_list])
        return out

    def _add_replay(self, obs,reward,done, action,clear):
        obs_cur = {"cur/" + k:v for k,v in obs.items()}

        if clear: self.obs_prev = obs
        obs_prev = {"prev/" + k:v for k,v in self.obs_prev.items()}

        ddict = self._merge_dicts(obs_cur, obs_prev, {"helper_reward": np.zeros_like(reward)},{"raw_reward":reward}, {"done":done}, {"action":action})

        self.replayBuffer.add(ddict)
        self.obs_prev = obs
    
    def _filter_action(self, action):
        action = np.tanh(action)
        action[:, 0] = action[:, 0] * 0.8
        #action[:, 0] = np.maximum(action[:, 0], 0)
        action[:, 0] = (action[:, 0] + 1) / 2
        #print(action)
        action[:, 1] = action[:, 1] * 3
        return action

    def sample(self):
        action = np.random.rand(self.env.num_agents, self.env.action_space)
        obs,reward,done,infos = self.env.step(action)
        self._add_replay(obs,reward,done,action, clear = False)
        return obs,reward,done,infos

    def _merge_dicts(self,*args):
        acc = {}
        for d in args:
            assert isinstance(d, dict)
            acc.update(d)
        return acc