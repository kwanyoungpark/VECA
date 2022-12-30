import numpy as np
from collections import defaultdict

class EnvDataLoader(): #TODO Make it as a generator (DataLoader pov)
    def __init__(self, env, curriculum):
        self.cache = defaultdict(list)
        self.env = env
        #self.reset()
        self.curriculum = curriculum
    def _preprocess(self, obs, infos):
        if "agent/img" in obs:
            obs["agent/img"] = obs["agent/img"] / 128
        obs["agent/goodobj"] = infos['agent/goodpos']
        obs["agent/badobj"] = infos['agent/badpos']

    def reset(self, replay_buffer):
        self.env.reset()
        action = np.zeros([self.env.num_agents, self.env.action_space])
        obs,reward,done,infos = self.env.step(action)
        self._preprocess(obs,infos)
        reward = {"helper_reward": self.curriculum.guide(infos, reward), "raw_reward": reward }
        replay_buffer.add(obs,reward,done,action, clear = True)

    def step(self, action, replay_buffer):
        obs,reward,done, infos = self.env.step(self._filter_action(action))
        self._preprocess(obs,infos)
        reward = {"helper_reward": self.curriculum.guide(infos, reward), "raw_reward": reward }
        replay_buffer.add(obs,reward,done, action, clear = False)
        return obs, reward, done, infos
        
    def sample(self, replay_buffer):
        action = np.random.rand(self.env.num_agents, self.env.action_space)
        obs,reward,done,infos = self.env.step(action)
        self._preprocess(obs,infos)
        reward = {"helper_reward": self.curriculum.guide(infos, reward), "raw_reward": reward }
        replay_buffer.add(obs,reward,done,action, clear = False)
        return obs,reward,done,infos
    
    def _filter_action(self, action):
        action = np.tanh(action)
        action[:, 0] = action[:, 0] * 0.8
        #action[:, 0] = np.maximum(action[:, 0], 0)
        action[:, 0] = (action[:, 0] + 1) / 2
        #print(action)
        action[:, 1] = action[:, 1] * 3
        return action

class MultiTaskDataLoader:
    def __init__(self, envs, curriculum):
        self.heads = []
        for env in envs:
            self.heads.append(EnvDataLoader(env = env, curriculum = curriculum))

    def step(self, actions,replay_buffers):
        assert len(actions) == len(self.heads)
        assert len(replay_buffers) == len(self.heads)
        collate = []
        for action, head, replay_buffer in zip(actions,self.heads, replay_buffers.replayBuffers):
            obs, reward, done, infos = head.step(action,replay_buffer)
            collate.append((obs,reward, done, infos))
        return tuple(zip(*collate))

    def sample(self, replay_buffers):
        assert len(replay_buffers) == len(self.heads)
        collate = []
        for head, replay_buffer in zip(self.heads, replay_buffers.replayBuffers):
            obs, reward, done, infos = head.sample(replay_buffer)
            collate.append((obs,reward, done, infos))
        return tuple(zip(*collate))

    def reset(self, replay_buffers):
        assert len(replay_buffers) == len(self.heads)
        for head, replay_buffer in zip(self.heads, replay_buffers.replayBuffers):
            head.reset(replay_buffer)
