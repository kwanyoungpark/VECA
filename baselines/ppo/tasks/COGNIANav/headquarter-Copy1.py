import numpy as np
from tasks.COGNIANav.replaybuffer import ReplayBuffer

class HeadQuarter():
    def __init__(self, env, timestep):
        self.cache = defaultdict(list)

        self.replayBuffer = ReplayBuffer(BUFFER_LENGTH = BUFFER_LENGTH, env = env)
        self.env = env
        self.timestep = timestep
        self.restart()

    def restart(self):
        self.env.reset()
        obs, rewards, done, infos = self.env.step(np.zeros([self.env.num_agents, self.env.action_space]))
        ddict = merge_dicts(obs, infos)
        for k, v in ddict.items():
            self.cache[k] =[v] * self.timestep
    
    def add_replay_buf(self, datadict):
        self.replayBuffer.add(datadict)
    
    def filter_action(self, action):
        action = np.tanh(self.action.copy())
        action[:, 0] = action[:, 0] * 0.8
        #action[:, 0] = np.maximum(action[:, 0], 0)
        action[:, 0] = (action[:, 0] + 1) / 2
        #print(action)
        action[:, 1] = action[:, 1] * 3
        return action

    def merge_dicts(self,x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z
    def step(self, action):
        obs, reward, done, infos = self.env.step(self.filter_action(action))
        ddict = merge_dicts(obs, infos)
        for k, v in ddict.items():
            self.cache[k] = self.cache[k][1:] + [v]
        self.add_replay_buf(self.cache)
        return done
