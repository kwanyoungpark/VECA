import numpy as np
class Curriculum:
    def __init__(self, stage1, stage2):
        self.stage1 = stage1
        self.stage2 = stage2
        self.step = 0
    def guide(self, infos,reward):
        goodpos = infos['agent/goodpos']
        badpos = infos['agent/badpos']
        N = 15
        func = lambda x : 0.01 * np.floor(N * np.exp((- 0.25) * x )) / N
        if (self.step < self.stage1):
            helper_reward = np.zeros_like(reward)
        elif self.stage1 <= self.step and self.step < self.stage2:
            distance_from_goodobj = np.linalg.norm(goodpos, axis = 1)
            helper_reward = func(distance_from_goodobj)
        else:
            distance_from_goodobj = np.linalg.norm(goodpos, axis = 1)
            distance_from_badobj = np.linalg.norm(badpos, axis = 1)
            helper_reward = func(distance_from_goodobj) - 0.3 * func(distance_from_badobj) 
        self.step += 1
        #print("Helper Reward:", helper_reward, " | Good:", np.linalg.norm(goodpos, axis = 1),  " Bad:", np.linalg.norm(badpos, axis = 1))
        return helper_reward

    def step(self):
        self.step += 1
