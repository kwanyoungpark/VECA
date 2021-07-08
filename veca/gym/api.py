from veca.gym.disktower import Environment as env_disktower
from veca.gym.kicktheball import Environment as env_kicktheball
from veca.gym.mazenav import Environment as env_mazenav
from veca.gym.babyrun import Environment as env_babyrun

task_env = {
        "disktower":env_disktower,
        "kicktheball": env_kicktheball,
        "mazenav": env_mazenav,
        "babyrun": env_babyrun,
        }

class Environment():
    def __init__(self, task, num_envs, ip, port, args):
        if task not in task_env.keys():
            raise ValueError("Task [" + task + "] not supported.")
        self.env = task_env[task](
                num_envs = num_envs, 
                ip = ip, port = port, 
                args = args)
    def step(self, action, ignore_agent_dim = False):
        return self.env.step(action,ignore_agent_dim)
    def reset(self, mask = None):
        return self.env.reset(mask = mask)
    def reset_connection(self):
        return self.env.reset_connection()
    def close(self):
        return self.env.close()
