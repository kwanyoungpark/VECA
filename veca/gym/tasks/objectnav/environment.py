import numpy as np
from veca.gym.core import EnvModule

USE_MENTOR_AUDIO = False
OBJ_NAME = sorted(["toybear", "spoon", "fork", "ball", "cap", "boat", "bus", "cup", "boots", "glove"])
COLOR_NAME = ["Red", "Green", "Blue"]


def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    #print(np.max(wav0), np.max(wav1))
    return wav

class Environment(EnvModule):
    def __init__(self, task, num_envs,  args, seeds,
            remote_env, port
        ):
        EnvModule.__init__(self,task, num_envs, args, seeds,
            remote_env, port,
            exec_path_win = "veca\\env_manager\\bin\\objectnav\\VECAUnityApp.exe",
            download_link_win = "https://drive.google.com/uc?export=download&id=15sMnrX4WLib4EZv_1QJAeQorERc6853j",
            exec_path_linux = "./veca/env_manager/bin/objectnav/objectnav.x86_64",
            download_link_linux = "https://drive.google.com/uc?export=download&id=1dhUCy8xDJBwQvZfaU0j_F1LWOaIjvQa0" 
            )
    
    def step(self, action):
        data, rewards, done = super().step(action)
        obs = {}
        for key in ["agent/img","agent/wav", "agent/targetobj", "agent/targetcolor"]:
            if key in data:
                obs[key] = data.pop(key)
        return (obs, rewards, done, data)
