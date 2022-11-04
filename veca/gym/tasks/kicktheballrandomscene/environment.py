import numpy as np
from veca.gym.core import EnvModule

IMG_H, IMG_W = 84, 84

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
    def __init__(self, task, num_envs, args, seeds,
            remote_env, port
            ):
        EnvModule.__init__(self, task, num_envs, args, seeds,
            remote_env, port,
            exec_path_win = "veca\\env_manager\\bin\\kicktheballrandomscene\\VECAUnityApp.exe",
            download_link_win = "https://drive.google.com/uc?export=download&id=1AoS5QsltcZmzLBxuIdKVuqQ4RpfI6FuE",
            exec_path_linux = "./veca/env_manager/bin/kicktheballrandomscene/kicktheballrandomscene.x86_64",
            download_link_linux = "https://drive.google.com/uc?export=download&id=1wkeyp9krs0Y6BM9E7bxzjWMR0vm9cwOJ"
            )
        self.num_envs = num_envs
    
    def step(self, action):
        data, rewards, done = super().step(action)
        obs = {}
        for key in ["agent/img","agent/wav", "agent/tactile"]:
            if key in data:
                obs[key] = data.pop(key)
        return (obs, rewards, done, data)


