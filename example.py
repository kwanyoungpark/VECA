import numpy as np
import veca.gym
import random, time
import matplotlib.pyplot as plt
import os, subprocess

if __name__=="__main__":
    
    print(veca.gym.list_tasks())                        # List available VECA tasks

    num_envs = 1

    env = veca.gym.make(
        task = "kicktheballrandomscene",                                 # VECA task name
        num_envs = num_envs,                                # Number of parallel environment instances to execute
        args = ["--train"],                   # VECA task additional arguments. Append "--help" to list valid arguments.
        seeds = random.sample(range(0, 2000), num_envs),    # seeds per env instances
        remote_env = False                                  # Whether to use the Environment Orchestrator process at a remote server.
        )

    action_dim = env.action_space
    env.reset()
    print("Env Init")

    for i in range(100):
        action = np.random.rand(num_envs, action_dim) * 2 - 1
        obs, reward, done, infos = env.step(action)
        print("Env infos:", infos.keys(), obs.keys() )
        if any(done):
            env.reset()
    
    env.close()
    print("Env Close")


