import numpy as np
import veca.gym


if __name__=="__main__":
    
    print(veca.gym.list_tasks())                        # List available VECA tasks

    num_envs = 1

    env = veca.gym.make(
        task = "mazenav",                               # VECA task name
        num_envs = num_envs,                            # Number of parallel environment instances to execute
        args = ["-train", "-timeout", "-1"],            # VECA task additional arguments
        remote_env = True,                              # Whether to use the Environment Orchestrator process at a remote server. If True, the orchestrator's ip and port should be given.               
        ip = "127.0.0.1", port= 8872,                   # ip and port of remote Envionment Orchestrator master
        )

    action_dim = env.action_space
    env.reset()
    print("Env Init")

    for i in range(100):
        action = np.random.rand(num_envs, action_dim) * 2 - 1
        obs, reward, done, infos = env.step(action)
        print("Env Step")
        if any(done):
            env.reset()
    
    env.close()
    print("Env Close")


