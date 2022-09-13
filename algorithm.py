import numpy as np
import veca.gym
import argparse

cfg_default = {
        "env_manager_ip" : "127.0.0.1",
        "env_manager_port" : 8872,
        "num_envs" : 1,
        "task":"disktower",
        "optional_args" : ["-train", "-timeout", "-1", "--numagents", "1"],
        #"optional_args" : ["-train", "-timeout", "-1", "-record"] # creates recorded video file on env.close()
}


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='VECA Algorithm Server')
    parser.add_argument('--task', type=str, 
                        default=cfg_default["task"], help='VECA Task')
    parser.add_argument('--ip', type=str, 
                        default=cfg_default["env_manager_ip"], help='Envionment Manager machine\'s ip')
    parser.add_argument('--port', type=int, metavar="ENV_PORT", 
                        default = cfg_default["env_manager_port"], help='Environment Manager\'s port')
    parser.add_argument('--num_envs', type=int, 
                        default = cfg_default["num_envs"], help='Number of parallel environments to execute')
    args = parser.parse_args()
    args.optional_args = cfg_default["optional_args"]
    
    
    print(veca.gym.list_tasks())
    env = veca.gym.make(task = args.task, ip = args.ip, port=args.port, 
            num_envs = args.num_envs, args = args.optional_args)
    action_dim = env.action_space
    env.reset()
    print("Env Init")

    for i in range(100):
        action = np.random.rand(args.num_envs, action_dim) * 2 - 1
        obs, reward, done, infos = env.step(action)
        print("Env Step")
        if any(done):
            env.reset()
    
    env.close()
    print("Env Close")


