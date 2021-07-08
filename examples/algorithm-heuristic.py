import numpy as np
from veca.gym.disktower import Environment
import argparse

cfg_default = {
        "env_manager_ip" : "127.0.0.1",
        "num_envs" : 1,
        "env_manager_port" : 8872,
        #"optional_args" : ["-train", "-timeout", "-1", "-notactile"],
        "optional_args" : ["-train", "-timeout", "-1", "-notactile", "-record"] # creates recorded video file on env.close()
}

def canGrab(pos):
    return np.abs(pos[0]) < 0.3 and np.abs(pos[1]) < 0.1 and pos[2] < 3.

def canPut(pos):
    return np.abs(pos[0]) < 0.1 and np.abs(pos[1]) < 0.2 and np.abs(pos[2] - 3.) < 0.1

def moveTowardsObject(pos):
    action = np.zeros(5)
    action[2] = 1 if pos[1] < 0 else -1
    action[3] = 0
    action[4] = 0
    if pos[2] > 0 and np.abs(pos[0] / pos[2]) < 0.5:
        action[0] = 1 if pos[2] > 3 else -1
        action[1] = 1 if pos[0] > 0 else -1
    else:
        action[0] = 0
        action[1] = 1
    return action

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='VECA Algorithm Server')
    parser.add_argument('--ip', type=str, 
                        default=cfg_default["env_manager_ip"], help='Envionment Manager machine\'s ip')
    parser.add_argument('--port', type=int, metavar="ENV_PORT", 
                        default = cfg_default["env_manager_port"], help='Environment Manager\'s port')
    parser.add_argument('--num_envs', type=int, 
                        default = cfg_default["num_envs"], help='Number of parallel environments to execute')
    args = parser.parse_args()
    args.optional_args = cfg_default["optional_args"]
   
    yoffset = 0.5
    
    env = Environment(ip = args.ip, port=args.port, num_envs = args.num_envs, args = args.optional_args)
    
    action_dim = env.action_space
    action_init = np.zeros((args.num_envs, action_dim))
    print(action_init.shape)
    obs, reward, done, infos = env.step(action_init)

    level = 1
    while True:
        if level == 6:
            if "--record" in args.optional_args:
                break
            env.reset()
            obs, reward, done, infos = env.step(np.zeros((args.num_envs, action_dim)))
            level = 1
        tot_action = np.zeros((args.num_envs, action_dim))
        for i in range(args.num_envs):
            image = obs['image'][i]
            pos = infos['pos'][i]
            grab = infos['grab'][i]
            grabbed = grab[level]
            if grabbed == False:
                action = np.zeros((args.num_envs, action_dim))
                pass
                if canGrab(pos[level]):
                    action = np.array([0, 0, 0, 0, 1])
                else:
                    action = moveTowardsObject(pos[level])
            else:
                if canPut(pos[level-1] + np.array([0., yoffset, 0.])):
                    action = np.array([0, 0, 0, 0, 1])
                    level += 1
                else:
                    action = moveTowardsObject(pos[level-1] + np.array([0., yoffset, 0.]))

            tot_action[i] = action
        obs, reward, done, infos = env.step(tot_action)

    env.write_record()

    env.close()


