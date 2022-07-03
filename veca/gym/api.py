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

def make(task, num_envs, ip, port, args):
    if task not in task_env.keys():
        raise ValueError("Task [" + task + "] not supported.\n"+
        "Supported tasks: " + str(list(task_env.keys()))
        )
    print(task)
    return task_env[task](
            task = task,
            num_envs = num_envs, 
            ip = ip, port = port, 
            args = args
            )
