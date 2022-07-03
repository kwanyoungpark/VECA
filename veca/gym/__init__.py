from veca.gym.api import make, task_env

def list_tasks():
    return list(task_env.keys())