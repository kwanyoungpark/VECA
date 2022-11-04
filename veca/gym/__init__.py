'''
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
'''
from veca.gym.core import EnvModule


import os, sys, importlib.util, glob
TASKS_ROOTDIR_DEFAULT = os.path.join('veca', 'gym','tasks')
ENV_FILENAME_DEFAULT = 'environment.py'

def import_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sanity_check(task_module):
    assert hasattr(task_module, "Environment")
    assert issubclass(task_module.Environment, EnvModule)
    assert hasattr(task_module.Environment, 'exec_path_win') and hasattr(task_module.Environment, 'download_link_win')
    assert hasattr(task_module.Environment, 'exec_path_linux') and hasattr(task_module.Environment, 'download_link_linux')
    return

def import_envs_from_dir(tasks_dir, task_name = None):
    modules = {}
    if task_name is None:
        env_paths = glob.glob(os.path.join(tasks_dir,'*',ENV_FILENAME_DEFAULT))
        for env_path in env_paths:
            parent_name = os.path.split(os.path.dirname(env_path))[-1]
            module = import_module_from_path(env_path, parent_name)
            sanity_check(module)
            modules[parent_name] = module.Environment
    else:
        env_paths = glob.glob(os.path.join(tasks_dir,task_name,ENV_FILENAME_DEFAULT))
        env_path = env_paths[0]
        module = import_module_from_path(env_path, task_name)
        sanity_check(module)
        modules[task_name] = module.Environment
    return modules


def list_tasks(tasks_dir:str = TASKS_ROOTDIR_DEFAULT):
    env_paths = glob.glob(os.path.join(tasks_dir,'*',ENV_FILENAME_DEFAULT))
    # sanity check in here for each env_path will be good
    out = [os.path.split(os.path.dirname(env_path))[-1] for env_path in env_paths]
    return out

from typing import Optional
def make(task: str, num_envs: int, args: 'list[str]', seeds:Optional['list[int]'] = None, remote_env:bool = False, ip:Optional[str] = None, port: int = 46489):
    tasks_list = list_tasks()
    if task not in tasks_list:
        raise ValueError("Task [" + task + "] not supported.\n"+
        "Supported tasks: " + str(tasks_list)
        )
    env = import_envs_from_dir(tasks_dir = TASKS_ROOTDIR_DEFAULT, task_name = task)[task]
    print("Imported " ,task, "module")

    if seeds is not None: assert len(seeds) == num_envs, "If using seeds, it should be same length as num_envs"

    return env(
            task = task,
            num_envs = num_envs, 
            args = args,
            seeds = seeds,
            remote_env = remote_env,
            port = port, 
            )