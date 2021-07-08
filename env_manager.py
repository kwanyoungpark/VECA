import numpy as np
from veca.env_manager import EnvManager
import argparse

cfg_default={
        "exposed_port" : 8872,
        "executable" : "./bin/disktower/BabyMindDG.exe",
        #"executable" : "./bin/kicktheball/VECA-BS.exe",
        #"executable" : "./bin/mazenav/VECA-BS.exe",
        #"executable" : "./bin/babyrun/VECA-BS.exe",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VECA Environment Manager')
    parser.add_argument('--executable', type=str, 
                        default=cfg_default["executable"], help='Unity Executable Path')
    parser.add_argument('--port', type=int, metavar="PORT", 
                        default = cfg_default["exposed_port"], help='Port exposed for algorithm')
    args = parser.parse_args()

    env = EnvManager(
        port = args.port,  
        exec_str = args.executable)
