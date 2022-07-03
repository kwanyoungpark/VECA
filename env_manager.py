import numpy as np
from veca.env_manager import EnvManager
import argparse


cfg_default={
        "exposed_port" : 8872,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VECA Environment Manager')
    parser.add_argument('--port', type=int, metavar="PORT", 
                        default = cfg_default["exposed_port"], help='Port exposed for algorithm')
    args = parser.parse_args()

    env = EnvManager(
        port = args.port)
