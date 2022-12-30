import numpy as np
import socket
import struct
from veca.network import decode, recvall, types, typesz, STATUS, build_packet, recv_json_packet, build_json_packet, response, request
import json, base64
import time
from multiprocessing import Process
from veca.env_manager import EnvOrchestrator

class EnvModule():

    @property
    def exec_path_win(self):
        raise NotImplementedError

    @property
    def download_link_win(self):
        raise NotImplementedError
        
    @property
    def exec_path_linux(self):
        raise NotImplementedError

    @property
    def download_link_linux(self):
        raise NotImplementedError

    def __init__(self, task, num_envs, args, seeds,
            remote_env, port,
            exec_path_win, download_link_win,
            exec_path_linux, download_link_linux,
        ):
        self.task = task
        self.num_envs = num_envs
        total_num_envs = num_envs
        self.remote_env = remote_env

        self.listen(port)
        if not self.remote_env:
            self.env_orchestrator = Process(target=EnvOrchestrator, kwargs={'ip': 'localhost', 'port': port, 'port_instance' : 46490})
            self.env_orchestrator.start()
            time.sleep(3)

        self.conn, addr = self.sock.accept()

        self.env_init(task, num_envs, total_num_envs, args, seeds,
            exec_path_win, download_link_win,
            exec_path_linux, download_link_linux,
        )
    
    def listen(self,port):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        hostName = socket.gethostbyname( '0.0.0.0' )
        self.sock.bind((hostName, port))
        print("VECA GYM API Listening to ",hostName, " PORT ",port)
        self.sock.listen(1) 
        
    def env_init(self, task, num_envs, total_num_envs, args, seeds,
            exec_path_win, download_link_win,
            exec_path_linux, download_link_linux,
        ):
        payload = {"task": task, "NUM_ENVS":num_envs, "TOTAL_NUM_ENVS": total_num_envs, "args": args, "seeds": np.array(seeds),
            "exec_path_win":exec_path_win, "download_link_win":download_link_win, 
            "exec_path_linux":exec_path_linux, "download_link_linux":download_link_linux, 
        }
        request(self.conn,STATUS.INIT, payload )
        
        _,_, packet = response(self.conn)
        
        self.agents_per_env = packet["AGENTS_PER_ENV"]
        self.action_space = packet["action_space"]
        
        self.num_agents = self.num_envs * self.agents_per_env
        
    def step(self, action, ignore_agent_dim = False):
        try:
            if ignore_agent_dim:
                assert self.agents_per_env == 1
            self.send_action(action)
            return self.collect_observations(ignore_agent_dim = ignore_agent_dim)
        except ConnectionError as ex:
            self.close()
            print(ex)
            raise
        except KeyboardInterrupt as ex:
            self.close()
            print(ex)
            raise

    def send_action(self, action):
        action = np.reshape(action, [self.num_envs, self.agents_per_env, self.action_space]).astype(np.float32)
        request(self.conn, STATUS.STEP, {"action":action})

    def collect_observations(self, ignore_agent_dim = False):
        status, metadata, data = response(self.conn)
        reward = data.pop("agent/reward")
        done = data.pop("agent/done", None)
        if done is None:
            done = data.pop("env/done")
        return data, reward, done
    
    def reset(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype = np.uint8)
        else:
            mask = np.array(mask, dtype = np.uint8)
        request(self.conn,STATUS.REST, {"mask":mask} )

    def reset_connection(self):
        request(self.conn, STATUS.RECO, {})
        
    def close(self):
        request(self.conn, STATUS.CLOS, {})
        time.sleep(1)
        self.conn.close()
        self.sock.close()
        time.sleep(3)

        if not self.remote_env:
            self.env_orchestrator.join()
