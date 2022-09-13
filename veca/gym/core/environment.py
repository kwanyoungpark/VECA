import numpy as np
import socket
import struct
from veca.network import decode, recvall, types, typesz, STATUS, build_packet, recv_json_packet, build_json_packet
import json, base64
import time

class EnvModule():

    @property
    def exec_path(self):
        raise NotImplementedError

    @property
    def download_link(self):
        raise NotImplementedError

    def __init__(self, task, num_envs, ip, port,args, exec_path, download_link):
        ip = socket.gethostbyname(ip)
        self.num_envs = num_envs
        total_num_envs = num_envs
        self.conn = self.start_connection(ip, port)
        self.env_init(task, num_envs, total_num_envs, exec_path, download_link, args)
    
    def start_connection(self,ip, port):
        conn = socket.socket()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("CONNECTING TO ENV SERVER : " + str(ip) + ":" + str(port))
        
        conn.connect((ip, port))
        print("CONNECTED TO ENV SERVER")
        return conn
        
    def env_init(self, task, num_envs, total_num_envs, exec_path, download_link,args):
        payload = {"task": task,"exec_path":exec_path, "download_link":download_link, 
            "NUM_ENVS":num_envs, "TOTAL_NUM_ENVS": total_num_envs, "args": args}
        packet = build_json_packet(STATUS.INIT, payload)
        self.conn.sendall(packet)
        
        _,_, packet = recv_json_packet(self.conn)
        self.agents_per_env = packet["AGENTS_PER_ENV"]
        self.action_space = packet["action_space"]
        
        self.num_agents = self.num_envs * self.agents_per_env
        
    def step(self, action, ignore_agent_dim = False):
        try:
            if ignore_agent_dim:
                assert self.agents_per_env == 1
            self.send_action(action)
            obs = self.collect_observations(ignore_agent_dim = ignore_agent_dim)
            return obs
        except ConnectionError as ex:
            self.close()
            print(ex)
            raise
        except KeyboardInterrupt as ex:
            self.close()
            print(ex)
            raise

    def send_action(self, action):
        action = np.reshape(action, [self.num_envs, self.agents_per_env, self.action_space])
        action = np.array(action).astype(np.float32)
        
        packet = build_packet(STATUS.STEP, [action.tobytes(),])
        self.conn.sendall(packet)

    def collect_observations(self, ignore_agent_dim = False):
        
        status_code, length, payload = recv_json_packet(self.conn) 
        obs_type_separated = payload
        
        obs = {}
        for type_key, obs_cur in obs_type_separated.items():
            for source in ["resAgent", "resEnv"]:
                obs_base64 = obs_cur[source]
                for key, values in obs_base64.items():
                    obs[key] = []
                    if isinstance(values, list):
                        for value in values:
                            value_bytes = value.encode('ascii')
                            value = base64.b64decode(value_bytes)
                            obs[key].append(decode(value, type_key + "[]"))
        return obs
    
    def reset(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype = np.uint8)
        else:
            mask = np.array(mask, dtype = np.uint8)
        packet = build_packet(STATUS.REST, [mask.tobytes(),])
        self.conn.sendall(packet)  

    def reset_connection(self):
        # Due to memory leak in socket
        packet = build_packet(STATUS.RECO, [])
        self.conn.sendall(packet)
        
    def close(self):
        packet = build_packet(STATUS.CLOS, [])
        self.conn.sendall(packet)
        time.sleep(1)
        self.conn.close()
