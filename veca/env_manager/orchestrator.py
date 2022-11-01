import numpy as np
import socket
import sys,json
import time
from veca.env_manager.instance import UnityInstance
from veca.network import decode, recvall, types, typesz, STATUS, build_packet, recv_json_packet, build_json_packet, response, request, HelpException
import base64 
import os.path, gdown, zipfile, os, stat

class TaskInfo():
    def __init__(self, path, download_link):
        self.path = path
        self.download_link = download_link

import platform 

def validate_ip_and_port(ip, port, num_envs):
    ip = socket.gethostbyname(ip)
    for i in range(num_envs):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip,port+i))
        if result != 0:
            raise ConnectionError("Port is not open")
        else:
            print("Port is open")
        sock.close()
    return ip,port

class EnvOrchestrator():
    def __init__(self,  port:int, port_instance:int = 46490):
        self.port_instance = port_instance
        
        self.listen(port)
        self.conn, addr = self.sock.accept()

        _, _, order = response(self.conn)
        print("Order:", order)
        #_, _, order = recv_json_packet(self.conn)

        self.download_unity_app(order)
        self.handshake(order)
        self.serve(order)
    
    def listen(self,port):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        hostName = socket.gethostbyname( '0.0.0.0' )
        self.sock.bind((hostName, port))
        print("Env Orchestrator Listening to ",hostName, " PORT ",port)
        self.sock.listen(1) 

    def download_unity_app(self, packet):
        self.task = packet["task"]

        cur_os = platform.system() 
        if "Windows" in cur_os:
            self.task_info = TaskInfo(packet["exec_path_win"], packet["download_link_win"])
        elif "Linux" in cur_os:
            if packet["download_link_linux"] == "":
                self.close()
                raise NotImplementedError("Linux currently not supported")
            self.task_info = TaskInfo(packet["exec_path_linux"], packet["download_link_linux"])
        else:
            self.close()
            raise NotImplementedError("OS not supported")

        print("download Link:", self.task_info.download_link)
        print("exec_path:", self.task_info.path)

        try:
            self.exec_str = self.task_info.path
            if not os.path.exists(self.exec_str):
                url = self.task_info.download_link
                exec_dir = os.path.dirname(self.exec_str)
                os.makedirs(exec_dir, exist_ok=True)
                output = os.path.join(exec_dir, self.task + ".zip")
                gdown.download(url, output, quiet=False)
                print(output)
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(exec_dir)
                    print(exec_dir)
                if not os.path.exists(self.exec_str):
                    raise ValueError('CANNOT DOWNLOAD TASK EXECUTABLE.')
                os.chmod(packet["exec_path_linux"], os.stat(packet["exec_path_linux"]).st_mode | stat.S_IEXEC)
        except Exception as ex:
            self.close()
            print(ex)
            raise

    def handshake(self, packet):
        self.TOTAL_NUM_ENVS = packet["TOTAL_NUM_ENVS"]
        self.NUM_ENVS = packet["NUM_ENVS"]
        self.args = packet["args"]
        self.seeds = packet['seeds']

        self.ENVS_PER_ENV = self.TOTAL_NUM_ENVS // self.NUM_ENVS
        try:
            self.envs = []
            for i in range(self.NUM_ENVS):
                if self.seeds is not None: args = self.args + ["-seed", str(self.seeds[i])]
                else: args = self.args
                self.envs.append(UnityInstance(self.ENVS_PER_ENV, self.port_instance + i, self.exec_str, args))
            self.envs = list(reversed(self.envs))

            self.AGENTS_PER_ENV = self.envs[0].AGENTS_PER_ENV
            self.NUM_AGENTS = self.AGENTS_PER_ENV * self.TOTAL_NUM_ENVS
            self.action_space = self.envs[0].action_space
            
            payload = {"AGENTS_PER_ENV":self.AGENTS_PER_ENV, "action_space": self.action_space}

            request(self.conn, STATUS.INIT, payload)
            #packet = build_json_packet(STATUS.INIT,payload)
            #self.conn.sendall(packet)

        except HelpException as ex:
            self.close()
            print("--help passed to execution arguments") 
            return                   
        except ConnectionError as ex:
            self.close()
            print(ex)
            raise ex
        except KeyboardInterrupt as ex:
            self.close()
            print(ex)
            raise ex

    def serve(self,packet):    
        try:
            while True:
                status, _, data = response(self.conn)
                #status_code = decode(recvall(self.conn, 1), 'uint8')
                
                if status == STATUS.REST:
                    #mask = recvall(self.conn, self.TOTAL_NUM_ENVS)
                    self.reset(data["mask"])
                    
                elif status == STATUS.STEP:
                    #action = recvall(self.conn, 4 * self.NUM_AGENTS * self.action_space)
                    observations = self.step(data["action"])
                    request(self.conn, STATUS.STEP, observations)
                    
                elif status == STATUS.RECO:
                    self.reset_connection()

                elif status == STATUS.CLOS:
                    self.close()
                    break
                    
        except ConnectionError as ex:
            self.close()
            print(ex)
            raise
        except KeyboardInterrupt as ex:
            self.close()
            print(ex)
            raise

    def step(self, action):
        for i in range(self.NUM_ENVS):
            s, e = 4*i*(self.AGENTS_PER_ENV*self.ENVS_PER_ENV*self.action_space),4*(i+1)*(self.AGENTS_PER_ENV*self.ENVS_PER_ENV*self.action_space)
            #self.envs[i].send_action(action)
            self.envs[i].send_action(action[s:e])
        self.get_observation()
    
    def get_observation(self):
        storage = []
        for i in range(self.NUM_ENVS):
            status, metadata, data = self.envs[i].get_observation()
            storage.append((status, metadata, data))
        
        collate = {}
        for _,_,data in storage:
            for key,value in data.items():
                if key not in collate: collate[key] = []
                collate[key].append(value)
        for key, valuelist in collate.items():
            collate[key] = np.concatenate(valuelist)
        return collate

    def reset(self, mask):
        for i, env in enumerate(self.envs):            
            s, e = i*self.ENVS_PER_ENV, (i+1)*self.ENVS_PER_ENV
            env.reset(mask[s:e])

    def reset_connection(self):
        for env in self.envs:
            env.close()
            time.sleep(3)
            env.start_connection()
        
    def close(self):
        for env in self.envs:
            env.close()
        time.sleep(3)
        self.conn.close()
        self.sock.close()
