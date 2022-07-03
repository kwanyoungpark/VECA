import numpy as np
import socket
import sys,json
import time
from veca.env_manager.env import UnityEnv
from veca.utils import decode, recvall, types, typesz, STATUS, build_packet, recv_json_packet, build_json_packet
import base64 
import os.path, gdown, zipfile, os

class TaskInfo():
    def __init__(self, path, download_link):
        self.path = path
        self.download_link = download_link

task_executable={
        "disktower" : TaskInfo("./env_manager/bin/disktower/VECA_latest.exe", "https://drive.google.com/uc?export=download&id=1jf4aWG9BR20HVj4sNArTEzqbK6SpSS6P"),
        "kicktheball" : TaskInfo("./env_manager/bin/kicktheball/VECA-BS.exe","https://drive.google.com/uc?export=download&id=1Qq9SuDMB_0yim05mB_fJwDHriT4amAQZ"),
        "mazenav" : TaskInfo("./env_manager/bin/mazenav/VECA-BS.exe","https://drive.google.com/uc?export=download&id=1nU512vgk7QytXQQtgNz9gZ5hHd4oR_wb"),
        "babyrun" : TaskInfo("./env_manager/bin/babyrun/VECA-BS.exe","https://drive.google.com/uc?export=download&id=1fbpQffo30ULbInX21NqP6U5nuEycKtW0"),
}

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

class EnvManager():
    def __init__(self,  port, localport = 46490):
        self.localport = localport
        
        self.listen(port)
        self.conn, addr = self.sock.accept()
        self.serve()
    
    def listen(self,port):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        hostName = socket.gethostbyname( '0.0.0.0' )
        self.sock.bind((hostName, port))
        print("Listening to ",hostName, " PORT ",port)
        self.sock.listen(1) 

    def serve(self):    
        _, _, packet = recv_json_packet(self.conn)
        self.task = packet["task"]
        self.TOTAL_NUM_ENVS = packet["TOTAL_NUM_ENVS"]
        self.NUM_ENVS = packet["NUM_ENVS"]
        self.args = packet["args"]
        #if self.TOTAL_NUM_ENVS % self.NUM_ENVS != 0:
        #    raise NotImplementedError('NOT SUPPORTED YET')
        if self.task not in task_executable.keys():
            raise NotImplementedError('TASK NOT SUPPORTED YET')
        self.exec_str = task_executable[self.task].path
        if not os.path.exists(self.exec_str):
            url = task_executable[self.task].download_link
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

        self.ENVS_PER_ENV = self.TOTAL_NUM_ENVS // self.NUM_ENVS
            
        self.envs = []
        for i in range(self.NUM_ENVS):
            self.envs.append(UnityEnv(self.ENVS_PER_ENV, self.localport + i, self.exec_str, self.args))
        self.envs = list(reversed(self.envs))
        
        try:
            self.AGENTS_PER_ENV = self.envs[0].AGENTS_PER_ENV
            self.NUM_AGENTS = self.AGENTS_PER_ENV * self.TOTAL_NUM_ENVS
            self.action_space = self.envs[0].action_space
            
            payload = {"AGENTS_PER_ENV":self.AGENTS_PER_ENV, "action_space": self.action_space}
            packet = build_json_packet(STATUS.INIT,payload)
            self.conn.sendall(packet)

            while True:
                status_code = decode(recvall(self.conn, 1), 'uint8')
                
                if status_code == STATUS.REST:
                    mask = recvall(self.conn, self.TOTAL_NUM_ENVS)
                    self.reset(mask)
                    
                elif status_code == STATUS.STEP:
                    action = recvall(self.conn, 4 * self.NUM_AGENTS * self.action_space)
                    self.step(action)
                    
                elif status_code == STATUS.RECO:
                    self.reset_connection()
                    
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
        payload = {}

        for type_num in range(5):
            payload[types[type_num]] = {}
            
            obsAgent, obsEnv, resAgent, resEnv = [], [], {}, {}
            for i in range(self.NUM_ENVS):
                obAgent, obEnv = self.envs[i].get_observation(type_num)
                obsAgent.append(obAgent)
                obsEnv.append(obEnv)

            for key in obsAgent[0].keys(): # We suppose that keys are same
                value = []
                
                for i in range(self.NUM_ENVS):
                    assert (len(obsAgent[i][key]) == 1)
                    value = value + [base64.b64encode(obsAgent[i][key][0]).decode('ascii'),]
                resAgent[key] = value
            payload[types[type_num]]["resAgent"] = resAgent
           

            for key in obsEnv[0].keys(): # We suppose that keys are same
                value = []
                for i in range(self.NUM_ENVS):
                    value = value + [base64.b64encode(obsEnv[i][key][0]).decode('ascii'),]
                resEnv[key] = value
            payload[types[type_num]]["resEnv"] = resEnv
            
        packet = build_json_packet(STATUS.STEP, payload)
        '''
        payload = json.dumps(payload).encode('utf-8')
        packet = build_packet(STATUS.STEP, [len(payload).to_bytes(4, 'little'),payload])
        '''
        self.conn.sendall(packet)


    def reset(self, mask):
        for i, env in enumerate(self.envs):            
            s, e = i*self.ENVS_PER_ENV, (i+1)*self.ENVS_PER_ENV
            env.reset(mask[s:e])

    def reset_connection(self):
        for env in self.envs:
            env.close()
            time.sleep(5)
            env.start_connection()
        
    def close(self):
        self.conn.close()
        for env in self.envs:
            env.close()
        time.sleep(2)
        self.sock.close()
