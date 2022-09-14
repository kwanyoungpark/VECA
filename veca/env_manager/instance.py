import numpy as np
import socket
import subprocess
import time
from veca.network import decode, recvall, types, typesz

class UnityInstance():
    def __init__(self, NUM_ENVS, port, exec_str, args):
        self.port = port
        self.exec_str = exec_str
        self.args = args
        self.NUM_ENVS = NUM_ENVS
        self.start_connection()
    
    def start_connection(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        hostName = socket.gethostbyname( '0.0.0.0' )

        exec_str = [self.exec_str] + self.args + ['-ip', 'localhost', '-port', str(self.port)]
        print("CONNECTING TO "+ '-ip ' + '127.0.0.1' + ' -port ' + str(self.port))

        self.sock.bind((hostName, self.port))
        self.sock.listen(1)
        print(exec_str)
        self.proc = subprocess.Popen(exec_str)
        
        (self.conn, self.addr) = self.sock.accept()
        print("CONNECTED")
        self.conn.sendall(self.NUM_ENVS.to_bytes(4, 'little'))
        self.AGENTS_PER_ENV = decode(recvall(self.conn, 4), 'int')
        self.NUM_AGENTS = self.AGENTS_PER_ENV * self.NUM_ENVS
        self.action_space = decode(recvall(self.conn, 4), 'int')
                
        print(self.NUM_ENVS.to_bytes(4, 'little'))
        print("GO")

    def send_action(self, action):
        self.conn.sendall(b'STEP')
        self.conn.sendall(action)
        #print('STEP', action)
    
    def get_observation(self, type_num):
        obsAgent = {}
        data_type = types[type_num]
        N = decode(recvall(self.conn, 4), 'int')
        #print('N', N)
        for _ in range(N):
            keyL = decode(recvall(self.conn, 4), 'int')
            #print('keyL', keyL)
            key = decode(recvall(self.conn, keyL), 'str')
            #print('key', key)
            value = []
            for i in range(self.NUM_ENVS):
                for j in range(self.AGENTS_PER_ENV):
                    valueL = decode(recvall(self.conn, 4), 'int')
                    #print(valueL
                    if valueL > 0:
                        value.append(recvall(self.conn, valueL * typesz[type_num]))
                    else:
                        value.append(None)
            obsAgent[key] = value
        obsEnv = {}
        N = decode(recvall(self.conn, 4), 'int')
        #print('N', N)
        for _ in range(N):
            keyL = decode(recvall(self.conn, 4), 'int')
            #print('keyL', keyL)
            key = decode(recvall(self.conn, keyL), 'str')
            #print('key', key)
            value = []
            for i in range(self.NUM_ENVS):
                valueL = decode(recvall(self.conn, 4), 'int')
                #print(valueL)
                if valueL > 0:
                    value.append(recvall(self.conn, valueL * typesz[type_num]))
                else:
                    value.append(None)
            obsEnv[key] = value

        return obsAgent, obsEnv
    
    def reset(self, mask = None):
        if mask is None:
            mask = np.ones(self.NUM_ENVS, dtype = np.uint8)
        self.conn.sendall(b'REST')
        self.conn.sendall(mask)    

    def reset_connection(self):
        if self.proc.poll() == None:
            print('Killed: Give more time to env?')
            self.proc.kill()
        self.start_connection()
        
    def close(self):
        self.conn.sendall(b'CLOS')
        self.conn.close()
        self.sock.close()
