import numpy as np
import socket
import struct

def readBytes(byteArray, numtype):
    if numtype == 'uint8':
        size = len(byteArray)
        return struct.unpack('B' * size, byteArray)
    if numtype == 'int16':
        size = len(byteArray) // 2
        return struct.unpack('h' * size, byteArray)
    if numtype == 'int':
        size = len(byteArray) // 4
        return struct.unpack('i' * size, byteArray)
    if numtype == 'float':
        size = len(byteArray) // 4
        return struct.unpack('f' * size, byteArray)
    if numtype == 'str':
        return byteArray.decode('ascii')
    if numtype == 'char':
        return byteArray.decode('utf-16')

def readData(socket, length):
    length_left = length
    res = b''
    while length_left > 0:
        s = socket.recv(min(4096, length_left))
        res += s
        length_left -= len(s)
    return res

types = ['char', 'int', 'float', 'uint8', 'int16']
typesz = [2, 4, 4, 1, 2]

class GeneralEnvironment():
    def __init__(self, NUM_ENVS, port):
        self.port = port
        self.num_envs = NUM_ENVS
        self.start_connection()
    
    def start_connection(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        hostName = socket.gethostbyname( '0.0.0.0' )
        print(hostName)
        #self.sock.bind(('localhost', PORT))
        self.sock.bind((hostName, self.port))
        self.sock.listen(1)
        (self.conn, self.addr) = self.sock.accept()
        print("CONNECTED")
        self.conn.sendall(self.num_envs.to_bytes(4, 'little'))
        self.agents_per_env = readBytes(readData(self.conn, 4), 'int')[0]
        self.action_space = readBytes(readData(self.conn, 4), 'int')[0]
        self.num_agents = self.num_envs * self.agents_per_env
        #self.conn.sendall(ACTION_LENGTH.to_bytes(4, 'little'))
        print(self.num_envs.to_bytes(4, 'little'))
        #print(ACTION_LENGTH.to_bytes(4, 'little'))
        print("GO")

    def send_action(self, action):
        self.conn.sendall(b'STEP')
        action = np.reshape(action, [self.num_envs, self.agents_per_env, self.action_space])
        #self.conn.sendall(self.num_envs.to_bytes(4, 'little'))
        #self.conn.sendall(self.action_space.to_bytes(4, 'little'))
        action = np.array(action).astype(np.float32)
        self.conn.sendall(action.tobytes())
        #print(b'STEP'+self.num_envs.to_bytes(4, 'little') + self.action_space.to_bytes(4, 'little') + action.tobytes(), len(action.tobytes()))

    def step(self, action, ignore_agent_dim = False):
        if ignore_agent_dim:
            assert self.agents_per_env == 1
        #print("1")
        self.send_action(action)
        #print("2")
        return self.collect_observations(ignore_agent_dim = ignore_agent_dim)

    def collect_observations(self, ignore_agent_dim = False):
        obs = {}
        for (T, data_type) in enumerate(types):
            # AgentObs
            N = readBytes(readData(self.conn, 4), 'int')[0]
            #print('N', N)
            for _ in range(N):
                keyL = readBytes(readData(self.conn, 4), 'int')[0]
                #print('keyL', keyL)
                key = readBytes(readData(self.conn, keyL), 'str')
                #print('key', key)
                value = []
                for i in range(self.num_envs):    
                    value_t = []
                    for j in range(self.agents_per_env):
                        valueL = readBytes(readData(self.conn, 4), 'int')[0] * typesz[T]
                        #print(valueL)
                        if valueL > 0:
                            value_t.append(readBytes(readData(self.conn, valueL), data_type))
                        else:
                            value_t.append(None)
                    if ignore_agent_dim:
                        value.append(value_t[0])
                    else:
                        value.append(value_t)
                obs[key] = value

            # EnvObs
            N = readBytes(readData(self.conn, 4), 'int')[0]
            #print('N', N)
            for _ in range(N):
                keyL = readBytes(readData(self.conn, 4), 'int')[0]
                #print('keyL', keyL)
                key = readBytes(readData(self.conn, keyL), 'str')
                #print('key', key)
                value = []
                for i in range(self.num_envs):
                    valueL = readBytes(readData(self.conn, 4), 'int')[0] * typesz[T]
                    #print(key, valueL, typesz[j], types[j])
                    if valueL > 0:
                        value.append(readBytes(readData(self.conn, valueL), data_type))
                    else:
                        value.append(None)
                obs[key] = value
        return obs
    
    def reset(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype = np.uint8)
        mask = np.array(mask, dtype = np.uint8)
        self.conn.sendall(b'REST')
        self.conn.sendall(mask.tobytes())    

    def reset_connection(self):
        # Due to memory leak in socket
        self.conn.sendall(b'RECO')
        #self.conn.close()
        #self.sock.close()
        #self.start_connection()
        
    def close(self):
        self.conn.close()
        self.sock.close()
