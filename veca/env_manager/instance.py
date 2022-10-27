from re import A
import numpy as np
import socket
import subprocess
import time
from veca.network import decode, recvall, types, typesz, build_json_packet, STATUS, recv_json_packet
import struct, base64
from typing import Tuple

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
        print("Unity Instance Deployed at "+ '-ip ' + '127.0.0.1' + ' -port ' + str(self.port))

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

    def _echo_test(self):
        '''
        data = {"test1":base64.b64encode(bytes([0,0,1,0])).decode("ascii"), 
            "test2":base64.b64encode(struct.pack('%sf' % 5,*[0.0,1.0,2.0,3.0,-4.0])).decode("ascii"), 
            "test3":base64.b64encode(np.ones((2,3,4,5)).tobytes()).decode("ascii")}
        metadata = {"test1":"float", "test2":"float", "test3":"np.float", "test4":"np.float32"}
        '''
        data = {"test1":[0,0,1,0], "test2":[0.0,1.0,2.0,3.0,-4.0], "test3":np.ones((2,3,4,5)), "test4":"HelloWorld!", "test5":1, "test6":6.0}
        metadata, data = self._protocol_encode(data)
        packet = build_json_packet(STATUS.REST, data, metadata, use_metadata= True)
        print("Send:", packet)
        self.conn.sendall(packet)
        output = recv_json_packet(self.conn, use_metadata = True)
        print("RECV:",output)

    def request(self, status, data:dict):
        metadata, data = self._protocol_encode(data)
        packet = build_json_packet(status, data, metadata, use_metadata= True)
        print("Send:", packet)
        self.conn.sendall(packet)

    def response(self):
        status, _, _, metadata, data = recv_json_packet(self.conn, use_metadata = True)
        return status, metadata, self._protocol_decode(metadata,data)

    def _protocol_encode(self,data:dict):
        output = {}
        metadata = {}
        def _encode(x) -> Tuple[str,str]:
            info = str(type(x))
            base64_ascii = lambda t:  base64.b64encode(t).decode("ascii")
            if isinstance(x,np.ndarray):
                return "/".join([str(type(x)), str(x.dtype), str(x.shape)]), base64_ascii(x.tobytes())
            elif isinstance(x, list):
                y = np.array(x)
                return "/".join([str(type(y)), str(y.dtype), str(y.shape)]), base64_ascii(y.tobytes())
            elif isinstance(x, str):
                return info, x
            elif isinstance(x, int):
                return info, base64_ascii(np.array([x]))
            elif isinstance(x, float):
                return info, base64_ascii(np.array([x]))
            elif isinstance(x, bytes):
                return info, base64_ascii(x)
            else:
                print(type(x))
                print(x.dtype)
                raise NotImplementedError()
        for key, value in data.items():
            type_enc, value_enc = _encode(value)
            output[key] = value_enc
            metadata[key] = type_enc

        return metadata, output
    def _protocol_decode(self, metadata:dict, data:dict):
        output = {}
        def _decode(x, info) : 
            info = info.split("/")
            typeinfo, shape = info[0], tuple(int(x) for x in info[1].replace("(","").replace(",)","").replace(")","").split(","))
            if "Byte[]" in typeinfo:
                return np.frombuffer(base64.b64decode(x[0].encode('ascii')), np.uint8).reshape(shape)
            elif "Int16[]" in typeinfo:
                return np.array(x).reshape(shape)
            elif "Single[]" in typeinfo:
                return np.array(x).reshape(shape)
            else:
                print(typeinfo)
                raise NotImplementedError()
        return {key: _decode(value,metadata[key]) for key,value in data.items()}

    def send_action(self, action):
        self._echo_test()

        self.request(STATUS.STEP, {"action":action})
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
        self._echo_test()

        self.request(STATUS.REST, {"mask":mask})
        if mask is None:
            mask = np.ones(self.NUM_ENVS, dtype = np.uint8)
        self.conn.sendall(b'REST')
        self.conn.sendall(mask)    

    def reset_connection(self):
        self._echo_test()

        if self.proc.poll() == None:
            print('Killed: Give more time to env?')
            self.proc.kill()
        self.start_connection()
        
    def close(self):
        self._echo_test()

        self.request(STATUS.CLOS, {})
        self.conn.sendall(b'CLOS')
        self.conn.close()
        self.sock.close()
