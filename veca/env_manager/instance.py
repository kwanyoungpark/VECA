from re import A
import numpy as np
import socket
import subprocess
import time
from veca.network import decode, recvall, types, typesz,  STATUS,  request, response
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
        request(self.conn, STATUS.INIT, {"num_envs":self.NUM_ENVS})
        status, _, data = response(self.conn)
        self.AGENTS_PER_ENV = data["agents_per_env"][0]
        self.NUM_AGENTS = self.AGENTS_PER_ENV * self.NUM_ENVS
        self.action_space = data["action_length"][0]

    def _echo_test(self):
        request(self.conn, STATUS.REST, 
                {"test1":[0,0,1,0], "test2":[0.0,1.0,2.0,3.0,-4.0], 
                "test3":np.ones((2,3,4,5)), "test4":"HelloWorld!", 
                "test5":1, "test6":6.0}
            )
        response(self.conn)



    def send_action(self, action):
        self._echo_test()

        request(self.conn, STATUS.STEP, {"action":action})
    
    def get_observation(self, type_num):
        return None
    
    def reset(self, mask = None):
        self._echo_test()

        if mask is None:
            mask = np.ones(self.NUM_ENVS, dtype = np.uint8)
        request(self.conn, STATUS.REST, {"mask":mask})

    def reset_connection(self):
        self._echo_test()

        if self.proc.poll() == None:
            print('Killed: Give more time to env?')
            self.proc.kill()
        self.start_connection()
        
    def close(self):
        self._echo_test()

        request(self.conn, STATUS.CLOS, {})
        self.conn.close()
        self.sock.close()
