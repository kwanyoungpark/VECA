import numpy as np
import socket
import subprocess
from veca.network import STATUS,  request, response, HelpException

class UnityInstance():
    def __init__(self, NUM_ENVS, port, exec_str, args):
        self.NUM_ENVS = NUM_ENVS
        exec_str = [exec_str] + args + ['--ip', 'localhost', '--port', str(port)]
        self.start_connection(port, exec_str, NUM_ENVS)
    
    def start_connection(self, port, exec_str, num_envs):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        hostName = socket.gethostbyname( '0.0.0.0' )
        self.sock.bind((hostName, port))

        self.sock.listen(1)
        print(exec_str)
        self.proc = subprocess.Popen(exec_str)
        print("Unity Instance Deployed at "+ '-ip ' + '127.0.0.1' + ' -port ' + str(port))
        
        (self.conn, self.addr) = self.sock.accept()
        print("CONNECTED")

        request(self.conn, STATUS.INIT, {"num_envs":num_envs})
        status, _, data = response(self.conn)
        if status == STATUS.HELP:
            print(data["help"])
            raise HelpException()
        elif status == STATUS.INIT:
            self.AGENTS_PER_ENV = data["agents_per_env"][0]
            self.NUM_AGENTS = self.AGENTS_PER_ENV * num_envs
            self.action_space = data["action_length"][0]
        else:
            raise NotImplementedError()

    def send_action(self, action):
        request(self.conn, STATUS.STEP, {"action":action})
    
    def get_observation(self):
        return response(self.conn)
    
    def reset(self, mask = None):
        if mask is None:
            mask = np.ones(self.NUM_ENVS, dtype = np.uint8)
        request(self.conn, STATUS.REST, {"mask":mask})

    def reset_connection(self):
        if self.proc.poll() == None:
            print('Killed: Give more time to env?')
            self.proc.kill()
        self.start_connection()
        
    def close(self):
        request(self.conn, STATUS.CLOS, {})
        self.conn.close()
        self.sock.close()
