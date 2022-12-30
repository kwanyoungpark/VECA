#METHOD = 'PPO'
METHOD = 'PPO'
#INFERENCE = True
INFERENCE = False
RECORD = False
if RECORD:
    INFERENCE = True
'''
Define task.
    SIM : Name of simulator (ATARI or VECA)
    TASK : Name of task
'''
#SIM = 'Pendulum'
#TASKS = ['Pendulum']

#SIM = 'CartPole'
#TASKS = ['CartPole']

#SIM = 'ATARI'
#TASKS = ['Pong']
#TASKS = ['Pong', 'Breakout']

SIM = 'VECA'
#TASKS = ['Navigation', 'KickTheBall', 'MANavigation']
#TASKS = ['MANavigation']
#TASKS = ['KickTheBall']
#TASKS = ['Navigation']
#TASKS = ['Transport']
TASKS = ['RunBaby']
#TASKS = ['MazeNav']
#TASKS = ['COGNIANav']
#TASKS = ['NavigationCogSci']

RESET_STEP = 2560
WAV_LENGTH = 250
STATE_LENGTH = 512
NUM_ENVS = 8

GAMMA = 0.99
LAMBDA = 0.95

RNN = False
if METHOD == 'PPO':
    NUM_CHUNKS = 4
    NUM_UPDATE = 1
    TRAIN_LOOP = 4
    if RNN:
        RNN_STEP = 8
        TIME_STEP = 128
    else:
        TIME_STEP = 128
    BUFFER_LENGTH = TIME_STEP
if METHOD == 'SAC':
    NUM_CHUNKS = 4
    NUM_UPDATE = 1
    TRAIN_LOOP = 4
    TIME_STEP = 128
    BUFFER_LENGTH = 2500

if INFERENCE:
    BUFFER_LENGTH = 8
    NUM_ENVS = 1
'''
if NUM_CHUNKS is None:
    BATCH_SIZE = 256
else:
    BATCH_SIZE = (TIME_STEP * NUM_ENVS) // NUM_CHUNKS
'''
FRAME_RATE = 15
MDN = False
if MDN == True:
    NUM_QR = 5
