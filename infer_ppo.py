import tensorflow as tf
import numpy as np
import os
from baselines.ppo.layer import *
from baselines.ppo.mtl_model_PPO import Model as MTLModel
from baselines.ppo.replaybuffer import MultiTaskReplayBuffer
from baselines.ppo.utils import AdaptiveLR, Saver
import veca.gym
import random
import time, argparse
from baselines.ppo.dataloader import MultiTaskDataLoader
from baselines.ppo.curriculum import Curriculum
import scipy.ndimage
import copy

import cv2
class Recorder:
    def __init__(self,  filedir):
        self.storage = []
        os.makedirs(filedir, exist_ok=True)
        self.filedir = filedir
    def add(self, data):
        self.storage.append(data)
    def flush(self, idx:int):
        data = self.storage[0]
        frameSize = data.shape[1:3]
        frameSize = (frameSize[1], frameSize[0])
        print("frame size", frameSize)
            
        writer = cv2.VideoWriter(os.path.join(self.filedir, f'{idx}.avi') ,cv2.VideoWriter_fourcc(*'XVID'), 12, frameSize)
        
        for img in self.storage:
            img = img[0]
            img = (img*255).astype(np.uint8)
            img_processed = copy.deepcopy(img)
            img_processed[:,:,2] = img[:,:,0]
            img_processed[:,:,0] = img[:,:,2]
            img_processed = (255 - img_processed)[::-1,:,:]
            writer.write(img_processed)
        writer.release()
        self.storage = []
        
import soundfile as sf
class AudioRecorder:
    def __init__(self,  filedir):
        self.storage = []
        os.makedirs(filedir, exist_ok=True)
        self.filedir = filedir
        self.sample_rate = 44100
    def add(self, data):
        self.storage.append(data)
    def flush(self, idx:int):
        output = np.concatenate(self.storage,axis = 0)
        sf.write(os.path.join(self.filedir,f'{idx}.wav'), output, self.sample_rate)
        self.storage = []

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='VECA Navigation')
    parser.add_argument('--stage1', type=int, 
                        help='stage 1 init step', required = True)
    parser.add_argument('--stage2', type=int, 
                        help='stage 2 init step', required = True)
    parser.add_argument('--tag', type=str, default = "PPO_COGNIANav", required=True)
    parser.add_argument('--gpuno', type=int, default = 0)
    parser.add_argument('--port', type=int, default = 10008)

    args = parser.parse_args()
    tag = args.tag
    PORT = args.port

    num_envs = 1
    TRAIN_STEP = 20000000
    SAVE_STEP = 100_000
    REC_STEP = 100000
    NUM_CHUNKS = 4
    NUM_UPDATE = 1
    TRAIN_LOOP = 4
    TIME_STEP = 128
    BUFFER_LENGTH = 128
    HORIZON = 600
    GAMMA = 0.99
    LAMBDA = 0.95
    NUM_CHUNKS = 4
    TIME_STEP = 128
    entropy_coeff = 0.01
    STAGE1 = args.stage1
    STAGE2 = args.stage2
    print("STAGE1", STAGE1, "STAGE2", STAGE2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()

    envs = [veca.gym.make(
                task = "cognianav",                                 # VECA task name
                num_envs = num_envs,                                # Number of parallel environment instances to execute
                args = ["--train", "--medium", "--record"],                   # VECA task additional arguments. Append "--help" to list valid arguments.
                seeds = random.sample(range(0, 2000),num_envs ),    # seeds per env instances
                remote_env = True, port = PORT                      # Whether to use the Environment Orchestrator process at a remote server.
            )]

    dl = MultiTaskDataLoader(envs, curriculum = Curriculum(STAGE1,STAGE2))
    buffers = MultiTaskReplayBuffer(len(envs), buffer_length=BUFFER_LENGTH, timestep=TIME_STEP)

    dl.reset(buffers)

    obs_sample = buffers.sample_batch()
    model = MTLModel(envs, sess, obs_sample, tag)

    result_dir = os.path.join("work_dir", tag)

    saver = Saver(sess)
    ckpt_steps = saver.load_if_exists(ckpt_dir = result_dir)
    
    obs, reward, done, infos = dl.sample(buffers)
    
    topview_video = Recorder(os.path.join(args.tag ,"topview"))
    birdeyeview_video = Recorder(os.path.join(args.tag, "birdeyeview"))
    fpv_video = Recorder(os.path.join(args.tag ,"fpvview"))
    left_audio = AudioRecorder(os.path.join(args.tag ,"leftaudio"))

    cumulative_reward = 0.0
    STEPS = 120000
    video_idx = 0
                                
    for step in range(STEPS):
        actions = model.get_action(obs)
        obs, reward, done, infos = dl.step(actions,buffers)
        
        left_audio.add(obs[0]['agent/wav'][0,0])
        topview_video.add(infos[0]['agent/Topview'])
        birdeyeview_video.add(infos[0]['agent/BirdEyeview'])
        fpv_video.add(infos[0]['agent/FPview'])
        cumulative_reward += reward[0]['raw_reward'] + reward[0]['helper_reward']
        print(f"step {step}, vid_idx {video_idx}, done {done[0][0]}" )
        
        if (done[0][0] == 1) or ((step+1) % HORIZON == 0):
            if ((step+1) % HORIZON) == 0:
                dl.reset(buffers)
            cumulative_reward = 0.0
            topview_video.flush(video_idx)
            birdeyeview_video.flush(video_idx)
            fpv_video.flush(video_idx)
            left_audio.flush(video_idx)
            video_idx += 1       
            
    topview_video.save()
    birdeyeview_video.save()
    fpv_video.save()
    
    [env.close() for env in envs]
    


