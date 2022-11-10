import tensorflow as tf
import numpy as np
import os
from baselines.ppo.utils import *
from baselines.ppo.mtl_model_PPO import Model as MTLModel
from baselines.ppo.replaybuffer import ReplayBuffer
import veca.gym
import random
from baselines.ppo.headquarter import HeadQuarter


if __name__ == "__main__":
    tag = "Moveobj_112233_level1"
    PORT = 10000

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    num_envs = 1
    TRAIN_STEP = 2000000
    SAVE_STEP = 80000 / num_envs
    REC_STEP = 100000
    NUM_CHUNKS = 4
    NUM_UPDATE = 1
    TRAIN_LOOP = 4
    TIME_STEP = 128
    BUFFER_LENGTH = 2500

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()

    envs = [veca.gym.make(
                task = "cognianav",                                 # VECA task name
                num_envs = num_envs,                                # Number of parallel environment instances to execute
                args = ["--train"],                   # VECA task additional arguments. Append "--help" to list valid arguments.
                seeds = random.sample(range(0, 2000),num_envs ),    # seeds per env instances
                remote_env = False                                  # Whether to use the Environment Orchestrator process at a remote server.
                )]

    
    class MultiTaskDataLoader:
        def __init__(self, envs):
            self.heads = []
            for env in envs:
                self.heads.append(HeadQuarter(env = env))

        def step(self, actions,replay_buffers):
            assert len(actions) == len(self.heads)
            assert len(replay_buffers) == len(self.heads)
            collate = []
            for action, head, replay_buffer in zip(actions,self.heads, replay_buffers.replayBuffers):
                obs, reward, done, infos = head.step(action,replay_buffer)
                collate.append((obs,reward, done, infos))
            return tuple(zip(*collate))

        def sample(self, replay_buffers):
            assert len(replay_buffers) == len(self.heads)
            collate = []
            for head, replay_buffer in zip(self.heads, replay_buffers.replayBuffers):
                obs, reward, done, infos = head.sample(replay_buffer)
                collate.append((obs,reward, done, infos))
            return tuple(zip(*collate))

        def reset(self, replay_buffers):
            assert len(replay_buffers) == len(self.heads)
            for head, replay_buffer in zip(self.heads, replay_buffers.replayBuffers):
                head.reset(replay_buffer)

    class MultiTaskReplayBuffer:
        def __init__(self, num_tasks, buffer_length, timestep):
            self.timestep = timestep
            self.replayBuffers = []
            for i in range(num_tasks):
                self.replayBuffers.append(ReplayBuffer(capacity= buffer_length))
        def __len__(self):
            return len(self.replayBuffers)
        def get_batch(self):
            collate = []
            for buffer in self.replayBuffers:
                collate.append(buffer.get(self.timestep))
            return collate
        def clear(self):
            for replay_buffer in self.replayBuffers:
                replay_buffer.clear()

    dl = MultiTaskDataLoader(envs)
    buffers = MultiTaskReplayBuffer(len(envs), buffer_length=BUFFER_LENGTH, timestep=TIME_STEP)

    dl.reset(buffers)

    obs_sample = buffers.get_batch()
    print(obs_sample)
    model = MTLModel(envs, sess,obs_sample, tag)


    SCHEDULE = True

    class AdaptiveLR:
        def __init__(self, schedule):
            self.schedule = schedule 
            if self.schedule:
                self.lrAM, self.lrA0, self.lrA = 1e-3, 1e-4, 1e-4
            else:
                self.lrA = 2.5e-4
        def step(self, backup, approxkl, loss, lossP):
            if self.schedule:
                if (approxkl > 0.01) or (loss < lossP):
                    self.lrA /= 2
                    self.lrA0 *= 0.95
                else:
                    self.lrA = min(self.lrA0, self.lrA * 2)
                    self.lrA0 *= 1.02
                if (approxkl > 0.015) or (loss < lossP):
                    backup.revert()
                else:
                    backup.commit()
                self.lrA0 = min(self.lrA0, self.lrAM) 
            else:
                backup.commit()



    lr_scheduler = AdaptiveLR(schedule = SCHEDULE)
    frac = 1.
    entropy_coeff = 0.01 * frac
    
    dl.reset(buffers)
    obs, reward, done, infos = dl.sample(buffers)

    for step in range(TRAIN_STEP):
        actions = model.get_action(obs)
        obs, reward, done, infos = dl.step(actions,buffers)

        if (step+1) % TIME_STEP == 0:
            batches = buffers.get_batch()

            summarys = model.feed_batch(batches)
            loss, lA, lC, ratio, pg_loss, grad = model.forward(ent_coef = entropy_coeff)
            

            for idx in range(TRAIN_LOOP):
                approxkl = model.optimize_step(lr = lr_scheduler.lrA / (TRAIN_LOOP), ent_coef = entropy_coeff)
                if approxkl > 0.01: break
            
            lossP, _, _, _, _, _ = model.forward(ent_coef = entropy_coeff)

            lr_scheduler.step(model.backup, approxkl, loss, lossP)

            buffers.clear()

            model.log(summarys, lr_scheduler.lrA, entropy_coeff, step )
            if approxkl <= 0.01: print("KLD {:.3f}, updated full gradient step.".format(approxkl))
            else: print("KLD {:.3f}, early stopping.".format(approxkl))
            print("STEP", step, "Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f}".format(
                loss, lA, lC, ratio, pg_loss, grad))

        if (step+1) % 1280 == 0:
            dl.reset(buffers)
        if (step + 1) % SAVE_STEP == 0:
            model.save(name = f'./model/mtl_model_{tag}/')


