import tensorflow as tf
import numpy as np
import os
from baselines.ppo.layer import *
from baselines.ppo.mtl_model_PPO import Model as MTLModel
from baselines.ppo.replaybuffer import MultiTaskReplayBuffer
from baselines.ppo.utils import AdaptiveLR
import veca.gym
import random
from baselines.ppo.dataloader import MultiTaskDataLoader


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
    HORIZON = 1280
    GAMMA = 0.99
    LAMBDA = 0.95
    NUM_CHUNKS = 4
    TIME_STEP = 128

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

    dl = MultiTaskDataLoader(envs)
    buffers = MultiTaskReplayBuffer(len(envs), buffer_length=BUFFER_LENGTH, timestep=TIME_STEP)

    dl.reset(buffers)

    obs_sample = buffers.sample_batch()

    model = MTLModel(envs, sess, obs_sample, tag)

    lr_scheduler = AdaptiveLR(schedule = True)
    frac = 1.
    entropy_coeff = 0.01 * frac
    
    obs, reward, done, infos = dl.sample(buffers)

    for step in range(TRAIN_STEP):
        actions = model.get_action(obs)
        obs, reward, done, infos = dl.step(actions,buffers)

        if (step+1) % TIME_STEP == 0:
            batches = buffers.sample_batch()

            summarys = model.feed_batch(batches)
            loss, loss_agent, loss_critic, ratio, pg_loss, grad = model.forward(ent_coef = entropy_coeff)
            
            for idx in range(TRAIN_LOOP):
                approxkl = model.optimize_step(lr = lr_scheduler.lrA / (TRAIN_LOOP), ent_coef = entropy_coeff)
                if approxkl > 0.01: break
            
            lossP, _, _, _, _, _ = model.forward(ent_coef = entropy_coeff)

            lr_scheduler.step(model.backup, approxkl, loss, lossP)

            buffers.clear()

            model.log(summarys, lr_scheduler.lrA, entropy_coeff, step)
            if approxkl <= 0.01: print("KLD {:.3f}, updated full gradient step.".format(approxkl))
            else: print("KLD {:.3f}, early stopping.".format(approxkl))
            print("STEP", step, "Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f}".format(
                loss, loss_agent, loss_critic, ratio, pg_loss, grad))

        if (step+1) % HORIZON == 0:
            dl.reset(buffers)
        if (step + 1) % SAVE_STEP == 0:
            model.save(name = f'./model/mtl_model_{tag}/')


