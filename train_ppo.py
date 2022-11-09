import tensorflow as tf
import numpy as np
import os
from baselines.ppo.utils import *
from baselines.ppo.mtl_model_PPO import Model as MTLModel
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
    heads = [HeadQuarter(env = env, bufferlength = BUFFER_LENGTH, timestep = TIME_STEP) for env in envs]

    [head.restart() for head in heads]
    head = heads[0]
    head.step(np.random.rand(num_envs, envs[0].action_space))
    obs_sample = head.get_batch(num = TIME_STEP)
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
    
    obs, reward, done, infos = zip(*[head.step(np.random.rand(num_envs, head.env.action_space)) for head in heads])

    for step in range(TRAIN_STEP):
        actions = model.get_action(obs)
        obs, reward, done, infos = zip(*[head.step(actions[i]) for i, head in enumerate(heads)])

        if (step+1) % TIME_STEP == 0:
            print("Training Actor & Critic...")

            summarys = model.make_batch(heads)
            loss, lA, lC, ratio, pg_loss, grad = model.forward(ent_coef = entropy_coeff)
            print("STEP", step, "Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f}".format(
                loss, lA, lC, ratio, pg_loss, grad))

            for idx in range(TRAIN_LOOP):
                approxkl = model.optimize_step(lr = lr_scheduler.lrA / (TRAIN_LOOP), ent_coef = entropy_coeff)
                if approxkl > 0.01: break

            if approxkl <= 0.01:
                print("KLD {:.3f}, updated full gradient step.".format(approxkl))
            else:
                print("KLD {:.3f}, early stopping.".format(approxkl))
            
            lossP, _, _, _, _, _ = model.forward(ent_coef = entropy_coeff)

            lr_scheduler.step(model.backup, approxkl, loss, lossP)

            model.log(summarys, lr_scheduler.lrA, entropy_coeff, step )

            [head.replayBuffer.clear() for head in heads]
        if (step+1) % 1280 == 0:
            [head.restart() for head in heads]
        if (step + 1) % SAVE_STEP == 0:
            model.save(name = f'./model/mtl_model_{tag}/')


