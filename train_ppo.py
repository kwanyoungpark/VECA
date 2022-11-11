import tensorflow as tf
import numpy as np
import os
from baselines.ppo.layer import *
from baselines.ppo.mtl_model_PPO import Model as MTLModel
from baselines.ppo.replaybuffer import MultiTaskReplayBuffer
from baselines.ppo.utils import AdaptiveLR, Saver
import veca.gym
import random
import time,argparse
from baselines.ppo.dataloader import MultiTaskDataLoader
from baselines.ppo.curriculum import Curriculum


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='VECA Navigation')
    parser.add_argument('--stage1', type=int, 
                        help='stage 1 init step', required = True)
    parser.add_argument('--stage2', type=int, 
                        help='stage 2 init step', required = True)
    parser.add_argument('--tag', type=str, default = "PPO_COGNIANav")
    parser.add_argument('--gpuno', type=int, default = 0)
    parser.add_argument('--port', type=int, default = 10008)

    args = parser.parse_args()
    tag = args.tag
    PORT = args.port

    num_envs = 1
    TRAIN_STEP = 8000000
    SAVE_STEP = 100_000
    REC_STEP = 100000
    NUM_CHUNKS = 4
    NUM_UPDATE = 1
    TRAIN_LOOP = 4
    TIME_STEP = 128
    BUFFER_LENGTH = 128
    HORIZON = 1280
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
                args = ["--train"],                   # VECA task additional arguments. Append "--help" to list valid arguments.
                seeds = random.sample(range(0, 2000),num_envs ),    # seeds per env instances
                remote_env = False                                  # Whether to use the Environment Orchestrator process at a remote server.
            )]

    class TensorboardLogger:
        def __init__(self, sess, summary_mt, summarys, logdir):
            self.sess = sess
            self.merge = tf.summary.merge(summary_mt) #self.summary())
            self.merge_models = [tf.summary.merge(summary) for summary in summarys]
            self.writer = tf.summary.FileWriter(logdir, self.sess.graph)
        def log(self, feed_dict_mt, feed_dicts, global_step):
            summary = self.sess.run(self.merge, feed_dict = feed_dict_mt) #{)
            self.writer.add_summary(summary, global_step)
            for summary_dict, merge in zip(feed_dicts,self.merge_models):
                summary = self.sess.run(merge, feed_dict = summary_dict)
                self.writer.add_summary(summary, global_step)

    dl = MultiTaskDataLoader(envs, curriculum = Curriculum(STAGE1,STAGE2))
    buffers = MultiTaskReplayBuffer(len(envs), buffer_length=BUFFER_LENGTH, timestep=TIME_STEP)

    dl.reset(buffers)

    obs_sample = buffers.sample_batch()
    model = MTLModel(envs, sess, obs_sample, tag)

    result_dir = os.path.join("work_dir", tag)

    logger = TensorboardLogger(sess, model.summary(), [submodel.summary() for submodel in model.models], logdir=result_dir)
    lr_scheduler = AdaptiveLR(schedule = True)

    saver = Saver(sess)
    saver.load_if_exists(ckpt_dir = result_dir)
    
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

            logger.log({model.summary_dict["lr"]:lr_scheduler.lrA, model.summary_dict["ent_coeff"]:entropy_coeff}, summarys,step)
            #model.log(summarys, lr_scheduler.lrA, entropy_coeff, step)
            if approxkl <= 0.01: print("KLD {:.3f}, updated full gradient step.".format(approxkl))
            else: print("KLD {:.3f}, early stopping.".format(approxkl))
            print("STEP", step, "Loss {:.5f} Aloss {:.5f} Closs {:.5f} Maximum ratio {:.5f} pg_loss {:.5f} grad {:.5f}".format(
                loss, loss_agent, loss_critic, ratio, pg_loss, grad))
        if (step+1) % HORIZON == 0:
            dl.reset(buffers)
        if step % SAVE_STEP == 0:
            saver.save(ckpt_dir = result_dir, global_step = step )


