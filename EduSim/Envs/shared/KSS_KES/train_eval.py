# coding: utf-8
# 2020/5/8 @ tongshiwei

import logging
from EduSim.SimOS import train_eval
from EduSim.utils import board_episode_callback
from tensorboardX import SummaryWriter


def kss_kes_train_eval(agent, env, max_steps: int = None, max_episode_num: int = None, n_step=False,
                       train=False,
                       logger=logging, level="episode", board_dir=None):
    def summary_callback(rewards, infos, logger):
        expected_reward = sum(rewards) / len(rewards)

        logger.info("Expected Reward: %s" % expected_reward)

        return expected_reward, infos

    sw = None
    if board_dir is not None:
        sw = SummaryWriter(board_dir)

        def episode_callback(episode, reward, *args):
            return board_episode_callback(episode, reward, sw)

    else:
        episode_callback = None

    train_eval(
        agent, env,
        max_steps, max_episode_num, n_step, train,
        logger, level,
        episode_callback=episode_callback,
        summary_callback=summary_callback,
    )

    sw.close()
