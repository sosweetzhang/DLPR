# coding: utf-8
from tensorboardX import SummaryWriter

__all__ = ["SummaryWriter", "board_episode_callback", "reward_summary_callback", "get_board_episode_callback"]


def get_board_episode_callback(board_dir=None, sw=None):
    if board_dir is not None or sw is not None:
        if sw is None:
            sw = SummaryWriter(board_dir)

        def episode_callback(episode, reward, *args):
            return board_episode_callback(episode, reward, sw)

    else:
        episode_callback = None

    return sw, episode_callback


def board_episode_callback(episode, reward, summary_writer: SummaryWriter):
    summary_writer.add_scalar(tag="episode_reward", scalar_value=reward, global_step=episode)


def reward_summary_callback(rewards, infos, logger):
    expected_reward = sum(rewards) / len(rewards)

    logger.info("Expected Reward: %s" % expected_reward)

    return expected_reward, infos
