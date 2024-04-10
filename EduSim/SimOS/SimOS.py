# coding: utf-8

from itertools import cycle
from gym.spaces import Space
import logging

from EduSim.Envs.meta.Env import Env
from EduSim.utils.callback import get_board_episode_callback, reward_summary_callback
from .config import as_level
from longling.ML.toolkit.monitor import ConsoleProgressMonitor, EMAValue


class MetaAgent(object):
    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError

    def end_episode(self, observation, reward, done, info):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def tune(self, *args, **kwargs):
        raise NotImplementedError

    def n_step(self):
        raise NotImplementedError


class RandomAgent(MetaAgent):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def begin_episode(self, *args, **kwargs):
        pass

    def end_episode(self, observation, reward, done, info):
        pass

    def observe(self, observation, reward, done, info):
        pass

    def step(self):
        return self.action_space.sample()

    def n_step(self, max_steps: int):
        return [self.step() for _ in range(max_steps)]

    def tune(self, *args, **kwargs):
        pass




def meta_train_eval(agent: MetaAgent, env: Env, max_steps: int = None, max_episode_num: int = None, n_step=False,
                    train=False,
                    logger=logging, level="episode", episode_callback=None, summary_callback=None,
                    values: dict = None, monitor=None):

    episode = 0

    level = as_level(level)

    rewards = []
    infos = []

    if values is None:
        values = {"Episode": EMAValue(["Reward"])}

    if monitor is None:
        if level >= as_level("summary"):
            monitor = ConsoleProgressMonitor(
                indexes={"Episode": ["Reward"]},
                values=values,
                total=max_episode_num,
                player_type="episode"
            )
        else:
            def monitor(x):
                return x

    loop = cycle([1]) if max_episode_num is None else range(max_episode_num)
    for _ in monitor(loop):
        if max_episode_num is not None and episode >= max_episode_num:
            break

        try:
            learner_profile = env.begin_episode()
            agent.begin_episode(learner_profile)
            episode += 1

            if level <= as_level("episode"):
                logger.info("episode [%s]: %s" % (episode, env.render("log")))

        except StopIteration:  # pragma: no cover
            break

        # recommend and learn
        if n_step is True:
            assert max_steps is not None
            # generate a learning path
            learning_path = agent.n_step(max_steps)
            for observation, reward, done, info in env.n_step(learning_path):
                agent.observe(observation, reward, done, info)
                if done:
                    break
        else:
            learning_path = []
            _step = 0

            if max_steps is not None:
                # generate a learning path step by step
                for _ in range(max_steps):
                    try:
                        learning_item = agent.step()
                        learning_path.append(learning_item)
                    except StopIteration:  # pragma: no cover
                        break
                    # observation, reward, done, info = env.step(learning_item)
                    info, observation, reward, done = env.step(learning_item)

                    if level <= as_level("step"):
                        _step += 1
                        logger.debug(
                            "step [%s]: agent -|%s|-> env, env state %s" % (_step, learning_item, env.render("log"))
                        )
                        logger.debug(
                            "step [%s]: observation: %s, reward: %s" % (_step, observation, reward)
                        )

                    agent.observe(observation, reward, done, info)
                    if done:
                        break
            else:
                while True:
                    learning_item = agent.step()
                    observation, reward, done, info = env.step(learning_item)
                    learning_path.append(learning_item)
                    agent.observe(observation, reward, done, info)
                    if done:
                        break

        # test the learner to see the learning effectiveness
        observation, reward, done, info = env.end_episode()
        agent.end_episode(observation, reward, done, info)

        rewards.append(reward)
        infos.append(info)

        if level <= as_level("episode"):
            logger.info("episode [%s] - learning path: %s" % (episode, learning_path))
            logger.info("episode [%s] - total reward: %s" % (episode, reward))
            logger.info("episode [%s]: %s" % (episode, env.render(mode="log")))

        values["Episode"].update("Reward", reward)

        env.reset()

        if episode_callback is not None:
            episode_callback(episode, reward, done, info, logger)

        if train is True:
            agent.tune()

    if summary_callback is not None and level <= as_level("summary"):
        return summary_callback(rewards, infos, logger)


def train_eval(agent: MetaAgent, env: Env, max_steps: int = None, max_episode_num: int = None, n_step=False,
               train=False,
               logger=logging, level="episode", board_dir=None,
               sw=None, episode_callback=None, summary_callback=None,
               *args, **kwargs):


    assert max_episode_num is not None, "infinity environment, max_episode_num should be set"

    if episode_callback is None:
        sw, episode_callback = get_board_episode_callback(board_dir, sw=sw)

    if summary_callback is None:
        summary_callback = reward_summary_callback

    meta_train_eval(
        agent, env,
        max_steps, max_episode_num, n_step, train,
        logger, level,
        episode_callback=episode_callback,
        summary_callback=summary_callback,
        *args, **kwargs
    )

    if board_dir:
        sw.close()
