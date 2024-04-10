# coding: utf-8
from copy import deepcopy
import networkx as nx
import random
from EduSim.Envs.meta import Env

import numpy as np
from EduSim.Envs.KSS.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.spaces import ListSpace
from .meta import KSSItemBase, KSSScorer
from .utils import load_environment_parameters
import math
import torch

import sys

sys.path.append('../')
from KT import Agent_KT

__all__ = ["KSSEnv"]


class KSSEnv(Env):
    def __init__(self, seed=None, initial_step=None):
        self.random_state = np.random.RandomState(seed)

        parameters = load_environment_parameters()
        self.knowledge_structure = parameters["knowledge_structure"]
        self.know_item = parameters["know_item"]
        self._item_base = KSSItemBase(
            parameters["knowledge_structure"],
            parameters["learning_order"],
            items=parameters["items"]
        )
        self.learning_item_base = deepcopy(self._item_base)
        self.learning_item_base.drop_attribute()
        self.test_item_base = self._item_base
        self.scorer = KSSScorer(parameters["configuration"].get("binary_scorer", True))

        self.action_space = ListSpace(self.learning_item_base.know_id_list, seed=seed)

        self.learners = LearnerGroup(self.knowledge_structure, seed=seed)

        self._order_ratio = parameters["configuration"]["order_ratio"]
        self._review_times = parameters["configuration"]["review_times"]
        self._learning_order = parameters["learning_order"]

        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        self._initial_step = parameters["configuration"]["initial_steps"] if initial_step is None else initial_step

        self._learner = None
        self._initial_score = None
        self._exam_reduce = "sum" if parameters["configuration"].get("exam_sum", True) else "ave"

        self.KT_model = Agent_KT()
        self.mastery_thresh_hold = 0.6

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
            "learning_item_base": self.learning_item_base
        }

    def _initial_logs(self, learner: Learner):
        logs = []

        while len(logs) < self._initial_step:
            if random.random() < 0.7:
                for knowledge in self._learning_order:
                    test_item_id = self.random_state.choice(self.test_item_base.knowledge2item[knowledge]).id # todo： 在该知识点下随机选择一个题目，不同学生出现差异
                    if learner.response(self.test_item_base[test_item_id]) < 0.6:
                        break
                    else:
                        break
                learning_item_id = test_item_id
            else:
                learning_item_id = self.learning_item_base.index[str(self.random_state.randint(0,49))].id

            item_id, score = self.learn_and_test(learner, learning_item_id)
            logs.append([item_id, score])

        learner.update_logs(logs)


    def learn_and_test(self, learner: Learner, item_id):
        learning_item = self.learning_item_base[item_id]
        learner.learn(learning_item)
        test_item_id = item_id
        test_item = self.test_item_base[test_item_id]
        score = self.scorer(learner.response(test_item), test_item.difficulty)
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce=None) -> (dict, int, float):
        if reduce is None:
            reduce = self._exam_reduce
        knowledge_response = {}
        for test_knowledge in learner.target:
            item = self.test_item_base.knowledge2item[test_knowledge][3]
            knowledge_response[test_knowledge] = [item.id, self.scorer(learner.response(item), item.difficulty)]
        if detailed:
            return knowledge_response
        elif reduce == "sum":
            return np.sum([v for _, v in knowledge_response.values()])
        elif reduce in {"mean", "ave"}:
            return np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        self._initial_logs(self._learner)
        self._initial_score = self._exam(self._learner)
        return self._learner.profile, self._exam(self._learner, detailed=True)

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        self._learner = None

        return observation, reward, done, info

    def retrive_state(self, state, relation):
        know_state = list()
        for know in relation.keys():
            know_state.append(torch.gather(state.squeeze(0), 0, torch.tensor(relation[know])).mean())

        know_state = torch.tensor(know_state)

        return know_state.unsqueeze(0)

    def step(self, ques_H, ans_H, learning_item_id, *args, **kwargs):
        a = self._exam(self._learner)
        observation = self.learn_and_test(self._learner, learning_item_id)
        b = self._exam(self._learner)
        ques_H.append(int(observation[0]))
        ans_H.append(int(observation[1]))
        state = self.KT_model.forward_state(ques_H,ans_H)
        state = state[:, -1, :]
        return state, observation, float(b - a)/len(self._learner.target), b == len(self._learner.target), ques_H, ans_H


    def step_l(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, 0
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}

        return observation, reward, done, info

    def step_p(self, ques_H, ans_H, practice_item_id, *args, **kwargs):
        observation = self.learn_and_test(self._learner, practice_item_id)
        last_diff =  self.learning_item_base.index[str(ques_H[-1])].difficulty
        cur_diff = self.learning_item_base.index[str(practice_item_id)].difficulty
        ques_H.append(int(observation[0]))
        ans_H.append(int(observation[1]))
        kc = self.learning_item_base.index[str(practice_item_id)].knowledge
        state = self.KT_model.forward_state(ques_H,ans_H)
        state = state[:, -1, :]
        know_state = self.retrive_state(state, self.know_item)
        return state, observation, -math.pow((cur_diff - last_diff), 5), (float(know_state.squeeze(0)[kc]) > self.mastery_thresh_hold), ques_H, ans_H, cur_diff


    def step_fordata(self, practice_item_id, *args, **kwargs):
        a = self._exam(self._learner)
        observation = self.learn_and_test(self._learner, practice_item_id)
        b = self._exam(self._learner)
        return observation, float(b - a)/len(self._learner.target), b == len(self._learner.target)

    def n_step(self, learning_path, *args, **kwargs):
        exercise_history = []
        a = self._exam(self._learner)
        for learning_item_id in learning_path:
            item_id, score = self.learn_and_test(self._learner, learning_item_id)
            exercise_history.append([item_id, score])
        b = self._exam(self._learner)
        return exercise_history, b - a, b == len(self._learner.target), None

    def reset(self):
        self._learner = None


    def render(self, mode='human'):
        if mode == "log":
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )

    def encoder_state(self,observation):
        emb = np.zeros(2*len(self.action_space))
        item = int(observation[0])-1
        score = int(observation[1])
        if score == 1:
            emb[item] = 1
        else:
            emb[item + len(self.action_space)] = 1

        return emb