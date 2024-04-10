# coding: utf-8
import os
import json
import time
import networkx as nx
import numpy as np
import torch
from EduSim.Envs.KES_Junyi.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.spaces import ListSpace
from EduSim.Envs.meta import Item
from EduSim.Envs.meta import Env
from EduSim.utils import get_proj_path, get_graph_embeddings
# from .meta import KESScorer
from meta import KESScorer

__all__ = ["KESEnv"]


class KESEnv(Env):
    def __init__(self, dataRec_path, seed=None):
        super(KESEnv, self).__init__()
        self.random_state = np.random.RandomState(seed)
        self.graph_embeddings = get_graph_embeddings('KES_junyi')
        self.dataRec_path = dataRec_path

        with open(f'{get_proj_path()}/data/dataProcess/junyi/graph_vertex.json') as f:
            ku_dict = json.load(f)

        with open(f'{get_proj_path()}/data/dataProcess/junyi/prerequisite.json') as f:
            prerequisite_edges = json.load(f)
            self.knowledge_structure = nx.DiGraph()
            # add this line by lqy
            self.knowledge_structure.add_nodes_from(ku_dict.values())
            self.knowledge_structure.add_edges_from(prerequisite_edges)
            self._topo_order = list(nx.topological_sort(self.knowledge_structure))
            assert not list(nx.algorithms.simple_cycles(self.knowledge_structure)), "loop in DiGraph"



        self.num_skills = len(ku_dict)
        self.max_sequence_length = 300
        self.feature_dim = 2 * self.num_skills
        self.embed_dim = 600
        self.hidden_size = 900
        self.item_list = [i for i in range(len(ku_dict))]
        self.learning_item_base = [Item(item_id=i, knowledge=i) for i in self.item_list]

        # KTnet
        KT_input_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.0,
        }
        self.KTnet = torch.load("env_KT.pt")
        self.scorer = KESScorer()
        self.action_space = ListSpace(self.item_list, seed=seed)

        # learners
        self.learners = LearnerGroup(self.dataRec_path, seed=seed)
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
        self.type = 'KES_junyi'
        self.env_name = 'KESjunyi'

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def learn_and_test(self, learner: Learner, item_id):
        state = learner.state
        score = self.scorer.response_function(state, item_id)
        learner.learn(item_id, score)
        self.update_learner_state()
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        state = learner.state
        knowledge_response = {}  # dict
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, self.scorer.response_function(state, test_item)]
        if detailed:
            return_thing = knowledge_response
        elif reduce == "sum":
            return_thing = np.sum([v for _, v in knowledge_response.values()])  # np.sum   []:list   knowledge_response
        elif reduce in {"mean", "ave"}:
            return_thing = np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)  # unknown reduce type
        return return_thing

    def update_learner_state(self):
        logs = self._learner.profile['logs']
        sequence_length = len(logs)
        input_data = self.get_feature_matrix(logs).unsqueeze(0)  # [bz,sequence_length,feture_dim]
        self._learner._state = torch.Sigmoid(
            self.KTnet(input_data).permute(1, 0, 2).squeeze(0)[sequence_length - 1])

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)  # learner（learning target、state、knowledge_structure）
        self.update_learner_state()
        self._initial_score = self._exam(self._learner)  # learner initial_score
        while self._initial_score >= len(self._learner.target):
            self._learner = next(self.learners)  # learner（learning target、state、knowledge_structure）
            self.update_learner_state()
            self._initial_score = self._exam(self._learner)  # learner initial_score
        return self._learner.profile, self._exam(self._learner, detailed=True)  # learner profile id、logs、target

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        self.episode_end_time = time.time()
        # print('episode_env_time:' + str(self.episode_end_time - self.episode_start_time))
        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):  # point-wise
        a = self._exam(self._learner)
        observation = self.learn_and_test(self._learner, learning_item_id)
        b = self._exam(self._learner)
        return observation, b - a, b == len(self._learner.target), None  # reward is the promotion

    def n_step(self, learning_path, *args, **kwargs):  # sequence-wise
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
            return_thing = "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )
        else:
            return_thing = 'for else return'
        return return_thing

    def get_feature_matrix(self, session):
        input_data = np.zeros(shape=(max(1, len(session)), self.feature_dim), dtype=np.float32)
        j = 0
        while j < len(session):
            problem_id = session[j][0]
            if session[j][1] == 0:
                input_data[j][problem_id] = 1.0
            elif session[j][1] == 1:
                input_data[j][problem_id + self.num_skills] = 1.0
            j += 1
        return torch.Tensor(input_data)
