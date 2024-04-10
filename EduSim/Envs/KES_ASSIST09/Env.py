# -*- coding:utf-8 _*-
import pickle
import time
import os
import networkx as nx
import numpy as np
import torch
from EduSim.Envs.KES_ASSIST09.meta.Learner import KESASSIST09LeanerGroup
from EduSim.spaces import ListSpace
from EduSim.deep_model import KTnet
from EduSim.Envs.meta import Item
from EduSim.Envs.KES_Junyi import KESEnv
from EduSim.utils import get_proj_path, get_graph_embeddings

from .meta import KESASSISTScorer
__all__ = ["KESASSISTEnv"]


class KESASSISTEnv(KESEnv):
    def __init__(self, dataRec_path, seed=None):
        super().__init__(dataRec_path, seed)
        self.type = 'KES'
        self.env_name = 'KESASSIST09'
        self.random_state = np.random.RandomState(seed)
        self.graph_embeddings = get_graph_embeddings('KES_ASSIST09')
        self.dataRec_path = dataRec_path

        self.knowledge_structure = nx.DiGraph()
        with open(f"{get_proj_path()}/data/dataProcess/ASSIST09/nxgraph.pkl", "rb") as file:
            self.knowledge_structure = pickle.loads(file.read())

        self._topo_order = list(nx.topological_sort(self.knowledge_structure))

        assert not list(nx.algorithms.simple_cycles(self.knowledge_structure)), "loop in DiGraph"


        self.num_skills = 97
        self.max_sequence_length = 300
        self.feature_dim = 2 * self.num_skills
        self.embed_dim = 64
        self.hidden_size = 128
        self.item_list = [i for i in range(self.num_skills)]
        self.learning_item_base = [Item(item_id=i, knowledge=i) for i in self.item_list]


        KT_para_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.0,
        }
        self.KTnet = KTnet(KT_para_dict)
        directory = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/meta_data'
        KT_file = f'{directory}/env_weights/ValBest.ckpt'
        # KT_file = f'{directory}/env_weights/SelectedValBest.ckpt'
        if os.path.exists(KT_file):
            param_dict = torch.load(KT_file)
            _, _ = torch.load(self.KTnet, param_dict)
        else:
            raise ValueError('KT net not trained yet!')
        self.scorer = KESASSISTScorer()
        self.action_space = ListSpace(self.item_list, seed=seed)

        # learners
        self.learners = KESASSIST09LeanerGroup(self.dataRec_path, seed=seed)
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
