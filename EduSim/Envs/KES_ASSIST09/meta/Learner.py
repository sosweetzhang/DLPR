# -*- coding:utf-8 _*-
import os
import copy
import numpy as np
from EduSim.Envs.KES_Junyi.meta.Learner import LearnerGroup as KESLearnerGroup
from EduSim.Envs.KES_Junyi.meta.Learner import Learner


class KESASSIST09LeanerGroup(KESLearnerGroup):
    def __init__(self, dataRec_path, seed=None):
        super().__init__(dataRec_path, seed)
        self.data_path = dataRec_path
        self.random_state = np.random.RandomState(seed)

    def __next__(self):
        dataset_number = self.random_state.randint(1, 6)
        all_students = os.listdir(f'{self.data_path}{dataset_number}/train/')
        session = [[0, 0]]
        learning_targets = set()
        while len({log[0] for log in session}) <= 6 or len(learning_targets) <= 1:
            index = self.random_state.randint(len(all_students))
            with open(f'{self.data_path}{dataset_number}/train/{index}.csv', 'r') as f:
                data = f.readlines()[1:]
            session = [[int(line.rstrip().split(',')[0]) - 1,
                        int(line.rstrip().split(',')[1])] for i, line in enumerate(data)]
            learning_targets = {step[0] for i, step in enumerate(session) if i >= 0.8 * len(session)}


        initial_log = copy.deepcopy(session[:int(len(session) * 0.6)])

        return Learner(
            initial_log=initial_log,
            learning_target=learning_targets,
        )
