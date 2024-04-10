# coding: utf-8

from EduSim.SimOS import RandomAgent
import numpy as np

class KSSAgent(RandomAgent):
    def __init__(self,action_space, knowledge2item):
        super(KSSAgent, self).__init__(action_space)
        self.knowledge2item = knowledge2item

    def step(self):
        knowledge = self.action_space.sample()
        item = np.random.choice(self.knowledge2item[knowledge],1)[0].id

        return item


