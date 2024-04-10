# coding: utf-8

import numpy as np
from EduSim.Envs.meta import ItemBase
from networkx import Graph, DiGraph

__all__ = ["KSSItemBase"]


class KSSItemBase(ItemBase):
    def __init__(self, knowledge_structure: (Graph, DiGraph), learning_order=None, items=None, seed=None,
                 reset_attributes=False):
        self.random_state = np.random.RandomState(seed)
        if items is None or reset_attributes:
            assert learning_order is not None

        super(KSSItemBase, self).__init__(
            items, knowledge_structure=knowledge_structure,
        )
        self.knowledge2item = dict()
        for item in self.items:
            if item.knowledge not in self.knowledge2item.keys():
                self.knowledge2item[item.knowledge] = []
            self.knowledge2item[item.knowledge].append(item)