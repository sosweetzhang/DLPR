# coding: utf-8

from pprint import pformat
from gym.spaces import Space

__all__ = ["ListSpace"]


class ListSpace(Space):
    def __init__(self, elements: list, seed=None):
        self.elements = elements
        super(ListSpace, self).__init__(shape=(len(self.elements),))
        self.seed(seed)
        self.n = self.__len__()

    def sample(self):
        return self.np_random.choice(self.elements)

    def sample_idx(self):
        return self.np_random.choice(list(range(len(self.elements))))

    def contains(self, item):
        return item in self.elements

    def __repr__(self):
        return pformat(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    def __len__(self):
        return len(self.elements)
