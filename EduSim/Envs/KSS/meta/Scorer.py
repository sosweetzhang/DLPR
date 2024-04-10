# coding: utf-8

from EduSim.utils import irt
from EduSim.Envs.meta import TraitScorer


class KSSScorer(TraitScorer):
    def __init__(self, binary_scorer=True):
        super(KSSScorer, self).__init__()
        self._binary = binary_scorer

    def response_function(self, user_trait, item_trait, binary=None, *args, **kwargs):
        _score = irt(user_trait, item_trait)
        binary = self._binary if binary is None else binary
        if binary:
            return 1 if _score >= 0.5 else 0
        else:
            return _score


class KTscorer(TraitScorer):
    def __init__(self, binary_scorer=True):
        super(KTscorer, self).__init__()
        self._binary = binary_scorer

    def response_function(self, user_trait, item_trait, binary=None, *args, **kwargs):
        _score = irt(user_trait, item_trait["difficulty"])
        binary = self._binary if binary is None else binary
        if binary:
            return 1 if _score >= 0.5 else 0
        else:
            return _score
