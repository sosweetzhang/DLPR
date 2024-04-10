# -*- coding:utf-8 _*-
import random
from EduSim.Envs.meta import TraitScorer


class KESASSISTScorer(TraitScorer):
    def response_function(self, user_trait, item_trait, *args, **kwargs):
        # return 1 if user_trait[item_trait] >= 0.75 else 0
        # return 1 if user_trait[item_trait] >= 0.67 else 0
        # return 1 if user_trait[item_trait] >= 0.6 else 0
        return 1 if user_trait[item_trait] >= 0.5 else 0

    def middle_response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if random.random() <= user_trait[item_trait] else 0
