import random
from EduSim.Envs.meta import TraitScorer


class KESScorer(TraitScorer):
    def response_function(self, user_trait, item_trait, *args, **kwargs):
        # return 1 if user_trait[item_trait] >= 0.55 else 0
        # return 1 if user_trait[item_trait] >= 0.67 else 0
        # return 1 if user_trait[item_trait] >= 0.73 else 0
        # return 1 if user_trait[item_trait] >= 0.75 else 0  # original
        return 1 if user_trait[item_trait] >= 0.5 else 0  # original

    def middle_response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if random.random() <= user_trait[item_trait] else 0
