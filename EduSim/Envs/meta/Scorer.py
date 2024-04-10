# coding: utf-8


class Scorer(object):
    def __call__(self, user, item, *args, **kwargs) -> ...:
        raise NotImplementedError


class RealScorer(Scorer):
    def answer_scoring(self, user_response, item_truth, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, user_response, item_truth, *args, **kwargs):
        return self.answer_scoring(user_response, item_truth, *args, **kwargs)


class RealChoiceScorer(RealScorer):
    def answer_scoring(self, user_response, item_truth, *args, **kwargs):
        return user_response == item_truth


class TraitScorer(Scorer):
    def __init__(self):
        super(TraitScorer, self).__init__()

    def response_function(self, user_trait, item_trait, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, user_trait, item_trait, *args, **kwargs):
        return self.response_function(user_trait, item_trait, *args, **kwargs)
