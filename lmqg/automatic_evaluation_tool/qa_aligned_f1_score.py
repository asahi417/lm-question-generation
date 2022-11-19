import os
import numpy as np


def get_score(base_metric, language: str = 'en'):
    if base_metric == 'bertscore':
        from .bertscore import BERTScore
        return BERTScore(language=language)
    if base_metric == 'moverscore':
        from .moverscore import MoverScore
        return MoverScore(language=language)
    if base_metric == 'rouge':
        from .rouge import Rouge
        return Rouge()
    if base_metric == 'meteor':
        from .meteor.meteor import Meteor
        return Meteor()
    if base_metric == 'bleu':
        from .bleu.bleu import Bleu
        return Bleu(4)
    raise ValueError(f'unknown metric {base_metric}')


class QAAlignedF1Score:

    def __init__(self,
                 language: str = 'en',
                 base_metric: str = 'bertscore',
                 separator: str = None):
        self.language = language
        self.base_metric = get_score(base_metric, language)

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode() for _id in _ids]
        refs = [gts[_id][0].decode() for _id in _ids]
        average_score = np.mean(np.array(_score))
        return average_score, np.array(_score)

    @staticmethod
    def method():
        return "BERTScore"
