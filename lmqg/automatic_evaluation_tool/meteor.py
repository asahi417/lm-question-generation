"""TODO: implement python version of meteor
https://stackoverflow.com/questions/63778133/how-can-i-implement-meteor-score-when-evaluating-a-model-when-using-the-meteor-s"""
import numpy as np
from nltk.translate.meteor_score import meteor_score


class METEOR:

    def __init__(self, normalize_hypothesis=True):
        self.normalize_hypothesis = normalize_hypothesis

    def get_score(self, hyps, refs):
        return np.array([meteor_score(r, h) for r, h in zip(hyps, refs)])

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode() for _id in _ids]
        refs = [gts[_id][0].decode() for _id in _ids]
        _score = self.get_score(hyps, refs)
        return np.mean(_score), _score

    @staticmethod
    def method():
        return "METEOR"
