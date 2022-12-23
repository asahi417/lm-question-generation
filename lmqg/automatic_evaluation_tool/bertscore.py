import os
import numpy as np
from .bert_score import score
BERTSCORE_BATCH = int(os.getenv('BERTSCORE_BATCH', '32'))


class BERTScore:

    def __init__(self, language: str = 'en'):
        self.language = language

    def get_score(self, hyps, refs):
        return np.array(score(hyps, refs, lang=self.language, verbose=True, batch_size=BERTSCORE_BATCH)[0].tolist())

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode() for _id in _ids]
        refs = [gts[_id][0].decode() for _id in _ids]
        _score = self.get_score(hyps, refs)
        return np.mean(_score), _score

    @staticmethod
    def method():
        return "BERTScore"
