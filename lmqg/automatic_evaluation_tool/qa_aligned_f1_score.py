from itertools import product, chain
from statistics import mean
import numpy as np

EPS = 1e-6


def get_score(base_metric, language: str = 'en'):
    if base_metric == 'bertscore':
        from lmqg.automatic_evaluation_tool.bertscore import BERTScore
        return BERTScore(language=language)
    if base_metric == 'moverscore':
        from lmqg.automatic_evaluation_tool.moverscore import MoverScore
        return MoverScore(language=language)
    raise ValueError(f'unknown metric {base_metric}: metric should be `moverscore`/`bertscore`')


class QAAlignedF1Score:

    def __init__(self,
                 language: str = 'en',
                 aggregation: str = 'macro',
                 base_metric: str = 'bertscore',
                 separator_dataset: str = " | ",
                 separator_prediction: str = None):
        self.language = language
        self.aggregation = aggregation
        self.base_metric = get_score(base_metric, language)
        self.separator_dataset = separator_dataset
        self.separator_prediction = separator_dataset if separator_prediction is None else separator_prediction

    def get_score(self, hyps, refs):
        pairs = list(chain(*[list(product(h, r)) for h, r in zip(hyps, refs)]))
        _hyps, _refs = list(zip(*pairs))
        scores = self.base_metric.get_score(_hyps, _refs)
        assert len(scores) == len(pairs), f"{len(scores)} != {len(pairs)}"
        pair_score = {f"{h}--{r}": s for (h, r), s in zip(pairs, scores)}
        output = []
        for hyp, ref in zip(hyps, refs):
            precision = mean(max(pair_score[f"{h}--{r}"] for r in ref) for h in hyp)
            recall = mean(max(pair_score[f"{h}--{r}"] for h in hyp) for r in ref)
            f1 = 2 * precision * recall / (precision + recall + EPS)
            output.append({"f1": f1, "precision": precision, "recall": recall})
        return output

    def compute_score(self, gts, res, return_precision_recall: bool = False):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode().split(self.separator_dataset) for _id in _ids]
        refs = [gts[_id][0].decode().split(self.separator_dataset) for _id in _ids]
        _score = self.get_score(hyps, refs)
        if self.aggregation == 'macro':
            f1 = mean(i['f1'] for i in _score)
        elif self.aggregation == 'micro':
            recall = mean(i['recall'] for i in _score)
            precision = mean(i['precision'] for i in _score)
            f1 = 2 * precision * recall / (precision + recall + EPS)
        else:
            raise ValueError(f"invalid aggregation: {self.aggregation}")
        if return_precision_recall:
            return f1, np.array([i['f1'] for i in _score]), np.array([i['precision'] for i in _score]), np.array([i['recall'] for i in _score])
        return f1, np.array([i['f1'] for i in _score])

    @staticmethod
    def method():
        return "QAAlignedF1Score"
