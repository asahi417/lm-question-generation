import logging
from typing import List
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
                 # aggregation: str = 'macro',
                 base_metric: str = 'bertscore',
                 instance_separator: str = " | ",
                 question_key: str = 'question: ',
                 answer_key: str = 'answer:',
                 qa_separator: str = ', '):
        self.language = language
        # self.aggregation = aggregation
        self.base_metric = get_score(base_metric, language)
        self.instance_separator = instance_separator
        self.question_key = question_key
        self.answer_key = answer_key
        self.qa_separator = qa_separator

    def sanity_check(self, sample: str):
        qa_pair = sample.split(self.qa_separator + self.answer_key)
        if len(qa_pair) != 2:
            logging.info(f'error (length != 2): {sample}')
            return False
        q, a = qa_pair
        if not q.startswith(self.question_key):
            logging.info(f'error (question not found): {sample}')
            return False
        if not a.startswith(self.answer_key):
            logging.info(f'error (answer not found): {sample}')
            return False
        return True

    def filter_qa_pairs(self, qa_pairs):
        qa_pairs = list(set(qa_pairs))  # remove duplication
        qa_pairs = filter(lambda x: self.sanity_check(x), qa_pairs)
        return list(qa_pairs)

    def get_score(self, hyps: List, refs: List):
        hyps = [self.filter_qa_pairs(hyp) for hyp in hyps]
        pairs = list(chain(*[list(product(h, r)) for h, r in zip(hyps, refs) if len(h) != 0]))
        scores = self.base_metric.get_score(*list(zip(*pairs)))
        assert len(scores) == len(pairs), f"{len(scores)} != {len(pairs)}"
        pair_score = {f"{h}--{r}": s for (h, r), s in zip(pairs, scores)}
        output = []
        for hyp, ref in zip(hyps, refs):
            if len(hyp) == 0:
                output.append({"f1": 0, "precision": 0, "recall": 0})
            else:
                precision = mean(max(pair_score[f"{h}--{r}"] for r in ref) for h in hyp)
                recall = mean(max(pair_score[f"{h}--{r}"] for h in hyp) for r in ref)
                f1 = 2 * precision * recall / (precision + recall + EPS)
                output.append({"f1": f1, "precision": precision, "recall": recall})
        return output

    def compute_score(self, gts, res, return_precision_recall: bool = False):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode().split(self.instance_separator) for _id in _ids]
        refs = [gts[_id][0].decode().split(self.instance_separator) for _id in _ids]
        _score = self.get_score(hyps, refs)
        # if self.aggregation == 'macro':
        f1 = np.array([i['f1'] for i in _score])
        # elif self.aggregation == 'micro':
        #     recall = mean(i['recall'] for i in _score)
        #     precision = mean(i['precision'] for i in _score)
        #     f1 = 2 * precision * recall / (precision + recall + EPS)
        # else:
        #     raise ValueError(f"invalid aggregation: {self.aggregation}")
        if return_precision_recall:
            return np.mean(f1), f1,\
                   np.array([i['precision'] for i in _score]),\
                   np.array([i['recall'] for i in _score])
        return np.mean(f1), f1

    @staticmethod
    def method():
        return "QAAlignedF1Score"
