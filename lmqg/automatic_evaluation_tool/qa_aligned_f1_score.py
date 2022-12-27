import logging
from typing import List
from itertools import product, chain
from statistics import mean
import numpy as np
from lmqg.spacy_module import SpacyPipeline

EPS = 1e-6


def get_score(base_metric, language: str = 'en'):
    if base_metric == 'bertscore':
        from .bertscore import BERTScore
        return BERTScore(language=language)
    if base_metric == 'moverscore':
        from .moverscore import MoverScore
        return MoverScore(language=language)
    raise ValueError(f'unknown metric {base_metric}: metric should be `moverscore`/`bertscore`')


class QAAlignedF1Score:

    def __init__(self,
                 language: str = 'en',
                 target_metric: str = 'f1',
                 base_metric: str = 'bertscore',
                 instance_separator: str = " | ",
                 question_key: str = 'question: ',
                 answer_key: str = 'answer: ',
                 qa_separator: str = ', '):
        self.language = language
        self.base_metric = get_score(base_metric, language)
        self.instance_separator = instance_separator
        self.question_key = question_key
        self.answer_key = answer_key
        # pipe = SpacyPipeline(self.language)
        # self.question_key = (pipe.token(question_key))
        # self.answer_key = pipe.token(answer_key)
        self.qa_separator = qa_separator
        assert target_metric in ['f1', 'recall', 'precision'], target_metric
        self.target_metric = target_metric

    def sanity_check(self, sample: str):
        if len(sample.split(self.qa_separator + self.answer_key)) != 2:
            logging.info(f'error (length != 2): {sample}')
            return False
        if not sample.startswith(self.question_key):
            logging.info(f'error (question not found): {sample}')
            return False
        return True

    def filter_qa_pairs(self, qa_pairs):
        qa_pairs = list(set(qa_pairs))  # remove duplication
        qa_pairs = filter(lambda x: self.sanity_check(x), qa_pairs)
        return list(qa_pairs)

    def get_score(self, hyps: List, refs: List):
        hyps = [h.split(self.instance_separator) for h in hyps]
        refs = [r.split(self.instance_separator) for r in refs]
        hyps = [self.filter_qa_pairs(hyp) for hyp in hyps]
        logging.info(f"found {len([i for i in hyps if len(i) == 0])} empty prediction from {len(hyps)}")
        pairs = list(chain(*[list(product(h, r)) for h, r in zip(hyps, refs) if len(h) != 0]))
        if len(pairs) == 0:
            return np.array([0.0])
        h, r = list(zip(*pairs))
        scores = self.base_metric.get_score(h, r)
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
        output = np.array([o[self.target_metric] for o in output])
        return output

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        _ids = gts.keys()
        hyps = [res[_id][0].decode() for _id in _ids]
        refs = [gts[_id][0].decode() for _id in _ids]
        _score = self.get_score(hyps, refs)
        return np.mean(_score), _score

    @staticmethod
    def method():
        return "QAAlignedF1Score"
