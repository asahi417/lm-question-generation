#!/usr/bin/env python
# 
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
import numpy as np

from .bleu_scorer import BleuScorer
from ..text_normalization import text_normalization


class Bleu:
    def __init__(self, n=4, normalize_hypothesis=True):
        self._n = n
        self._normalize_hypothesis = normalize_hypothesis
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        bleu_scorer = BleuScorer(n=self._n)
        for _id in gts.keys():
            hypo = res[_id]
            ref = gts[_id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)
            hypo = hypo[0]
            if self._normalize_hypothesis:
                if type(hypo) == str:
                    hypo = hypo.encode()
                assert type(hypo) == bytes, f"{hypo} ({type(hypo)})"
                hypo = text_normalization(hypo.decode()).encode('utf-8')
            bleu_scorer += (hypo, ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)

        return score, scores

    def get_score(self, hyps, refs):
        assert len(hyps) == len(refs), f"{len(hyps)} != {len(refs)}"
        bleu_scorer = BleuScorer(n=self._n)
        for h, r in zip(hyps, refs):
            assert type(h) is str, h
            r = r if type(r) is list else [r]
            r = [_r.encode() if type(_r) is str else _r for _r in r]
            h = h.encode() if type(h) is str else h
            if self._normalize_hypothesis:
                assert type(h) == bytes, f"{h} ({type(h)})"
                h = text_normalization(h.decode()).encode('utf-8')
            bleu_scorer += (h, r)
        _, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        return np.array(scores)

    @staticmethod
    def method():
        return "Bleu"
