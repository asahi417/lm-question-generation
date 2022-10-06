""" SQuAD QG evaluation (sentence/answer level) """
import json
import logging
import os
from collections import defaultdict
from os.path import join as pj
from typing import Dict, List
import torch
from .automatic_evaluation_tool.bleu.bleu import Bleu
from .automatic_evaluation_tool.meteor.meteor import Meteor
from .automatic_evaluation_tool.rouge import Rouge
from .automatic_evaluation_tool.bertscore import BERTScore
from .automatic_evaluation_tool.moverscore import MoverScore
from .spacy_module import SpacyPipeline
from .language_model import TransformersQG
from .data import get_reference_files, get_dataset

LANG_NEED_TOKENIZATION = ['ja', 'zh']


class QGEvalCap:

    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self, bleu_only: bool = False, language: str = 'en', skip: List = None):
        output = {}
        output_individual = {}
        if bleu_only:
            scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
        else:
            scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"),
                       (BERTScore(language=language), "BERTScore"), (MoverScore(language=language), 'MoverScore')]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            if skip is not None:
                if type(method) == list:
                    if all(m in skip for m in method):
                        continue
                else:
                    if method in skip:
                        continue
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    logging.info("%s: %0.5f" % (m, sc))
                    output[m] = sc
                    output_individual[m] = scs
            else:
                logging.info("%s: %0.5f" % (method, score))
                output[method] = score
                output_individual[method] = scores
                if method in ['BERTScore', 'MoverScore']:
                    torch.cuda.empty_cache()
        return output, output_individual


def compute_metrics(out_file,
                    tgt_file,
                    src_file: str = None,
                    prediction_aggregation: str = 'first',
                    bleu_only: bool = False,
                    language: str = 'en',
                    skip: List = None):
    """ compute automatic metric """
    if language in LANG_NEED_TOKENIZATION:
        spacy_model = SpacyPipeline(language=language)
    else:
        spacy_model = None

    pairs = []

    with open(tgt_file, "r") as infile:
        for n, line in enumerate(infile):
            if line.endswith('\n'):
                line = line[:-1]
            if spacy_model is None:
                pairs.append({'tokenized_question': line.strip(), 'tokenized_sentence': n})
            else:
                pairs.append({'tokenized_question': ' '.join(spacy_model.token(line.strip())), 'tokenized_sentence': n})

    # print(len(pairs), src_file, tgt_file)
    if src_file is not None and prediction_aggregation is not None:
        # group by the source (sentence where the question are produced); this is used for grouping but not evaluation
        with open(src_file, 'r') as infile:
            for n, line in enumerate(infile):
                if line.endswith('\n'):
                    line = line[:-1]
                if spacy_model is None:
                    pairs[n]['tokenized_sentence'] = line.strip().lower()
                else:
                    pairs[n]['tokenized_sentence'] = ' '.join(spacy_model.token(line.strip().lower()))

    # fix prediction's tokenization: lower-casing and detaching sp characters
    with open(out_file, 'r') as infile:
        for n, line in enumerate(infile):
            # print(n, line)
            if line.endswith('\n'):
                line = line[:-1]
            if spacy_model is not None:
                pairs[n]['prediction'] = ' '.join(spacy_model.token(line.strip()))
            else:
                pairs[n]['prediction'] = line.strip()
    # eval
    json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])

    for pair in pairs:

        # key is the sentence where the model generates the question
        key = pair['tokenized_sentence']

        # one generation per sentence
        if 'prediction' in pair:
            pred = pair['prediction'].encode('utf-8')
        else:
            logging.warning('prediction not found at the evaluation')
            pred = ''.encode('utf-8')
        res[key].append(pred)

        # multiple gold question per sentence
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    res_filtered = defaultdict(lambda: [])
    for k, v in res.items():

        # answer-level evaluation
        if prediction_aggregation is None:
            assert len(v) == 1
            res_filtered[k] = v

        elif prediction_aggregation == 'first':
            # the first one
            res_filtered[k] = [v[0]]
        elif prediction_aggregation == 'last':
            # the last one
            res_filtered[k] = [v[-1]]
        elif prediction_aggregation == 'long':
            # the longest generation
            res_filtered[k] = [v[v.index(sorted(v, key=len)[-1])]]
        elif prediction_aggregation == 'short':
            # the shortest generation
            res_filtered[k] = [v[v.index(sorted(v, key=len)[0])]]
        elif prediction_aggregation == 'middle':
            # middle length generation
            res_filtered[k] = [v[v.index(sorted(v, key=len)[int(len(v)/2)])]]
        else:
            raise ValueError('unknown aggregation method: {}'.format(prediction_aggregation))

    return QGEvalCap(gts, res_filtered).evaluate(bleu_only=bleu_only, language=language, skip=skip)


def evaluate(export_dir: str = '.',
             batch_size: int = 32,
             n_beams: int = 4,
             hypothesis_file_dev: str = None,
             hypothesis_file_test: str = None,
             model: str = None,
             max_length: int = 512,
             max_length_output: int = 64,
             dataset_path: str = 'asahi417/qg_squad',
             dataset_name: str = 'default',
             input_type: str = 'paragraph_answer',
             output_type: str = 'question',
             prediction_aggregation: str = 'first',
             prediction_level: str = 'sentence',
             data_caches: Dict = None,
             bleu_only: bool = False,
             overwrite: bool = False,
             use_auth_token: bool = False,
             language: str = 'en'):
    """ Evaluate question-generation model """
    path_metric = pj(export_dir, f'metric.{prediction_aggregation}.{prediction_level}.{input_type}.{output_type}.{dataset_path.replace("/", "_")}.{dataset_name}.json')
    metric = {}
    if not overwrite:
        if os.path.exists(path_metric):
            with open(path_metric, 'r') as f:
                metric = json.load(f)
                if bleu_only:
                    return metric
    os.makedirs(export_dir, exist_ok=True)
    reference_files = get_reference_files(dataset_path, dataset_name)

    if model is not None:
        lm = TransformersQG(model, max_length=max_length, max_length_output=max_length_output,
                            drop_overflow_text=False, skip_overflow_error=True, language=language)
        lm.eval()

        def get_model_prediction_file(split):
            path_hypothesis = pj(export_dir, f'samples.{split}.hyp.{input_type}.{output_type}.{dataset_path.replace("/", "_")}.{dataset_name}.txt')
            raw_input, _ = get_dataset(
                dataset_path, dataset_name, split=split, input_type=input_type, output_type=output_type,
                use_auth_token=use_auth_token)

            if os.path.exists(path_hypothesis) and not overwrite:
                with open(path_hypothesis) as f_read:
                    tmp = f_read.read().split('\n')
                if len(tmp) == len(raw_input):
                    return path_hypothesis
                logging.warning('recompute {}'.format(path_hypothesis))
            output = lm.generate_q(
                raw_input, batch_size=batch_size, num_beams=n_beams,
                cache_path=None if data_caches is None else data_caches[split])
            with open(path_hypothesis, 'w') as f_writer:
                f_writer.write('\n'.join(output))
            return path_hypothesis

        try:
            hypothesis_file_test = get_model_prediction_file('test')
        except ValueError:
            hypothesis_file_test = None
        try:
            hypothesis_file_dev = get_model_prediction_file('validation')
        except ValueError:
            hypothesis_file_dev = None
    assert hypothesis_file_dev is not None or hypothesis_file_test is not None,\
        f'model ({model}) or file path ({hypothesis_file_dev}, {hypothesis_file_test}) is needed'

    def get_metric(split, hypothesis_file, skip_metrics):
        assert prediction_level in ['sentence', 'paragraph', 'answer'], prediction_level
        if prediction_level == 'answer':
            src_file = None
        else:
            src_file = reference_files['{}-{}'.format(prediction_level, split)]

        __metric, _ = compute_metrics(
            out_file=hypothesis_file,
            tgt_file=reference_files['question-{}'.format(split)],
            src_file=src_file,
            prediction_aggregation=prediction_aggregation,
            language=language,
            bleu_only=bleu_only,
            skip=skip_metrics
        )
        return __metric

    for _hypothesis_file, _split in zip([hypothesis_file_dev, hypothesis_file_test], ['validation', 'test']):
        if _hypothesis_file is None:
            continue
        # print(_split, metric)
        if _split in metric:
            # print(list(metric[_split].keys()))
            _metric = get_metric(_split, _hypothesis_file, list(metric[_split].keys()))
            metric[_split].update(_metric)
        else:
            metric[_split] = get_metric(_split, _hypothesis_file, None)

    with open(path_metric, 'w') as f:
        json.dump(metric, f)
    return metric


