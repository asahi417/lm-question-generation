""" Training model. """
# import itertools
# import os
import logging
# from typing import List
from itertools import chain
import torch
from sklearn.metrics import precision_recall_fscore_support

from .lm_bert import BERTClassifier
from .lm_t5 import T5Classifier
from .data import get_dataset
PROGRESS_INTERVAL = 50


def compute_metric_f1(label, prediction, prefix: str = ''):
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(label, prediction, average='macro')
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(label, prediction, average='micro')
    return {"{}f1_micro".format(prefix): f_mic * 100, "{}f1_macro".format(prefix): f_mac * 100,
            "{}recall_micro".format(prefix): r_mic * 100, "{}recall_macro".format(prefix): r_mac * 100,
            "{}precision_micro".format(prefix): p_mic * 100, "{}precision_macro".format(prefix): p_mac * 100}


def evaluate(model_name: str,
             model_type: str = None,
             max_length: int = 128,
             batch: int = 512,
             num_workers: int = 1,
             cache_dir: str = None):
    if model_type is None:
        model_type = 't5' if 't5' in model_name else 'bert'
    if model_type == 'bert':
        model = BERTClassifier(model=model_name, max_length=max_length, add_phrase_classifier=True)
    elif model_type == 't5':
        model = T5Classifier(model=model_name, max_length=max_length)
    else:
        raise ValueError('unknown model: {}'.format(model_type))
    model.eval()
    logging.info('dataset preprocessing')
    # evaluation on the original dataset (non-translated)
    target = ['en/en', 'ja/ja', 'de/de']
    full_data, full_data_meta = get_dataset(target, target, cache_dir)
    logging.info('start model evaluation')
    with torch.no_grad():
        metric = {'model_name': model_name, 'model_ckpt': model_name.split('/epoch')[0]}
        for data_type in ['valid', 'test']:

            logging.info('running model inference: {}'.format(data_type))
            data, data_meta = full_data[data_type], full_data_meta[data_type]
            cfd_prediction, cfd_label, cp_prediction, cp_label \
                = model.get_prediction(data, batch_size=batch, num_workers=num_workers)
            logging.info('aggregating metric over languages')
            language = [i[3] for i in data_meta]  # language type
            for lan in set(language):
                _cfd_label = [y for x, y in zip(language, cfd_label) if x == lan]
                _cfd_prediction = [y for x, y in zip(language, cfd_prediction) if x == lan]
                # print(_cfd_label, _cfd_prediction)
                # print(lan)
                metric.update(compute_metric_f1(_cfd_label, _cfd_prediction,
                                                prefix='cfd_label/{}/{}/'.format(lan[:2], data_type)))
                if len(cp_label) > 0 and len(cp_prediction) > 0:
                    _cp_label = list(chain(*[y for x, y in zip(language, cp_label) if x == lan]))
                    _cp_prediction = list(chain(*[y for x, y in zip(language, cp_prediction) if x == lan]))
                    metric.update(compute_metric_f1(_cp_label, _cp_prediction,
                                                    prefix='clue_phrase/{}/{}/'.format(lan[:2], data_type)))
                # metric['target'] = lan
                # metric['type'] = data_type
                # metric['model_name'] = model_name
    logging.info(str(metric))
    # output.append(metric)
    return metric
