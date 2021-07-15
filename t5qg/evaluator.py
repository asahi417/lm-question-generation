import json
import os
import random
from typing import List

import torch
from nlgeval import compute_metrics
from .lm_t5 import T5
from .data import get_dataset


def evaluate_qg(model: str,
                export_dir: str,
                dataset: str = "squad",
                language: List or str = 'en',
                max_length: int = 512,
                max_length_output: int = 32,
                batch: int = 128,
                num_beams: int = 4,
                random_seed:  int = 32):
    path_metric = '{}/metric.json'.format(export_dir)
    if os.path.exists(path_metric):
        with open(path_metric, 'r') as f:
            return json.load(f)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    lm = T5(model, max_length=max_length, max_length_output=max_length_output)
    lm.eval()
    os.makedirs(export_dir, exist_ok=True)
    metrics_dict = {}

    for split in ['dev', 'test']:
        path_hypothesis = '{}/samples.{}.hyp.txt'.format(export_dir, split)
        path_reference = '{}/samples.{}.ref.txt'.format(export_dir, split)
        if not os.path.exists(path_hypothesis) or not os.path.exists(path_reference):
            raw_input, raw_output = get_dataset(dataset, split=split, language=language, task_type='qg')
            output = lm.get_prediction(raw_input,
                                       batch_size=batch,
                                       num_beams=num_beams,
                                       drop_overflow_text=True)
            with open(path_hypothesis, 'w') as f:
                f.write('\n'.join(output))
            with open(path_reference, 'w') as f:
                f.write('\n'.join(raw_output))

        metrics_dict[split] = compute_metrics(hypothesis=path_hypothesis, references=[path_reference],
                                              no_skipthoughts=True, no_glove=True)

    with open(path_metric, 'w') as f:
        json.dump(metrics_dict, f)
    return metrics_dict


