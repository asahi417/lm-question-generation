import os
from glob import glob
from itertools import combinations

from Levenshtein import distance
import pandas as pd


HF_MODEL_DIR = os.getenv('HF_MODEL_DIR', './huggingface_repo')
EXPORT_DIR = os.getenv('EXPORT_DIR', 'data/')
NON_LM_PREDICTION = os.getenv('NON_LM_PREDICTION', '../non_lm_baseline/nonlm_fixed.sample.test.hyp.txt')
PREDICTION_FILES = glob('{}/*/eval/samples.test.hyp.*.txt'.format(HF_MODEL_DIR))
raw_target_file = 'raw_squad_test_data/question-test.txt'


def load_file(path):
    with open(path) as f:
        return [i for i in f.read().split('\n') if len(i) != 0]


def get_model_input(model_alias):
    # check the model type
    model_input = 'passage+answer'
    task_type = 'qg'
    if model_alias.endswith('default'):
        model_input = 'passage+answer (default)'
    elif model_alias.endswith('no-answer'):
        model_input = 'passage+sentence'
        task_type = 'qg_no_answer'
    elif model_alias.endswith('no-passage'):
        model_input = 'sentence+answer'
        task_type = 'qg_no_passage'
    return model_input, task_type


def get_language_model_name(model_alias):
    # check base language model
    if model_alias.startswith('lmqg-t5-small'):
        return 'T5 (SMALL)'
    elif model_alias.startswith('lmqg-t5-base'):
        return 'T5 (BASE)'
    elif model_alias.startswith('lmqg-t5-large'):
        return 'T5 (LARGE)'
    elif model_alias.startswith('lmqg-bart-base'):
        return 'BART (BASE)'
    elif model_alias.startswith('lmqg-bart-large'):
        return 'BART (LARGE)'
    else:
        raise ValueError('unknown model: {}'.format(model_alias))


def average_edit_distance(file_a, file_b):
    data_a = load_file(file_a)
    data_b = load_file(file_b)
    assert len(data_a) == len(data_b), '{} != {}'.format(len(data_a), len(data_b))
    d = []
    exact_match = []
    for a, b in zip(data_a, data_b):
        d.append(distance(a, b))
        exact_match.append(a == b)
    return sum(d)/len(d), sum(exact_match)/len(exact_match) * 100


def filename_conversion(filename):
    return filename.split('lmqg-')[-1].replace('-squad', '').replace('eval/samples.test.hyp.', '').replace('.txt', '')


def get_similarity():
    # get reference/NQG baseline
    PREDICTION_FILES.append(raw_target_file)
    PREDICTION_FILES.append(NON_LM_PREDICTION)

    # get edit distance as a proxy of sentence similarity across all possible combination
    output = []
    for a, b in combinations(PREDICTION_FILES, 2):
        d, exact_match = average_edit_distance(file_a=a, file_b=b)
        output.append([filename_conversion(a), filename_conversion(b), d, exact_match])

    df = pd.DataFrame(output, columns=['Prediction A', 'Prediction B', 'Average Edit-distance', 'Exact Match Ratio'])
    return df


if __name__ == '__main__':
    df_sim = get_similarity()
    df_sim.to_csv('{}/prediction.similarity.csv'.format(EXPORT_DIR), index=False)
