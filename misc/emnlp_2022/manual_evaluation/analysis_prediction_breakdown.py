import json
import os
from glob import glob
from tqdm import tqdm
from lmqg.automatic_evaluation import compute_metrics
from lmqg import get_reference_files


HF_MODEL_DIR = os.getenv('HF_MODEL_DIR', './huggingface_repo')
NON_LM_PREDICTION = os.getenv('NON_LM_PREDICTION', '../non_lm_baseline/nonlm_fixed.sample.test.hyp.txt')
EXPORT_DIR = os.getenv('EXPORT_DIR', 'data')
target_models = ['t5-large/paragraph_answer.question', 't5-small/paragraph_answer.question',
                 'bart-large/paragraph_answer.question', 't5-large-no-paragraph/sentence_answer.question',
                 't5-large-no-answer/paragraph_sentence.question', '../non_lm_baseline/nonlm_fixed.sample.test.hyp']
raw_target_file = 'raw_squad_test_data/question-test.txt'
raw_target_file_a = 'raw_squad_test_data/answer-test.txt'
raw_target_file_p = 'raw_squad_test_data/paragraph-test.txt'
raw_target_file_s = 'raw_squad_test_data/sentence-test.txt'


def load_file(path):
    with open(path) as f_reader:
        tmp = f_reader.read().split('\n')
        if tmp[-1] == '':
            tmp = tmp[:-1]
        return tmp
        # return [i for i in f_reader.read().split('\n') if len(i) != 0]


def filename_conversion(_filename):
    return _filename.split('lmqg-')[-1].replace('-squad', '').replace('eval/samples.test.hyp.', '').replace('.txt', '')


if __name__ == '__main__':
    reference_files = get_reference_files()
    question_norm = load_file(reference_files['question-test'])
    question_raw = load_file(raw_target_file)
    paragraph_raw = load_file(raw_target_file_p)
    answer_raw = load_file(raw_target_file_a)
    sentence_raw = load_file(raw_target_file_s)

    prediction_files = glob('{}/*/eval/samples.test.hyp.*.txt'.format(HF_MODEL_DIR))
    prediction_files.append(NON_LM_PREDICTION)
    _file = '{}/prediction.json'.format(EXPORT_DIR)
    output_dict = {k: {'prediction': v} for k, v in zip(
        ['gold_question_norm', 'gold_question_raw', 'gold_paragraph_raw', 'gold_answer_raw', 'gold_sentence_raw'],
        [question_norm, question_raw, paragraph_raw, answer_raw, sentence_raw])}
    if os.path.exists(_file):
        with open(_file) as f:
            tmp_dict = json.load(f)
            for i in ['gold_question_norm', 'gold_question_raw', 'gold_paragraph_raw', 'gold_answer_raw', 'gold_sentence_raw']:
                tmp_dict.pop(i)

            output_dict.update(tmp_dict)
    for prediction_file in tqdm(prediction_files):
        filename = filename_conversion(prediction_file)
        if filename in output_dict:
            continue
        if filename not in target_models:
            continue
        # answer level metric
        _, metric_individual = compute_metrics(
            out_file=prediction_file,
            tgt_file=reference_files['question-test'],
            bleu_only=False)
        output_dict[filename_conversion(prediction_file)] = {
            'prediction': load_file(prediction_file),
            'metric': {k: list(v) for k, v in metric_individual.items()}
        }
    with open(_file, 'w') as f_writer:
        json.dump(output_dict, f_writer)
