""" With trained QG model, generate synthetic QA dataset from the wikipedia paragraph collected by
https://github.com/xinyadu/harvestingQA/tree/master/dataset.
First, download `train.json` file from https://drive.google.com/drive/folders/1E6Cg7c0XkWBOszMgHq_gHNRZVnAW2mEa, and
locate it under `./data/`, then run this script.
"""
import os
import argparse
import json
from os.path import join as jp
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import pipeline
from datasets import load_dataset

path_original_data = jp('data', 'train.json')
path_original_data_valid = jp('data', 'validation.jsonl')
path_processed_data = jp('data', 'train.synthetic.jsonl')
assert os.path.exists(path_original_data),\
    'download data from https://github.com/xinyadu/harvestingQA/tree/master/dataset'
highlight_token = '<hl>'
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
MAX_CHAR_LENGTH = 1000


class ListDataset(Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def pre_process_data():
    if os.path.exists(path_processed_data):
        return

    with open(path_processed_data, 'w') as f_writer:
        with open(path_original_data) as f_reader:
            data = json.load(f_reader)['data']
        for d in tqdm(data):
            title = d['title']
            for p in d['paragraphs']:
                context = p['context']
                for a in p['qas']:
                    for ans in a['answers']:
                        ans = {"answer_start": [ans['answer_start']], "text": [ans["text"]]}
                        f_writer.write(
                            json.dumps({'context': context, 'id': a['id'], 'title': title, 'answers': ans}) + '\n')
    return


def get_synthetic_data(model_path, export_file, batch_size: int = 32):
    os.makedirs(os.path.dirname(export_file), exist_ok=True)
    pipe = pipeline("text2text-generation", model_path, device=0 if CUDA_VISIBLE_DEVICES is not None else -1)
    prefix = '' if pipe.model.config.add_prefix else 'generate question: '

    def format_input(context, answer):
        input_text = context.replace(answer, '{0} {1} {0}'.format(highlight_token, answer))
        return '{}{}'.format(prefix, input_text)

    with open(path_processed_data) as f:
        data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
        data = [i for i in data if len(i['context']) < MAX_CHAR_LENGTH]
        model_input = [format_input(i['context'], i['answers']['text']) for i in data]

    if os.path.exists(export_file + '.tmp'):
        with open(export_file + '.tmp', 'r') as f:
            output = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    else:
        output = []
    with open(export_file + '.tmp', 'w') as f:
        if len(output) > 0:
            f.write('\n'.join([json.dumps(i) for i in output]) + '\n')
        dataset = ListDataset(model_input[len(output):])
        pbar = tqdm(total=len(model_input))
        for out in pipe(dataset, batch_size=batch_size):
            pbar.update(1)
            f.write('\n'.join([json.dumps(i) for i in out]) + '\n')

    with open(export_file + '.tmp', 'r') as f:
        output = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]

    with open(export_file, 'w') as f:
        assert len(output) == len(data), str([len(output), len(data)])
        for g, _json in zip(output, data):
            _json['question'] = g['generated_text']
            f.write(json.dumps(_json) + '\n')


def output_squad_dev():
    if os.path.exists(path_original_data_valid):
        return
    validation = load_dataset("squad", split="validation")
    with open(path_original_data_valid, 'w') as f:
        for i in validation:
            f.write(json.dumps(i) + '\n')


def get_options():
    parser = argparse.ArgumentParser(description='QAE data creation.')
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('-e', '--export', required=True, type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    pre_process_data()
    get_synthetic_data(opt.model, opt.export, opt.batch_size)
    output_squad_dev()

