import json
from random import seed, shuffle

import pandas as pd


EXPORT_FILE = './data/main_format.csv'
PATH_TO_PREDICTION = 'data/prediction.json'
PATH_TO_REFERENCE = 'raw_squad_test_data'
TASK_PER_HIT = 5


def byte_check(string):
    return ''.join(char for char in string if len(char.encode('utf-8')) < 3)


def load_file(path):
    with open(path) as f:
        _data = f.read().split('\n')
        if _data[-1] == '':
            return _data[:-1]
        return _data


with open(PATH_TO_PREDICTION) as f:
    tmp = {k: v['prediction'] for k, v in json.load(f).items()}
    df = pd.DataFrame(tmp)
df['paragraph'] = load_file('{}/paragraph-test.txt'.format(PATH_TO_REFERENCE))
df['sentence'] = load_file('{}/sentence-test.txt'.format(PATH_TO_REFERENCE))
df['answer'] = load_file('{}/answer-test.txt'.format(PATH_TO_REFERENCE))
df = df[df['sentence'] != '']

target_models = [
    't5-large/qg',  # a
    't5-small/qg',  # b
    'bart-large/qg',  # c
    't5-large-no-passage/qg_no_passage',  # d
    't5-large-no-answer/qg_no_answer',  # e
    './misc/non_lm_baseline/nonlm_fixed.sample.test.hyp'  # f
]

# collect all possible data
list_flat = []
for paragraph, df_sub in df.groupby('paragraph'):
    fixed_info = None
    for i in range(len(df_sub)):
        sentence = df_sub['sentence'].tolist()[i]
        answer = df_sub['answer'].tolist()[i]
        tmp_split = [s for s in paragraph.split(sentence) if s != '']
        if answer in sentence and len(tmp_split) == 2:
            passage_before, passage_after = tmp_split
            fixed_info = [passage_before, sentence, passage_after, answer]
            break
    if fixed_info is None:
        continue
    list_flat.append(fixed_info + [df_sub[m].tolist()[-1] for m in target_models])

# format for mturk
seed(1234)
shuffle(list_flat)
mturk_format_data = []
while True:
    try:
        tmp = {}
        for i in range(TASK_PER_HIT):
            data = list_flat.pop(0)
            tmp['passage_{}_before'.format(i + 1)] = byte_check(data.pop(0))
            tmp['sentence_{}'.format(i + 1)] = byte_check(data.pop(0))
            tmp['passage_{}_after'.format(i + 1)] = byte_check(data.pop(0))
            tmp['answer_{}'.format(i + 1)] = byte_check(data.pop(0))
            tmp['question_{}a'.format(i + 1)] = byte_check(data.pop(0))
            tmp['question_{}b'.format(i + 1)] = byte_check(data.pop(0))
            tmp['question_{}c'.format(i + 1)] = byte_check(data.pop(0))
            tmp['question_{}d'.format(i + 1)] = byte_check(data.pop(0))
            tmp['question_{}e'.format(i + 1)] = byte_check(data.pop(0))
            tmp['question_{}f'.format(i + 1)] = byte_check(data.pop(0))
        mturk_format_data.append(tmp)
    except IndexError:
        break
df = pd.DataFrame(mturk_format_data)
df.iloc[:50, :].to_csv(EXPORT_FILE.replace('.csv', '.1.csv'), index=False)
df.iloc[50:100, :].to_csv(EXPORT_FILE.replace('.csv', '.2.csv'), index=False)
df.iloc[100:150, :].to_csv(EXPORT_FILE.replace('.csv', '.3.csv'), index=False)
df.iloc[150:200, :].to_csv(EXPORT_FILE.replace('.csv', '.4.csv'), index=False)
df.iloc[200:, :].to_csv(EXPORT_FILE.replace('.csv', '.5.csv'), index=False)

