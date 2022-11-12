import json
import os
import re
from collections import Counter
from glob import glob
from itertools import chain
import pandas as pd

prediction_file = 'data/prediction.json'
models = {'a': 'T5 Large',
          'b': 'T5 Small',
          'c': 'BART Large',
          'd': 'T5 Large (No Passage)',
          'e': 'T5 Large (No Answer)', 'f': 'LSTM'}
models_alias = {
    'a': 't5-large/paragraph_answer.question',
    'b': 't5-small/paragraph_answer.question',
    'c': 'bart-large/paragraph_answer.question',
    'd': 't5-large-no-paragraph/sentence_answer.question',
    'e': 't5-large-no-answer/paragraph_sentence.question',
    'f': '../non_lm_baseline/nonlm_fixed.sample.test.hyp'
}
metric_type = ['correctness', 'grammaticality', 'understandability']
df = pd.concat([pd.read_csv(_file) for _file in glob('./mturk_output/main/*csv')])
required_annotation = 5
for _file in glob('./data/main_format.extra*csv'):
    os.remove(_file)


def get_majority(_list):
    _freq = get_frequency(_list)
    max_val = max([_i[1] for _i in _freq])
    max_key = [_i[0] for _i in _freq if _i[1] == max_val]
    return max_key


def get_frequency(_list):
    c = Counter(_list)
    return sorted(c.most_common(), key=lambda _x: _x[0])


# aggregate annotation
worker_id_dict = {i: n for n, i in enumerate(sorted(df["WorkerId"].unique().tolist()))}
output = {}
for n in models.keys():  # 'a' to 'f'
    for _, tmp_df in df.iterrows():
        for i in range(1, 6):
            key = tmp_df['Input.passage_{}_before'.format(i)]
            # question_to_annotate = tmp_df['Input.question_{}{}'.format(i, n)]
            if key not in output:
                output[key] = {}
            if models[n] not in output[key]:
                output[key][models[n]] = {m: [] for m in metric_type}
                output[key][models[n]]['worker'] = []
            for m in metric_type:
                output[key][models[n]][m].append(tmp_df['Answer.{}_{}{}'.format(m, i, n)])
            output[key][models[n]]['question'] = tmp_df['Input.question_{}{}'.format(i, n)]

            worker_id = worker_id_dict[tmp_df['WorkerId']]
            output[key][models[n]]['worker'].append(worker_id)

# check the number of annotation
n_annotation = {}
for k, v in output.items():
    scores = []
    for _k, _v in v.items():
        n_score = list(set([len(_v[__v]) for __v in metric_type]))
        assert len(n_score) == 1, n_score
        scores.append(n_score[0])
    n_score = list(set(scores))
    assert len(n_score) == 1, n_score
    n_annotation[k] = n_score[0]
# check how many annotation we need more and generate `extra` files for further annotation
df_format_data = pd.concat([pd.read_csv(_file) for _file in glob('./data/main_format*csv')])
for i in range(required_annotation + 1):
    annotation_required = [k for k, v in n_annotation.items() if required_annotation - v == i]
    print('- {} data need {} more annotation'.format(len(annotation_required), i))

    def match_check(passage):
        passage = re.sub(r'\s*\Z', '', passage)
        for a in annotation_required:
            a = re.sub(r'\s*\Z', '', a)
            if a == passage:
                return True
        return False

    if len(annotation_required) > 0:
        tmp_df = df_format_data[[match_check(p) for p in df_format_data['passage_1_before']]]
        passage_from_data = list(chain(*[tmp_df['passage_{}_before'.format(n)].values.tolist() for n in range(1, 6)]))
        assert len(passage_from_data) == len(annotation_required)
        passage_from_data = sorted(passage_from_data)
        annotation_required = sorted(annotation_required)
        for x, y in zip(passage_from_data, annotation_required):
            x = re.sub(r'\s*\Z', '', x)
            y = re.sub(r'\s*\Z', '', y)
            assert x == y, str([x, y])
        tmp_df.to_csv('data/main_format.extra.{}.csv'.format(i), index=False)

# qualified worker
worker_id_freq = get_frequency(df['WorkerId'].tolist())
worker_id = [i[0] for i in worker_id_freq]
files = [i for i in glob('./mturk_output/users/*.csv') if not i.endswith('update.csv')]
assert len(files) == 1, files
df_users = pd.read_csv(files[0])
qualified_worker_id = df_users[df_users['CURRENT-QG Evaluation Qualified'] == 100]['Worker ID'].tolist()
# worker qualified, but never worked on the task
never_worked_qualified_worker = [i for i in qualified_worker_id if i not in worker_id]

# statistics
f = open('mturk_output/main/report.txt', 'w')
f.write('## INFO ##\n')
f.write('\t - num of qualified worker: {}\n'.format(len(qualified_worker_id)))
f.write('\t - num of unique worker   : {}\n'.format(len(worker_id)))
f.write('\t - num of new worker      : {}\n'.format(len(never_worked_qualified_worker)))
f.write('\t - frequency              : {}\n\n'.format(worker_id_freq))
tmp = [0] * len(df_users)
df_users['UPDATE-QG Evaluation Qualified (new worker)'] = 0
df_users['UPDATE-QG Evaluation Qualified (new worker)'][[i in never_worked_qualified_worker for i in df_users['Worker ID'].tolist()]] = 100
assert (df_users['UPDATE-QG Evaluation Qualified (new worker)'] == 100).sum() == len(never_worked_qualified_worker),\
    str([(df_users['UPDATE-QG Evaluation Qualified (new worker)'] == 100).sum(), len(never_worked_qualified_worker)])
# print('* new workers: {}'.format(never_worked_qualified_worker))
df_users.to_csv(files[0].replace('.csv', '.update.csv'), index=False)

f.write('## AGREEMENTS ##\n')
for n, model_name in models.items():
    _tmp_output = {k: v[model_name] for k, v in output.items()}
    f.write('Model: {}\n'.format(model_name))
    for m in metric_type:
        _tmp_output_metric = {k: v[m] for k, v in _tmp_output.items()}
        n_agree = len([i for i in _tmp_output_metric.values() if len(set(i)) == 1])
        n_split = len([i for i in _tmp_output_metric.values() if len(get_majority(i)) != 1])
        n_disagree = len(_tmp_output_metric) - n_agree - n_split
        f.write(' * {}\n'.format(m))
        f.write('\t - Agree   : {}/{}\n'.format(n_agree, len(_tmp_output_metric)))
        f.write('\t - Split   : {}/{}\n'.format(n_split, len(_tmp_output_metric)))
        f.write('\t - Disagree: {}/{}\n'.format(n_disagree, len(_tmp_output_metric)))
    f.write('\n')

# score report
f.write('## AVERAGE SCORE ##\n')
for n, model_name in models.items():
    _tmp_output = {k: v[model_name] for k, v in output.items()}
    f.write('Model: {}\n'.format(model_name))
    for m in metric_type:
        _tmp_output_metric = {k: v[m] for k, v in _tmp_output.items()}
        _scores = [get_majority(i) for i in _tmp_output_metric.values()]
        # remove split vote
        _scores = [i[0] for i in _scores if len(i) == 1]
        freq = get_frequency(_scores)
        f.write(' * {}\n'.format(m))
        for value, count in freq:
            f.write('\t - score {}: {}\n'.format(value, count))
    f.write('\n')
f.close()


# export data
def normalize(_string):
    _string = re.sub(r'\A\s+', '', _string)
    _string = re.sub(r'\s+\Z', '', _string)
    _string = _string.replace('ṛṣṇ', '')
    return _string


with open(prediction_file) as f:
    prediction = json.load(f)

# input(prediction.keys())
metric = ['Bleu_4', 'METEOR', 'ROUGE_L', 'BERTScore', 'MoverScore']
print(prediction.keys())
reference_norm = prediction['gold_question_norm']['prediction']
reference_raw = prediction['gold_question_raw']['prediction']
paragraph_raw = prediction['gold_paragraph_raw']['prediction']
sentence_raw = prediction['gold_sentence_raw']['prediction']
answer_raw = prediction['gold_answer_raw']['prediction']

final_output = {}
for k in models_alias.keys():
    tmp = [i[models[k]] for i in output.values()]
    new_tmp = []
    for i in tmp:
        _tmp = {m: i[m] for m in metric_type + ['worker']}
        _tmp['prediction'] = normalize(i['question'])
        new_tmp.append(_tmp)
    # tmp = {normalize(i['question']): {m: i[m] for m in metric_type + ['worker']} for i in tmp}
    _pred = prediction[models_alias[k]]
    _metric = _pred.pop('metric')
    for m in metric:
        _pred['{}'.format(m)] = _metric[m]

    # print(len(sentence_raw), len(reference_raw), len(answer_raw))
    def add_reference(_dict, _n):

        _dict['reference_raw'] = reference_raw[_n]
        _dict['answer_raw'] = answer_raw[_n]
        _dict['paragraph_raw'] = paragraph_raw[_n]
        _dict['sentence_raw'] = sentence_raw[_n]
        _dict['reference_norm'] = reference_norm[_n]
        return _dict


    _pred = {
        normalize(_pred['prediction'][n]): add_reference({k: _pred[k][n] for k in _pred.keys() if k != 'prediction'}, n)
        for n in range(len(_pred['prediction']))}

    for i in new_tmp:
        i.update(_pred[i['prediction']])

    final_output[models[k]] = new_tmp

with open('data/final_result.json', 'w') as f:
    json.dump(final_output, f)

output = []
for i in final_output.keys():
    for k in final_output[i]:
        k["correctness"] = sum(k["correctness"]) / 5
        k["grammaticality"] = sum(k["grammaticality"]) / 5
        k["understandability"] = sum(k["understandability"]) / 5
        k["model"] = i
        k.pop("worker")
        output.append(k)
pd.DataFrame(output).to_csv('data/final_result.csv', index=False)
