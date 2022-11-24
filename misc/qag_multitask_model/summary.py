import json
import os.path
from os.path import join as pj
import requests
from glob import glob
import pandas as pd


#### SUMMARY FOR QAE ####
if not os.path.exists("./summary.qae.csv"):
    output = []
    for i in glob("qa_eval_output/silver_qa.*/*/test_result.json") + glob("qa_eval_output/gold_qa/*/test_result.json"):
        with open(i) as _f:
            tmp = json.load(_f)
        path = i.split('/')
        config = path[2].split('.')
        if len(config) == 3:
            lm, _data, data_name = path[2].split('.')
            tmp['qg_model'] = path[1].replace('silver_qa.', '')
        else:
            lm, _data, data_name, filtering = path[2].split('.')
            tmp['qg_model'] = f"{path[1].replace('silver_qa.', '')}.{filtering}"
        tmp['data'] = _data
        tmp['data_name'] = data_name
        tmp['qa_model'] = lm
        output.append(tmp)
    df = pd.DataFrame(output)
    df.to_csv("./summary.qae.csv", index=False)


#### SUMMARY FOR QA F1 ####
TMP_DIR = 'metric'
os.makedirs(TMP_DIR, exist_ok=True)


def download(filename: str, url: str):
    try:
        with open(filename) as f:
            json.load(f)
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    with open(filename) as f:
        return json.load(f)


def get_metric(account: str = 'lmqg',
               model: str = 't5-small',
               data: str = 'squad',
               data_type: str = 'default',
               suffix: str = None):
    model = f'{model}-{data}'
    data_type = 'default' if data_type is None else data_type
    if data_type != 'default':
        model = f'{model}-{data}'
    if suffix is not None:
        model = f'{model}-{suffix}'
    url = f"https://huggingface.co/{account}/{model}/raw/main/eval/" \
          f"metric.first.answer.paragraph.questions_answers.lmqg_qg_{data}.{data_type}.json"
    print(url)
    filename = pj(TMP_DIR, f'qag_metric.{account}.{model}.eval.qg_{data}.{data_type}.json')
    _metric = download(filename, url)
    _metric = {k: 100 * v for k, v in _metric['test'].items()}
    _metric['model'] = model
    _metric['data'] = data
    return _metric


full_result = []
for m in ["t5-small", "t5-base", "t5-large"]:
    full_result.append(get_metric(model=m))
    full_result.append(get_metric(model=m, suffix='multitask'))
for m in ["bart-base", "bart-large"]:
    full_result.append(get_metric(model=m))
for m in ["ja", "es", "ko", "fr", "ru", "it", "de"]:
    full_result.append(get_metric(model='mt5-small', data=f'{m}quad'))
    full_result.append(get_metric(model='mt5-small', data=f'{m}quad', suffix='multitask'))
    full_result.append(get_metric(model='mbart-large-cc25', data=f'{m}quad'))
df = pd.DataFrame(full_result)
df.to_csv("./summary.qa_aligned_f1.csv", index=False)

