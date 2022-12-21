import os
import requests
from itertools import product
from statistics import mean
from os.path import join as pj
import pandas as pd
from datasets import load_dataset

LMS = ["t5-small-squad",
       "t5-base-squad",
       "t5-large-squad",
       "bart-base-squad",
       "bart-large-squad"]
DOMAINS = ["new_wiki", "reddit", "amazon", "nyt"]
QAG_TYPES = ['qg_reference', 'pipeline', 'multitag', 'end2end']

table = []
for d, q, l in product(DOMAINS, QAG_TYPES, LMS):
    tmp = {"domain": d, "qag_type": q, "lm": l}
    _data = load_dataset("lmqg/qa_squadshifts_synthetic", f"{l}.{q}.{d}")
    for _split in _data:
        _data[_split].filter(lambda x: len(x['answers']['text']) > 0 and len(x['answers']['text'][0]) < 512)
        df = .to_pandas()
        length = [len(g) for _, g in df.groupby('context')]
        tmp[f"{_split}/mean"] = mean(length)
        tmp[f"{_split}/sum"] = sum(length)
    table.append(tmp)
pd.DataFrame(table).to_csv('summary.dataset.squadshifts.csv', index=False)
TMP_DIR = 'metric'
os.makedirs(TMP_DIR, exist_ok=True)


def download(filename: str, url: str):
    try:
        with open(filename) as f:
            return f.read()
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    with open(filename) as f:
        return f.read()


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
          f"samples.test.hyp.paragraph.questions_answers.lmqg_qg_{data}.{data_type}.txt"
    print(url)
    filename = pj(TMP_DIR, f'qag_prediction.{account}.{model}.eval.qg_{data}.{data_type}.json')
    pred = download(filename, url)
    nums = [len(p.split(' | ')) for p in pred.split('\n') if len(p) > 0]
    return sum(nums), mean(nums)


table = []
_sum, _mean = get_metric()
table.append({"data": "squad", "model": "original", "sum": _sum, "mean": _mean})
for i in ["t5-small", "t5-base", "t5-large"]:
    _sum, _mean = get_metric(model=i, suffix='multitask')
    table.append({"data": "squad", "model": f"{i}-squad-multitask", "sum": _sum, "mean": _mean})
for m in ["ja", "es", "ko", "fr", "ru", "it", "de"]:
    _sum, _mean = get_metric(model='mt5-small', data=f'{m}quad')
    table.append({"data": f"{m}quad", "model": "original", "sum": _sum, "mean": _mean})
    _sum, _mean = get_metric(model='mt5-small', data=f'{m}quad', suffix='multitask')
    table.append({"data": f"{m}quad", "model": f"mt5-small-{m}quad-multitask", "sum": _sum, "mean": _mean})

pd.DataFrame(table).to_csv('summary.dataset.squad.csv', index=False)
