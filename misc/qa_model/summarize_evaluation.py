""" Get Basic Evaluation Metric for QG/AE/QA """
import os
import json
import requests
from os.path import join as pj
import pandas as pd


LM = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large']
DATA = ['tweetqa']
METRIC_PERC = ["AnswerF1Score", "AnswerExactMatch"]
TMP_DIR = 'metric_files'
EXPORT_DIR = "summary"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
def url_pattern(x): return f"metric.first.answer.paragraph_question.answer.lmqg_qg_{x}.default.json"


def download(filename, url):
    try:
        with open(filename) as f:
            json.load(f)
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    with open(filename) as f:
        tmp = json.load(f)
    return tmp


def get_qa_metric(account: str = 'lmqg', model: str = 't5-small', data: str = 'tweetqa'):
    model = f'{model}-{data}-qa'
    url = f"https://huggingface.co/{account}/{model}/raw/main/eval/{url_pattern(data)}"
    print(url)
    tmp = download(pj(TMP_DIR, f'{account}.{model}.{data}.json'), url)
    url = f"https://huggingface.co/{account}/{model}/raw/main/trainer_config.json"
    config = download(pj(TMP_DIR, f'config.{account}.{model}.{data}.json'), url)
    metric = {
        "Model": f"[`{account}/{model}`](https://huggingface.co/{account}/{model})",
        "Data": f"[`{account}/qg_{data}`](https://huggingface.co/datasets/{account}/qag_{data})"
    }
    metric.update(
        {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in sorted(tmp['test'].keys())})

    return metric, config


if __name__ == '__main__':
    metrics = []
    configs = []
    for _lm in LM:
        for _d in DATA:
            _metric, _config = get_qa_metric(model=os.path.basename(_lm), data=_d)
            _metric['BLEU-1'] = _metric.pop('Bleu_1')
            _metric['BLEU-2'] = _metric.pop('Bleu_2')
            _metric['BLEU-3'] = _metric.pop('Bleu_3')
            _metric['BLEU-4'] = _metric.pop('Bleu_4')
            _metric['ROUGE-L'] = _metric.pop('ROUGE_L')
            _metric['Language Model'] = f"[`{_lm}`](https://huggingface.co/{_lm})"
            metrics.append(_metric)
            configs.append(_config)
    df = pd.DataFrame(metrics).round(2)
    print(df.to_markdown(index=False))
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, 'metric.csv'), index=False)
    pd.DataFrame(configs).to_csv(pj(EXPORT_DIR, 'config.csv'), index=False)

