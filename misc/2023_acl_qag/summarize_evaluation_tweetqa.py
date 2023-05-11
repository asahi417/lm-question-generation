""" Get Basic Evaluation Metric for QG/AE/QA """
import os
import json
import requests
from os.path import join as pj
import pandas as pd


LM = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large']
DATA = ['tweetqa']
TMP_DIR = 'metric_files'
EXPORT_DIR = "summary"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
def url_pattern(m, d): return f"https://huggingface.co/lmqg/{m}-{d}-qag/raw/main/eval/metric.first.answer.paragraph.questions_answers.lmqg_qag_{d}.default.json"
def url_pattern_config(m, d): return f"https://huggingface.co/lmqg/{m}-{d}-qag/raw/main/trainer_config.json"


def download(filename, url):
    try:
        with open(filename) as f:
            json.load(f)
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(url)
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    with open(filename) as f:
        tmp = json.load(f)
    return tmp


def get_qag_metric(account: str = 'lmqg', model: str = 't5-small', data: str = 'tweetqa'):
    tmp = download(pj(TMP_DIR, f'{account}.{model}.{data}.json'), url_pattern(model, data))
    metric = {
        "Model": f"[`{account}/{model}-{data}-qag`](https://huggingface.co/{account}/{model}-{data}-qag)",
        "Data": f"[`{account}/qag_{data}`](https://huggingface.co/datasets/{account}/qag_{data})"
    }
    metric.update(
        {k: 100 * tmp['test'][k] for k in sorted(tmp['test'].keys())})
    config = download(pj(TMP_DIR, f'config.{account}.{model}.{data}.json'), url_pattern_config(model, data))
    return metric, config


if __name__ == '__main__':
    metrics = []
    configs = []
    for _lm in LM:
        for _d in DATA:
            _metric, _config = get_qag_metric(model=os.path.basename(_lm), data=_d)
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

