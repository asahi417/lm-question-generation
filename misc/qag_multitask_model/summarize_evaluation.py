""" Get Basic Evaluation Metric for QG/AE/QA """
import os
import json
import requests
from itertools import product
from os.path import join as pj
import pandas as pd

DATA = ['squad']
LM_QG = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large']
LM_AE = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large']
LM_QG_AE = ['t5-small', 't5-base', 't5-large']

DATA_ML = ['ruquad', 'jaquad', 'itquad', 'koquad', 'esquad', 'dequad', 'frquad']
LM_QG_ML = ['google/mt5-small', 'google/mt5-base', 'facebook/mbart-large-cc25']
LM_AE_ML = []
LM_QG_AE_ML = ['mt5-small', 'mt5-base']

METRIC_PERC = ["AnswerF1Score", "AnswerExactMatch"]
TMP_DIR = 'metric_files'
EXPORT_DIR = "summary"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
def url_ae(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.answer.paragraph_sentence.answer.lmqg_qg_{d}.default.json"
def url_qg(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{d}.default.json"
def url_qag(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.answer.paragraph.questions_answers.lmqg_qg_{d}.default.json"


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
        return json.load(f)


if __name__ == '__main__':

    metrics_ae = []
    metrics_qg = []
    metrics_qag = []
    for _d, _lm in product(DATA, LM_AE):
        _m = os.path.basename(_lm)
        _metric = {
            "Model": f"[`lmqg/{os.path.basename(_m)}-{_d}-ae`](https://huggingface.co/lmqg/{_m}-{_d}-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "AE",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})"
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.ae.ae.json'), url_ae(_m, _d, 'ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_ae.append(_metric)

    for _d, _lm in list(product(DATA, LM_QG)) + list(product(DATA_ML, LM_QG_ML)):
        _m = os.path.basename(_lm)
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg`](https://huggingface.co/lmqg/{_m}-{_d}-qg)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "QG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})"
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg.qg.json'), url_qg(_m, _d, 'qg'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qg.append(_metric)

        if _m in LM_QG_AE:
            _metric = {
                "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
                "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
                "Type": "Multitask",
                "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})"
            }
            tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.qag.json'), url_qag(_m, _d, 'qg-ae'))
            _metric.update(
                {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
                 sorted(tmp['test'].keys())})
            metrics_qag.append(_metric)

    for _d, _lm in list(product(DATA, LM_QG_AE)) + list(product(DATA_ML, LM_QG_AE_ML)):
        _m = os.path.basename(_lm)

        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "Multitask",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})"
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.ae.json'), url_ae(_m, _d, 'qg-ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_ae.append(_metric)

        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "Multitask",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})"
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.qg.json'), url_qg(_m, _d, 'qg-ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qg.append(_metric)

        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "Multitask",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})"
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.qag.json'), url_qag(_m, _d, 'qg-ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qag.append(_metric)

    df = pd.DataFrame(metrics_ae).round(2)
    df['BLEU-1'] = df.pop('Bleu_1')
    df['BLEU-2'] = df.pop('Bleu_2')
    df['BLEU-3'] = df.pop('Bleu_3')
    df['BLEU-4'] = df.pop('Bleu_4')
    df['ROUGE-L'] = df.pop('ROUGE_L')
    print('- Answer Extraction\n')
    print(df.to_markdown(index=False), '\n\n')
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, "summary.ae.csv"), index=False)

    df = pd.DataFrame(metrics_qg).round(2)
    df['BLEU-1'] = df.pop('Bleu_1')
    df['BLEU-2'] = df.pop('Bleu_2')
    df['BLEU-3'] = df.pop('Bleu_3')
    df['BLEU-4'] = df.pop('Bleu_4')
    df['ROUGE-L'] = df.pop('ROUGE_L')
    print('- Question Generation\n')
    print(df.to_markdown(index=False), '\n\n')
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, "summary.qg.csv"), index=False)

    df = pd.DataFrame(metrics_qag).round(2)
    print('- Question & Answer Pairs Generation\n')
    print(df.to_markdown(index=False), '\n\n')
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, "summary.qag.csv"), index=False)


