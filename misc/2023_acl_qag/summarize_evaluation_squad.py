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
LM_QAG = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large']

DATA_ML = ['ruquad', 'jaquad', 'itquad', 'koquad', 'esquad', 'dequad', 'frquad']
LM_QG_ML = ['google/mt5-small', 'google/mt5-base', 'facebook/mbart-large-cc25']
LM_AE_ML = ['google/mt5-small', 'google/mt5-base', 'facebook/mbart-large-cc25']
LM_QG_AE_ML = ['google/mt5-small', 'google/mt5-base', 'facebook/mbart-large-cc25']
LM_QAG_ML = ['google/mt5-small', 'google/mt5-base', 'facebook/mbart-large-cc25']
LANG_MAP = {'ruquad': "Russian", 'jaquad': "Japanese", 'itquad': "Italian", 'koquad': "Korean", 'esquad': "Spanish", 'dequad': "German", 'frquad': "French", "squad": "English"}
METRIC_PERC = ["AnswerF1Score", "AnswerExactMatch"]
TMP_DIR = 'metric_files'
EXPORT_DIR = "summary"
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
def url_ae(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.answer.paragraph_sentence.answer.lmqg_qg_{d}.default.json"
def url_qg(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{d}.default.json"
def url_qag(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.answer.paragraph.questions_answers.lmqg_qag_{d}.default.json"
def url_multitask(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.answer.paragraph.questions_answers.lmqg_qg_{d}.default.json"
def url_pipeline(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval_pipeline/metric.first.answer.paragraph.questions_answers.lmqg_qg_{d}.default.lmqg_{m}-{d}-ae.json"
def url_reference(m, d, suffix): return f"https://huggingface.co/lmqg/{m}-{d}-{suffix}/raw/main/eval/metric.first.answer.paragraph.questions_answers.lmqg_qg_{d}.default.json"


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
    for _d, _lm in list(product(DATA, LM_AE)) + list(product(DATA_ML, LM_AE_ML)):
        _m = os.path.basename(_lm)

        # AE metrics
        _metric = {
            "Model": f"[`lmqg/{os.path.basename(_m)}-{_d}-ae`](https://huggingface.co/lmqg/{_m}-{_d}-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "AE",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.ae.ae.json'), url_ae(_m, _d, 'ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_ae.append(_metric)

    for _d, _lm in list(product(DATA, LM_QG)) + list(product(DATA_ML, LM_QG_ML)):
        _m = os.path.basename(_lm)

        # QG metrics
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg`](https://huggingface.co/lmqg/{_m}-{_d}-qg)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "QG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg.qg.json'), url_qg(_m, _d, 'qg'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qg.append(_metric)

        try:
            # QAG metrics: Pipeline QAG
            _metric = {
                "Model": f"[`lmqg/{_m}-{_d}-qg`](https://huggingface.co/lmqg/{_m}-{_d}-qg), [`lmqg/{_m}-{_d}-ae`](https://huggingface.co/lmqg/{_m}-{_d}-ae)",
                "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
                "Type": "Pipeline QAG",
                "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
                "Language": LANG_MAP[_d]
            }
            tmp = download(pj(TMP_DIR, f'{_m}.{_d}.pipeline.qag.json'), url_pipeline(_m, _d, 'qg'))
            _metric.update(
                {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
                 sorted(tmp['test'].keys())})
            metrics_qag.append(_metric)
        except Exception:
            print(f"{_m}-{_d}-qg")
            pass

        # QAG metrics: QG with reference answer
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg`](https://huggingface.co/lmqg/{_m}-{_d}-qg)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "QG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg.qag.json'), url_reference(_m, _d, 'qg'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qag.append(_metric)

    for _d, _lm in list(product(DATA, LM_QG_AE)) + list(product(DATA_ML, LM_QG_AE_ML)):
        _m = os.path.basename(_lm)

        # AE metrics: multitask QAG model
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "Multitask QAG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.ae.json'), url_ae(_m, _d, 'qg-ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_ae.append(_metric)

        # QG metrics: multitask QAG model
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "Multitask QAG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.qg.json'), url_qg(_m, _d, 'qg-ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qg.append(_metric)

        # QAG metrics: multitask QAG model
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qg-ae`](https://huggingface.co/lmqg/{_m}-{_d}-qg-ae)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "Multitask QAG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qg-ae.qag.json'), url_multitask(_m, _d, 'qg-ae'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qag.append(_metric)

    for _d, _lm in list(product(DATA, LM_QAG)) + list(product(DATA_ML, LM_QAG_ML)):
        _m = os.path.basename(_lm)

        # QAG metrics: e2e QAG model
        _metric = {
            "Model": f"[`lmqg/{_m}-{_d}-qag`](https://huggingface.co/lmqg/{_m}-{_d}-qag)",
            "Data": f"[`lmqg/qg_{_d}`](https://huggingface.co/datasets/lmqg/qg_{_d})",
            "Type": "End2end QAG",
            "Language Model": f"[`{_lm}`](https://huggingface.co/{_lm})",
            "Language": LANG_MAP[_d]
        }
        tmp = download(pj(TMP_DIR, f'{_m}.{_d}.qag.qag.json'), url_qag(_m, _d, 'qag'))
        _metric.update(
            {k: 100 * tmp['test'][k] if k not in METRIC_PERC else tmp['test'][k] for k in
             sorted(tmp['test'].keys())})
        metrics_qag.append(_metric)

    print('- Answer Extraction\n')
    df = pd.DataFrame(metrics_ae).round(2).sort_values(by=['Data', 'Language Model', 'Type'])
    df.pop('Bleu_1')
    df.pop('Bleu_2')
    df.pop('Bleu_3')
    df.pop('Bleu_4')
    df.pop('ROUGE_L')
    df.pop('MoverScore')
    df.pop('METEOR')
    df.pop('BERTScore')
    df = df.sort_values(by=["Language", "Language Model", "Type"])
    df = df[df["Type"] == "AE"]
    df.pop("Type")
    print(df.to_markdown(index=False), '\n\n')
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, "summary.ae.csv"), index=False)

    df = pd.DataFrame(metrics_qg).round(2).sort_values(by=['Data', 'Language Model', 'Type'])
    df.pop('Bleu_1')
    df.pop('Bleu_2')
    df.pop('Bleu_3')
    df.pop('Bleu_4')
    df['ROUGE-L'] = df.pop('ROUGE_L')
    df = df.sort_values(by=["Language", "Language Model", "Type"])
    df = df[df["Type"] == "QG"]
    df.pop("Type")
    print('- Question Generation\n')
    print(df.to_markdown(index=False), '\n\n')
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, "summary.qg.csv"), index=False)

    df = pd.DataFrame(metrics_qag).round(2).sort_values(by=['Data', 'Language Model', 'Type'])
    print('- Question & Answer Pairs Generation\n')
    df.pop('Bleu_1')
    df.pop('Bleu_2')
    df.pop('Bleu_3')
    df.pop('Bleu_4')
    df.pop('ROUGE_L')
    df.pop('MoverScore')
    df.pop('METEOR')
    df.pop('BERTScore')
    df = df[df['Type'] != "QG"]
    df = df.sort_values(by=["Language", "Language Model", "Type"])
    print(df.to_markdown(index=False), '\n\n')
    df['Model'] = [i.split("`")[1] for i in df['Model']]
    df['Data'] = [i.split("`")[1] for i in df['Data']]
    df['Language Model'] = [i.split("`")[1] for i in df['Language Model']]
    df.to_csv(pj(EXPORT_DIR, "summary.qag.csv"), index=False)
    input()

    # multilngual result for paper
    df = pd.read_csv("summary/summary.qag.csv")
    for c in ['Model', 'BERTScore', 'METEOR', 'MoverScore',
              'QAAlignedF1Score (MoverScore)', 'QAAlignedPrecision (MoverScore)',
              'QAAlignedRecall (MoverScore)', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L']:
        df.pop(c)
    df = df[[i not in ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large'] for i in df["Language Model"]]]
    f1 = df.pop("QAAlignedF1Score (BERTScore)")
    pre = df.pop("QAAlignedPrecision (BERTScore)")
    rec = df.pop("QAAlignedRecall (BERTScore)")
    df["QAAligned BERTScore"] = [f"{round(a, 1)} / {round(b, 2)} / {round(c, 3)}" for a, b, c in zip(f1, pre, rec)]
    df['Type'] = [i.replace(" QAG", "") for i in df['Type']]
    df['Language'] = [i.replace("lmqg/qg_", "")[:2].upper() for i in df.pop("Data")]

    def pretty_name(i):
        if i == 'google/mt5-small':
            return "MT5\textsubscript{SMALL}"
        if i == 'google/mt5-base':
            return "MT5\textsubscript{BASE}"
        if i == 'facebook/mbart-large-cc25':
            return "MBART\textsubscript{LARGE}"
        raise ValueError("Unknown model")

    df['Language Model'] = [pretty_name(i) for i in df['Language Model']]

