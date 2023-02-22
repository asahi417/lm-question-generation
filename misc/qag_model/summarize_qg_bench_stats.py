import os
import requests
from statistics import mean
from datasets import load_dataset
import pandas as pd

TMP_DIR = 'metric_files'


def download(filename, url):
    try:
        with open(filename) as f:
            return [i for i in f.read().split('\n') if len(i) > 0]
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(url)
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    with open(filename) as f:
        return [i for i in f.read().split('\n') if len(i) > 0]


full_data = ["qag_squad", "qag_dequad", "qag_esquad", "qag_frquad", "qag_itquad", "qag_jaquad", "qag_koquad", "qag_ruquad"]
output = []
for data in full_data:
    dataset = load_dataset(f"lmqg/{data}", split="test")
    output.append({
        "language": data.split("_")[1][:2] if data != "qag_squad" else "en",
        "size": mean([len(i.split("|")) for i in dataset['questions_answers']]),
        "model": "reference",
        "type": "reference"
    })



# models_en = ['t5-small', 't5-base', 't5-large', 'bart-base', 'bart-large', 'flan-t5-small', 'flan-t5-base', 'flan-t5-large']
models_en = ['t5-small', 't5-base', 't5-large', 'bart-base', 'bart-large', 'flan-t5-small', 'flan-t5-base']
models_multi = ['mt5-small', 'mt5-base', 'mbart-large-cc25']

for data in full_data:
    data = data.split('_')[1]
    if data == "squad":
        for m in models_en:
            _url = f"https://huggingface.co/lmqg/{m}-{data}-qg/raw/main/eval_pipeline/samples.test.hyp.paragraph.questions_answers.lmqg_qg_{data}.default.lmqg_{m}-{data}-ae.txt"
            pred = download(f"{TMP_DIR}/test_prediction.pipeline.{m}.{data}.txt", _url)
            output.append({
                "language": "en",
                "size": mean([len(i.split("|")) for i in pred]) if len(pred) > 0 else 0,
                "model": m,
                "type": "pipeline"
            })
    else:
        for m in models_multi:
            _url = f"https://huggingface.co/lmqg/{m}-{data}-qg/raw/main/eval_pipeline/samples.test.hyp.paragraph.questions_answers.lmqg_qg_{data}.default.lmqg_{m}-{data}-ae.txt"
            pred = download(f"{TMP_DIR}/test_prediction.pipeline.{m}.{data}.txt", _url)
            output.append({
                "language": data[:2],
                "size": mean([len(i.split("|")) for i in pred]) if len(pred) > 0 else 0,
                "model": m,
                "type": "pipeline"
            })


df = pd.DataFrame(output)
df.to_csv("summary/summary.qag.size.csv", index=False)