""" Get Basic Evaluation Metric for QG/AE/QA """
import os
import json
import requests
import pandas as pd

tmp_dir = 'metric_files'


def download(model):
    url = f"https://huggingface.co/lmqg/{model}/raw/main/trainer_config.json"
    filename = f"{tmp_dir}/config.{model}.json"
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
        out = json.load(f)
    out['model'] = model
    return out


if __name__ == '__main__':
    configs = []
    for m in ['t5-small-squad', 't5-base-squad', 't5-large-squad', 'bart-base-squad', 'bart-large-squad']:
        for t in ['qg', 'ae', 'qag', 'qg-ae']:
            configs.append(download(f"{m}-{t}"))

    df = pd.DataFrame(configs)
    df.pop("fp16")
    df.pop("random_seed")
    df.pop("output_types")
    df.pop("input_types")
    df.pop("dataset_name")
    df.pop("prefix_types")
    df.pop("max_length")
    df.pop("max_length_output")
    df.pop("dataset_path")
    df['batch'] = df.pop("batch") * df.pop("gradient_accumulation_steps")
    print(df)
    df.to_csv('summary/config.csv', index=False)
