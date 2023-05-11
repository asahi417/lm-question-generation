import os
import json
import requests
from os.path import join as pj
import pandas as pd


LM = ['t5-small', 't5-base', 't5-large', 'facebook/bart-base', 'facebook/bart-large']
LM_MULTITASK = ['t5-small', 't5-base', 't5-large']
ML_LM = ['mt5-small', 'mt5-base', 'facebook/mbart-large-cc25']
ML_LM_MULTITASK = ['mt5-small']
ML = ['squad', 'ruquad', 'jaquad', 'itquad', 'koquad', 'esquad', 'dequad', 'frquad']
TYPES = {
    'subjqa': ["books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"],
    'squadshifts': ["new_wiki", "nyt", "reddit", "amazon"]
}
TMP_DIR = 'metric_files'


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


def get_metric(account: str = 'lmqg',
               model: str = 't5-small',
               train_data: str = 'squad',
               train_data_type: str = None,
               test_data: str = None,
               test_data_type: str = 'default',
               additional_prefix: str = None,
               suffix: str = None):
    model = os.path.basename(model)
    model = f'{model}-{train_data}'
    if additional_prefix is not None:
        model = f'{model}-{additional_prefix}'
    if train_data_type is not None and train_data_type != 'default':
        model = f'{model}-{train_data_type}'
    model = f'{model}-qg'
    if suffix is not None:
        model = f'{model}-{suffix}'
    evl = 'eval'
    test = f'qg_{train_data}'
    if test_data is not None:
        evl = f"{evl}_ood"
        test = f'qg_{test_data}'

    model_link = f"[`{account}/{model}`](https://huggingface.co/{account}/{model})"
    url = f"https://huggingface.co/{account}/{model}/raw/main/{evl}/" \
          f"metric.first.sentence.paragraph_answer.question.lmqg_{test}.{test_data_type}.json"
    print(url)
    filename = pj(TMP_DIR, f'{account}.{model}.{evl}.{test}.{test_data_type}.json')
    tmp = download(filename, url)

    url = f"https://huggingface.co/{account}/{model}/raw/main/trainer_config.json"
    filename = pj(TMP_DIR, f'config.{account}.{model}.{evl}.{test}.{test_data_type}.json')
    config = download(filename, url)
    return {k: 100 * v for k, v in tmp['test'].items()}, config, model_link


def summary_ml():
    output = []
    configs = []
    target_lms = ML_LM
    for lm in target_lms:
        for data in ML:
            # supervised result
            _metric, config, model_link = get_metric(model=lm, train_data=data)
            metric = {
                "model": f"{model_link}",
                "language model": f"[`{lm}`](https://huggingface.co/{lm})",
                'training data': f"[`lmqg/qg_{data}`](https://huggingface.co/datasets/lmqg/qg_{data})",
                'test data': f"[`lmqg/qg_{data}`](https://huggingface.co/datasets/lmqg/qg_{data})",
            }
            configs.append(config)
            metric.update(_metric)
            output.append(metric)
            if data != 'squad':
                _metric, _, model_link = get_metric(model=lm, train_data='squad', test_data=data)
                metric = {
                    "model": f"{model_link}",
                    "language model": f"[`{lm}`](https://huggingface.co/{lm})",
                    'training data': f"[`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)",
                    'test data': f"[`lmqg/qg_{data}`](https://huggingface.co/datasets/lmqg/qg_{data})",
                }
                metric.update(_metric)
                output.append(metric)
    return pd.DataFrame(output), configs


def summary():
    output = []
    configs = []
    data = 'squad'
    target_lms = LM
    for lm in target_lms:
        _metric, config, model_link = get_metric(model=lm, train_data=data)
        metric = {
            "model": f"{model_link}",
            "language model": f"[`{lm}`](https://huggingface.co/{lm})",
            'training data': f"[`lmqg/qg_{data}`](https://huggingface.co/datasets/lmqg/qg_{data})",
            'test data': f"[`lmqg/qg_{data}`](https://huggingface.co/datasets/lmqg/qg_{data})",
            'test data type': 'default'
        }
        metric.update(_metric)
        output.append(metric)
        configs.append(config)
    return pd.DataFrame(output), configs


def summary_ood():
    output = []
    configs = []
    for lm in LM:
        for k, v in TYPES.items():
            for _v in v:

                # squad zeroshot
                _metric, config, model_link = get_metric(model=lm, train_data='squad', test_data=k, test_data_type=_v)
                metric = {
                    "model": f"{model_link}",
                    "language model": f"[`{lm}`](https://huggingface.co/{lm})",
                    'training data': f"[`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)",
                    'test data': f"[`lmqg/qg_{k}`](https://huggingface.co/datasets/lmqg/qg_{k})",
                    'test data type': _v
                }
                metric.update(_metric)
                output.append(metric)

                # squad + in-domain
                _metric, config, model_link = get_metric(
                    account='lmqg', model=lm, train_data=k, train_data_type=_v, test_data_type=_v)
                metric = {
                    "model": f"{model_link}",
                    "language model": f"[`{lm}`](https://huggingface.co/{lm})",
                    'training data': f"[`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) + [`lmqg/qg_{k}`](https://huggingface.co/datasets/lmqg/qg_{k})",
                    'test data': f"[`lmqg/qg_{k}`](https://huggingface.co/datasets/lmqg/qg_{k})",
                    'test data type': _v
                }
                metric.update(_metric)
                output.append(metric)
                configs.append(config)

                # in-domain
                _metric, config, model_link = get_metric(
                    account='research-backup',
                    model=lm,
                    train_data=k,
                    train_data_type=_v,
                    additional_prefix='vanilla',
                    test_data_type=_v)
                metric = {
                    "model": f"{model_link}",
                    "language model": f"[`{lm}`](https://huggingface.co/{lm})",
                    'training data': f"[`lmqg/qg_{k}`](https://huggingface.co/datasets/lmqg/qg_{k})",
                    'test data': f"[`lmqg/qg_{k}`](https://huggingface.co/datasets/lmqg/qg_{k})",
                    'test data type': _v
                }
                metric.update(_metric)
                output.append(metric)
                configs.append(config)

    return pd.DataFrame(output), configs


def summary_squad_ablation():
    output = []
    configs = []
    for lm in LM:

        # paragraph-level
        _metric, config, model_link = get_metric(model=lm, train_data='squad')
        metric = {"model": f"{model_link}", "language model": f"[`{lm}`](https://huggingface.co/{lm})", 'type': 'paragraph-level'}
        metric.update(_metric)
        output.append(metric)

        # sentence-level
        _metric, config, model_link = get_metric(account='research-backup', model=lm, train_data='squad', suffix='no-paragraph')
        metric = {"model": f"{model_link}", "language model": f"[`{lm}`](https://huggingface.co/{lm})", 'type': 'sentence-level'}
        metric.update(_metric)
        output.append(metric)
        configs.append(config)

        # answer-free
        _metric, config, model_link = get_metric(account='research-backup', model=lm, train_data='squad', suffix='no-answer')
        metric = {"model": f"{model_link}", "language model": f"[`{lm}`](https://huggingface.co/{lm})", 'type': 'answer-free'}
        metric.update(_metric)
        output.append(metric)
        configs.append(config)

        # no-parameter optimization
        _metric, config, model_link = get_metric(account='research-backup', model=lm, train_data='squad', suffix='default')
        metric = {"model": f"{model_link}", "language model": f"[`{lm}`](https://huggingface.co/{lm})", 'type': 'no-parameter-optimization'}
        metric.update(_metric)
        output.append(metric)

    return pd.DataFrame(output), configs


def config_formatting(df_config):
    pretty_name = {
        'lmqg/qg_squad': 'SQuAD', 'lmqg/qg_ruquad': 'MLQG-Ru', 'lmqg/qg_jaquad': 'MLQG-Ja',
        'lmqg/qg_itquad': 'MLQG-It', 'lmqg/qg_koquad': 'MLQG-Ko', 'lmqg/qg_esquad': 'MLQG-Es',
        'lmqg/qg_dequad': 'MLQG-De', 'lmqg/qg_frquad': 'MLQG-Fr', 'lmqg/qg_subjqa': 'SubjQA',
        'lmqg/qg_squadshifts': 'SQuADShifts'
    }
    # print(df_config)
    df_config.pop('fp16')
    df_config.pop('max_length')
    df_config.pop('max_length_output')
    df_config.pop('prefix_types')
    df_config.pop('random_seed')
    df_config['type'] = 'paragraph-level QG'
    df_config['Dataset'] = [pretty_name[i] for i in df_config.pop('dataset_path')]
    input_types = df_config.pop('input_types')
    output_types = df_config.pop('output_types')
    df_config['type'][['sentence_answer' in i for i in input_types]] = 'sentence-level QG'
    df_config['type'][['paragraph_sentence' in i and 'answer' not in o for i, o in zip(input_types, output_types)]] = 'answer-free QG'
    df_config['type'][['answer' in i for i in output_types]] = 'multitask QG'
    df_config = df_config[[i != 'multitask QG' for i in df_config['type']]]

    # SQuAD/MLQG
    df_config_mlqg = df_config[['MLQG' in i or 'SQuAD' == i for i in df_config['Dataset']]]
    df_config_mlqg.pop('dataset_name')
    df_config_mlqg = df_config_mlqg.sort_values(by=['Dataset', 'model', 'type'])

    # SubjQA
    df_config_subjqa = df_config[[i == 'SubjQA' for i in df_config['Dataset']]]
    df_config_subjqa.pop('Dataset')
    df_config_subjqa.pop('type')
    df_config_subjqa = df_config_subjqa.sort_values(by=['model', 'dataset_name'])

    # SQuADShifts
    df_config_ss = df_config[[i == 'SQuADShifts' for i in df_config['Dataset']]]
    df_config_ss.pop('Dataset')
    df_config_ss.pop('type')
    df_config_ss = df_config_ss.sort_values(by=['model', 'dataset_name'])

    return df_config_mlqg, df_config_subjqa, df_config_ss


if __name__ == '__main__':
    os.makedirs('summary', exist_ok=True)

    all_config = []

    df, c = summary()
    df.round(2).to_csv(pj('summary', 'squad.csv'), index=False)
    all_config += c

    df, c = summary_ml()
    df.round(2).to_csv(pj('summary', 'mlqg.csv'), index=False)
    all_config += c

    df, c = summary_ood()
    df.round(2).to_csv(pj('summary', 'squad_ood.csv'), index=False)
    all_config += c

    df, c = summary_squad_ablation()
    df.round(2).to_csv(pj('summary', 'squad_ablation.csv'), index=False)
    all_config += c

    os.makedirs('config', exist_ok=True)

    df = pd.DataFrame(all_config)
    c_ml, c_sub, c_ss = config_formatting(df)
    c_ml.to_csv(pj('config', 'main.csv'), index=False)
    c_sub.to_csv(pj('config', 'subjqa.csv'), index=False)
    c_ss.to_csv(pj('config', 'squadshifts.csv'), index=False)
