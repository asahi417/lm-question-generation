import os
from statistics import mean
from os.path import join as pj
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


multiquad = [f'lmqg/qg_{la}quad' for la in ['ja', 'es', 'de', 'ru', 'ko', 'it', 'fr']]
tokenizer_en = AutoTokenizer.from_pretrained('t5-small')
tokenizer_multi = AutoTokenizer.from_pretrained('google/mt5-small')
dfs = []
samples = []
sample_size = 20

for d in ['lmqg/qg_squad', 'lmqg/qg_subjqa', 'lmqg/qg_squadshifts'] + multiquad:   # 'lmqg/qg_newsqa'
    if d == 'lmqg/qg_subjqa':
        data_all = [load_dataset(d, i) for i in
                    ["books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"]]
        data_name = [f"{d} ({i})" for i in
                     ["books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"]]
    elif d == 'lmqg/qg_squadshifts':
        data_all = [load_dataset(d, i) for i in ["new_wiki", "nyt", "reddit", "amazon"]]
        data_name = [f"{d} ({i})" for i in ["new_wiki", "nyt", "reddit", "amazon"]]
    else:
        data_all = [load_dataset(d)]
        data_name = [d]
    tokenizer = tokenizer_en if d not in multiquad else tokenizer_multi
    for data, _d in zip(data_all, data_name):
        for s, data_instance in data.items():
            dfs.append({
                'data': _d,
                'split': s,
                'question (average number of token)': mean([len(tokenizer.tokenize(i)) for i in data_instance['question']]),
                'paragraph (average number of token)': mean([len(tokenizer.tokenize(i)) for i in data_instance['paragraph']]),
                'answer (average number of token)': mean([len(tokenizer.tokenize(i)) for i in data_instance['answer']]),
                'sentence (average number of token)': mean([len(tokenizer.tokenize(i)) for i in data_instance['sentence']]),
                'question (average number of character)': mean([len(i) for i in data_instance['question']]),
                'paragraph (average number of character)': mean([len(i) for i in data_instance['paragraph']]),
                'answer (average number of character)': mean([len(i) for i in data_instance['answer']]),
                'sentence (average number of character)': mean([len(i) for i in data_instance['sentence']]),
                'number of instances': len(data_instance['question'])
            })
            if s == 'test':
                data_instance = data_instance.shuffle()
                df = pd.DataFrame(data_instance[:sample_size])
                df['data'] = _d
                samples.append(df)


os.makedirs('summary', exist_ok=True)
pd.DataFrame(dfs).to_csv(pj('summary', 'dataset_stats.csv'), index=False)
pd.concat(samples)[['answer', 'question', 'sentence', 'paragraph', 'data', 'domain']].to_csv(pj('summary', 'data_samples.csv'), index=False)
