import os
from statistics import mean
from itertools import product
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

os.makedirs('summary', exist_ok=True)
LMS = ["t5-small-squad", "t5-base-squad", "t5-large-squad", "bart-base-squad", "bart-large-squad"]
DOMAINS = ["amazon", "new_wiki", "nyt", "reddit"]
QAG_TYPES = ['qg_reference', 'pipeline', 'multitask', 'end2end']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
output = []
for d, q, l in product(DOMAINS, QAG_TYPES, LMS):
    _data = load_dataset("lmqg/qa_squadshifts_synthetic", f"{l}.{q}.{d}")
    tmp = {"domain": d, "qag_type": q, "lm": l}
    tmp.update({_split: len(_data[_split]) for _split in ['train', 'validation', 'test']})
    output.append(tmp)
for q, l in product(QAG_TYPES, LMS):
    tmp = {
        "domain": "average",
        "qag_type": q,
        "lm": l,
        "train": mean([i['train'] for i in output if i['qag_type'] == q and i['lm'] == l]),
        "validation": mean([i['validation'] for i in output if i['qag_type'] == q and i['lm'] == l]),
        "test": mean([i['test'] for i in output if i['qag_type'] == q and i['lm'] == l])
    }
    output.append(tmp)
df = pd.DataFrame(output)
df.to_csv('summary/summary.dataset.csv', index=False)
