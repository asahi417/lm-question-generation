import os
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
    if q == 'multitask' and not l.startswith('t5'):
        continue
    _data = load_dataset("lmqg/qa_squadshifts_synthetic", f"{l}.{q}.{d}")
    tmp = {"domain": d, "qag_type": q, "lm": l}
    tmp.update({_split: len(_data[_split]) for _split in _data})
    output.append(tmp)
df = pd.DataFrame(output)
df.to_csv('summary/summary.dataset.csv', index=False)
