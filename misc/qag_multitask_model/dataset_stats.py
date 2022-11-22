import pandas as pd
from datasets import load_dataset

LMS = ["t5-small-squad",
       "t5-base-squad",
       "t5-large-squad",
       "t5-small-squad-multitask",
       "t5-base-squad-multitask",
       "t5-large-squad-multitask",
       "bart-base-squad",
       "bart-large-squad"]
DOMAINS = ["new_wiki", "reddit", "amazon", "nyt"]

table = []
for lm in LMS:
    for d in DOMAINS:
        data = load_dataset("lmqg/qa_squadshifts_pseudo", f"{d}.{lm}")
        table.append({"lm": lm, "domain": d, "train": len(data["train"]), "validation": len(data["validation"]), "test": len(data["test"])})
    pd.DataFrame(table).to_csv('summary.dataset.csv', index=False)