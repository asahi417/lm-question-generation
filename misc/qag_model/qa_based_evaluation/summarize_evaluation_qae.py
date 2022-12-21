""" QAEval"""
import os
import json
from glob import glob
import pandas as pd
os.makedirs('summary', exist_ok=True)

output = []
for _file in glob("qa_eval_output/*/*/test_result.json"):
    _file_name, _domain, _ = _file.split("/")[1:]
    print(_file)
    with open(_file) as f:
        tmp = json.load(f)
    tmp['domain'] = _domain.split(".")[1]
    tmp['model'] = 'gold' if _file_name.startswith("gold_qa") else _file_name.split('.')[1]
    if tmp['model'] != 'gold':
        tmp['qag_type'] = _file_name.split('.')[2]
    output.append(tmp)
df = pd.DataFrame(output).sort_values(by=['domain', 'model', 'qag_type']).round(2)
# df.to_csv("summary/summary.qae.csv", index=False)
