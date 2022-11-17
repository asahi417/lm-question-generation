import os
import json
from glob import glob
import pandas as pd

output = []
for i in glob("qa_eval_output/silver_qa.*/*/test_result.json") + glob("qa_eval_output/gold_qa/*/test_result.json"):
    with open(i) as f:
        tmp = json.load(f)
    path = i.split('/')
    lm, data, data_name = path[2].split('.')
    tmp['qg_model'] = path[1].replace('silver_qa.', '')
    tmp['data'] = data
    tmp['data_name'] = data_name
    tmp['qa_model'] = lm
    output.append(tmp)
df = pd.DataFrame(output)
if os.path.exists("./summary.qae.csv"):
    _df = pd.read_csv("./summary.qae.csv", index_col=0)
    df = pd.concat([_df, df])
df.to_csv("./summary.qae.csv", index=False)
