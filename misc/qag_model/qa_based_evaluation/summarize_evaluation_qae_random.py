import os
import json
from glob import glob
import pandas as pd

os.makedirs('summary', exist_ok=True)
output = []
for d in ['amazon', 'new_wiki', 'nyt', 'reddit']:
    for _file in glob(f"qa_eval_output/random_sampling*/*.{d}/test_result.json"):
        with open(_file) as f:
            data = json.load(f)
            data['Domain'] = d
            data['model'] = _file.split('qa_eval_output/random_sampling.')[-1].split('/')[0]
            output.append(data)
df = pd.DataFrame(output)
df['QAEval (F1)'] = df.pop('eval_f1')
df['QAEval (EM)'] = df.pop('eval_exact_match')
# df.to_csv("summary.qae.random.csv", index=False)

df.to_csv("summary/summary.qae.random.csv", index=False)
