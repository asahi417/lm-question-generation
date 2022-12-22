import os
import json
from glob import glob
import pandas as pd

os.makedirs('summary', exist_ok=True)
output = []
for _file in glob("qa_eval_output/*/*/test_result.json"):
    _file_name, _domain, _ = _file.split("/")[1:]
    with open(_file) as f:
        tmp = json.load(f)
    tmp['domain'] = _domain.split(".")[-1]
    tmp['model'] = 'gold' if _file_name.startswith("gold_qa") else _file_name.split('.')[1]
    if tmp['model'] != 'gold':
        tmp['qag_type'] = _file_name.split('.')[-1]
    output.append(tmp)
df = pd.DataFrame(output)
ave = [dict([('model', m), ("qag_type", q), ('domain', 'average')] + list(g.mean().items())) for (m, q), g in df.groupby(['model', 'qag_type'])]
ave.append({"model": "gold", "domain": "average", "eval_f1": df[df.model == "gold"].mean().eval_f1,
            "eval_exact_match": df[df.model == "gold"].mean().eval_exact_match})
df = pd.concat([df, pd.DataFrame(ave)[df.columns]])
model_order = ['gold', 'bart-base-squad', 'bart-large-squad', 't5-small-squad', 't5-base-squad', 't5-large-squad']
qag_type_order = ['gold', 'qg_reference', 'pipeline', 'multitask', 'end2end']
df['sort_model'] = [model_order.index(i) for i in df['model']]
df['sort_qag_type'] = [qag_type_order.index(i) if type(i) is str else -1 for i in df['qag_type']]
df = df.sort_values(by=['domain', 'sort_model', 'sort_qag_type']).round(2)
df.pop('sort_model')
df.pop('sort_qag_type')

df.to_csv("summary/summary.qae.csv", index=False)
df = df[df.domain == 'average']
df = df[df.model != 'gold']


def pretty_name(_name):
    if _name == 'end2end':
        return "End2end QAG"
    elif _name == 'qg_reference':
        return "QG"
    elif _name == 'pipeline':
        return "Pipeline QAG"
    elif _name == 'multitask':
        return "Multitask QAG"
    raise ValueError(_name)


df['Type'] = df['qag_type'].apply(lambda x: pretty_name(x))
df['Language Model'] = df['model'].apply(lambda x: f"facebook/{x.replace('-squad', '')}" if x.startswith('bart') else x.replace('-squad', ''))
df['QAEval (F1)'] = df.pop('eval_f1')
df['QAEval (EM)'] = df.pop('eval_exact_match')
df.pop("qag_type")
df.pop("domain")
df.pop("model")
df.to_csv("summary/summary.qae.average.csv", index=False)