import pandas as pd

df = pd.read_csv("summary/summary.qae.csv")
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

df_qag = pd.read_csv("summary/summary.qag.csv")
df_qag.pop('Model')
df_qag.pop('Data')
df = pd.merge(left=df, right=df_qag, on=['Language Model', 'Type'])
corr = df.corr()
for c in corr.columns:
    if not c.startswith("QAEval"):
        corr.pop(c)
corr.to_csv("summary/correlation.csv")
