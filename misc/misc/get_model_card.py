import pandas as pd
from huggingface_hub import HfApi

header = 'https://huggingface.co'


def parse(alias):
    if "mbart" in alias:
        lm = f"[facebook/mbart-large-cc25]({header}/facebook/mbart-large-cc25)"
        _alias = alias.split('mbart-large-cc25-')[1]
    elif 'bart-base' in alias:
        lm = f"[facebook/bart-base]({header}/facebook/bart-base)"
        _alias = alias.split('bart-base-')[1]
    elif 'bart-large' in alias:
        lm = f"[facebook/bart-large]({header}/facebook/bart-large)"
        _alias = alias.split('bart-large-')[1]
    elif 'flan-t5-small' in alias:
        lm = f"[google/flan-t5-small]({header}/google/flan-t5-small)"
        _alias = alias.split('flan-t5-small-')[1]
    elif 'flan-t5-base' in alias:
        lm = f"[google/flan-t5-base]({header}/google/flan-t5-base)"
        _alias = alias.split('flan-t5-base-')[1]
    elif 'flan-t5-large' in alias:
        lm = f"[google/flan-t5-large]({header}/google/flan-t5-large)"
        _alias = alias.split('flan-t5-large-')[1]
    elif 'mt5-small' in alias:
        lm = f"[google/mt5-small]({header}/google/mt5-small)"
        _alias = alias.split('mt5-small-')[1]
    elif 'mt5-base' in alias:
        lm = f"[google/mt5-base]({header}/google/mt5-base)"
        _alias = alias.split('mt5-base-')[1]
    elif 't5-small' in alias:
        lm = f"[t5-small]({header}/t5-small)"
        _alias = alias.split('t5-small-')[1]
    elif 't5-base' in alias:
        lm = f"[t5-base]({header}/t5-base)"
        _alias = alias.split('t5-base-')[1]
    elif 't5-large' in alias:
        lm = f"[t5-large]({header}/t5-large)"
        _alias = alias.split('t5-large-')[1]
    else:
        raise ValueError(alias)

    if _alias.startswith('squadshifts'):
        dataset = f"[lmqg/qg_squadshifts ({_alias.split('-')[1]})]({header}/datasets/lmqg/qg_squadshifts)"
    elif _alias.startswith('subjqa'):
        dataset = f"[lmqg/qg_subjqa ({_alias.split('-')[1]})]({header}/datasets/lmqg/qg_subjqa)"
    else:
        if _alias.endswith('qag'):
            dataset = f"[lmqg/qag_{_alias.split('-')[0]}]({header}/datasets/lmqg/qag_{_alias.split('-')[0]})"
        else:
            dataset = f"[lmqg/qg_{_alias.split('-')[0]}]({header}/datasets/lmqg/qg_{_alias.split('-')[0]})"
    if _alias.endswith('qag'):
        task = 'End2end QAG'
    elif _alias.endswith('qg-ae'):
        task = 'Multitask QAG'
    elif _alias.endswith('qg'):
        task = 'QG'
    elif _alias.endswith('ae'):
        task = 'AE'
    elif _alias.endswith('qa'):
        task = 'QA'
    else:
        raise ValueError(_alias)
    if 'mt5' in alias or 'mbart' in alias:
        la = _alias[:2] if _alias.split('-')[0] != 'squad' else "en"
    else:
        la = 'en'
    return {"Model": f"[{alias}]({header}/{alias})", "Language": la, "LM": lm, "Task": task, "Dataset": dataset}


api = HfApi()

models = sorted([model.id for model in api.list_models(author='lmqg') if 'tweetqa' not in model.id])
ablation_models = pd.DataFrame([parse(i) for i in models if 'subjqa' in i or 'squadshifts' in i])
ablation_models.pop("Task")
ablation_models = ablation_models.to_markdown(index=False)
print(ablation_models)
input()
models_full = pd.DataFrame([parse(i) for i in models if 'subjqa' not in i and 'squadshifts' not in i])
models = models_full[models_full['Task'] != 'QA']
for la, g in models.groupby('Language'):
    g = g.sort_values(by=['LM', 'Task'])
    g.pop("Language")
    la = str(la).replace("ko", "Korean").replace("ru", "Russian").replace("it", "Italian").replace("fr", "French").replace("es", "Spanish").replace("de", "German").replace("ja", "Japanese").replace("en", "English")
    print(f"### {la}\n")
    print(g.to_markdown(index=False))
    print()

models = models_full[models_full['Task'] == 'QA']
models.pop("Task")
models = models.sort_values(by=['Language', 'LM'])
print(models.to_markdown(index=False))
