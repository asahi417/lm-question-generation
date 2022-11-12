"""
python williams.py --r12 0.65 --r13 0.55 --r23 0.3 --n 3000
"""
import os
import json
from itertools import chain
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

os.makedirs('correlation', exist_ok=True)

evaluation_types = ['correctness', 'grammaticality', 'understandability']
evaluation_types_name = {'correctness': 'answerability'}
evaluation_types_fixed = ['answerability', 'grammaticality', 'understandability']
# metrics_norm = ['BLEU4', 'ROUGE-L', 'METEOR', 'BERTScore', 'MoverScore']
# metrics_norm_alias = {'Bleu_4': 'BLEU4', 'METEOR': 'METEOR', 'ROUGE_L': 'ROUGE-L', 'BERTScore': 'BERTScore',
#                       'MoverScore': 'MoverScore'}
metrics_norm = ['B4', 'R-L', 'MTR', 'BS', 'MS']
metrics_norm_alias = {'Bleu_4': 'B4', 'METEOR': 'MTR', 'ROUGE_L': 'R-L', 'BERTScore': 'BS',
                      'MoverScore': 'MS'}



def mean(_list):
    return sum(_list)/len(_list)


def calculate_pvalues(df, corr_type='spearman'):
    df = df.dropna()._get_numeric_data()
    df_cols = pd.DataFrame(columns=df.columns)
    p_values = df_cols.transpose().join(df_cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if corr_type == 'pearson':
                p_values[r][c] = pearsonr(df[r], df[c])[1]
            elif corr_type == 'spearman':
                p_values[r][c] = spearmanr(df[r], df[c])[1]
            else:
                raise ValueError('unknown: {}'.format(corr_type))
    return p_values


def save_corr(df, prefix, corr_type='spearman'):
    if 'reference_norm' in df.columns:
        df.pop('reference_norm')
    if 'reference_raw' in df.columns:
        df.pop('reference_raw')
    if 'worker' in df.columns:
        df.pop('worker')
    # df = df/df.max(axis=0)
    corr_manual_auto = df.corr(corr_type)[evaluation_types_fixed].T[metrics_norm]
    corr_manual_auto.to_csv('./correlation/corr.{}.{}.manual_auto.csv'.format(corr_type, prefix))
    calculate_pvalues(df, corr_type)[evaluation_types_fixed].T[metrics_norm].to_csv(
        './correlation/corr.{}.{}.manual_auto.p.csv'.format(corr_type, prefix))
    corr_manual_manual = df.corr(corr_type)[evaluation_types_fixed].T[evaluation_types_fixed]
    corr_manual_manual.to_csv('./correlation/corr.{}.{}.manual_manual.csv'.format(corr_type, prefix))
    calculate_pvalues(df, corr_type)[evaluation_types_fixed].T[evaluation_types_fixed].to_csv(
        './correlation/corr.{}.{}.manual_manual.p.csv'.format(corr_type, prefix))
    corr_auto_auto = df.corr(corr_type)[metrics_norm].T[metrics_norm]
    corr_auto_auto.to_csv('./correlation/corr.{}.{}.auto_auto.csv'.format(corr_type, prefix))
    calculate_pvalues(df, corr_type)[metrics_norm].T[metrics_norm].to_csv('./correlation/corr.{}.{}.auto_auto.p.csv'.format(
        corr_type, prefix))
    # heatmap
    get_heatmap(corr_auto_auto, './correlation/corr.{}.{}.auto_auto.png'.format(corr_type, prefix))
    get_heatmap(corr_manual_auto, './correlation/corr.{}.{}.manual_auto.png'.format(corr_type, prefix))
    get_heatmap(corr_manual_manual, './correlation/corr.{}.{}.manual_manual.png'.format(corr_type, prefix))


def get_heatmap(df, path_to_save):
    # plot heatmap
    fig = plt.figure()
    fig.clear()
    df = df.astype(float).round(2)
    sns_plot = sns.heatmap(df, annot=True, fmt="g", cmap='BuGn', annot_kws={"size": 13})
    # sns_plot = sns.heatmap(df, annot=True, fmt="g", square=True)
    # sns_plot = sns.heatmap(df, annot=True, fmt="g", cbar=False)
    # sns_plot = sns.heatmap(df, annot=True, fmt="g", cbar=False, square=True)
    # ax1 = sns.heatmap(hmap, cbar=0, cmap="YlGnBu", linewidths=2, ax=ax0, vmax=3000, vmin=0, square=True)
    sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=60)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=60)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig(path_to_save)


# Formatting data
with open('data/final_result.json') as f:
    data = json.load(f)
for k, v in data.items():
    for _v in v:
        for _k, __v in metrics_norm_alias.items():
            if _k in _v:
                _v[__v] = _v.pop(_k)
        for e in evaluation_types:
            _v[e] = mean(_v[e]) / 3

# Get high level stats
df_high_level = {}
for k, v in data.items():
    df_high_level[k] = {e: mean([_v[e] for _v in v]) for e in evaluation_types + metrics_norm}
df_high_level = pd.DataFrame(df_high_level)
df_high_level = df_high_level.T
# save_corr(df_high_level, 'model_wise')
for k in metrics_norm:
    df_high_level[k] = (df_high_level[k] * 100).round(2)
for k in evaluation_types:
    df_high_level[k] = (df_high_level[k] * 3).round(2)

df_high_level.to_csv('./correlation/summary.model_wise.csv')

# Get flatten stats
flatten = list(chain(*[list(data[k]) for k in data.keys()]))
df_flatten = pd.DataFrame(flatten)
for k, v in evaluation_types_name.items():
    if k in df_flatten.columns:
        df_flatten[v] = df_flatten.pop(k)
save_corr(df_flatten, 'flatten', 'pearson')
save_corr(df_flatten, 'flatten', 'spearman')
