"""NOT WORKING ANYMORE"""
import os
from os.path import join as pj
import pandas as pd
from matplotlib import pyplot as plt
from itertools import groupby

label_map = {
    'amazon': 'AMZ.',
    'new_wiki': 'Wiki',
    'nyt': 'News',
    'reddit': 'Reddit',
    'books': 'Book',
    'electronics': 'Elec.',
    'grocery': 'Groc.',
    'movies': 'Movie',
    'restaurants': 'Rest.',
    'tripadvisor': 'Trip'
}
pretty_name = {'Bleu_4': 'B4', 'METEOR': 'MTR', 'ROUGE_L': 'R-L', 'BERTScore': 'BS', 'MoverScore': 'MS'}


def get_df(lm, metric):
    _df = pd.read_csv(pj('summary', 'squad_ood.csv'), index_col=0)
    _df = _df[_df.lm == lm]
    __dict = {
        "squad": "zeroshot",
        "subjqa": "ft",
        "squadshifts": "ft",
        "squad + subjqa": "cont",
        "squad + squadshifts": "cont"
    }
    _df['type'] = [__dict[i] for i in _df['train_data']]
    _df['dataset'] = ['SQuADShifts' if 'squadshifts' == i else 'SubjQA' for i in _df['test_data']]
    _df['domain'] = _df['test_data_type']
    _df = _df[[i != 'default' for i in _df['domain']]].sort_values(
        by=["dataset", 'domain', 'type'], ascending=False)
    __df = _df[_df.type == 'cont'][['dataset', 'domain', 'type', metric]]
    __df.columns = ['Data', 'Domain', 'type', 'SQuAD + In-domain']
    __df['SQuAD'] = _df[_df.type == 'zeroshot'][metric].values
    __df['In-domain'] = _df[_df.type == 'ft'][metric].values
    __df.pop('type')
    __df['Domain'] = [label_map[i] for i in __df['Domain']]
    # __df = __df[['Data', 'Domain', 'In-domain Fine-tuning', 'SQuAD Fine-tuning', 'SQuAD + In-domain Fine-tuning']]
    __df = __df[['Data', 'Domain', 'In-domain', 'SQuAD', 'SQuAD + In-domain']]
    return __df


def add_line(ax, xpos, ypos, margin=.1):
    line = plt.Line2D([xpos, xpos], [ypos + margin, ypos - margin], transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level): return [(k, sum(1 for i in g)) for k, g in groupby(my_index.get_level_values(level))]


def plot(lm, export, metric: str = None):
    if metric is None:
        metrics = ['Bleu_4', 'ROUGE_L', 'METEOR', 'BERTScore', 'MoverScore']
        fig = plt.figure(figsize=(6, 8))
        single = False
    else:
        metrics = [metric]
        fig = plt.figure(figsize=(6, 4))
        single = True
    for i, metric in enumerate(metrics):
        ax = plt.subplot(len(metrics) * 100 + 11 + i)
        df = get_df(lm, metric).groupby(['Data', 'Domain']).sum()
        df.plot(kind='bar', ax=fig.gca())

        bars = ax.patches
        hatches = ''.join(h * len(df) for h in 'x/O.')

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Below 3 lines remove default labels
        labels = ['' for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.set_xlabel('')

        if i == len(metrics) - 1:
            # Add line to label
            ypos = -.1
            scale = 1. / df.index.size
            for level in range(df.index.nlevels)[::-1]:
                pos = 0
                for label, rpos in label_len(df.index, level):
                    if single:
                        ax.text(
                            (pos + .5 * rpos) * scale,
                            ypos + 0.04 if single else ypos - 0.05,
                            label,
                            ha='center',
                            transform=ax.transAxes,
                            fontsize=10 if label not in ['SQuADShifts', 'SubjQA'] else 12)
                    else:
                        ax.text(
                            (pos + .5 * rpos) * scale,
                            ypos + 0.04 if single else ypos - 0.05,
                            label,
                            ha='center',
                            transform=ax.transAxes,
                            fontsize=10 if label not in ['SQuADShifts', 'SubjQA'] else 12)
                    if single:
                        add_line(ax, pos * scale, ypos+0.07, margin=0.03)
                    else:
                        add_line(ax, pos * scale, ypos)
                    pos += rpos
                if single:
                    add_line(ax, pos * scale, ypos+0.07, margin=0.03)
                else:
                    add_line(ax, pos * scale, ypos)
                if single:
                    ypos -= .05
                else:
                    ypos -= .2

        if i == 0:
            if single:
                ax.legend(loc='best')
            else:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, fontsize=12)
        else:
            ax.get_legend().remove()
        plt.yticks(fontsize=10)
        plt.ylabel(pretty_name[metric], fontsize=12)
    plt.tight_layout()
    plt.savefig(export)


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    plot('t5-large', pj('figures', 'domain_diff_plot.t5-large.meteor.png'), 'METEOR')
    for model in ['t5-base', 't5-small', 'bart-large', 'bart-base', 't5-large']:
        plot(model, pj('figures', f'domain_diff_plot.{model}.png'))
