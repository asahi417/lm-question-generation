import sys
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scipy import stats


def williams_test(r12, r13, r23, n):
    """The Williams test (Evan J. Williams. 1959. Regression Analysis, volume 14. Wiley, New York, USA)
    https://github.com/inmoonlight/nlp-williams
    A test of whether the population correlation r12 equals the population correlation r13.
    Significant: p < 0.05

    Arguments:
        r12 (float): correlation between x1, x2
        r13 (float): correlation between x1, x3
        r23 (float): correlation between x2, x3
        n (int): size of the population

    Returns:
        t (float): Williams test result
        p (float): p-value of t-dist
    """
    if r12 < r13:
        print('r12 should be larger than r13')
        sys.exit()
    elif n <= 3:
        print('n should be larger than 3')
        sys.exit()
    else:
        K = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
        denominator = np.sqrt(2 * K * (n - 1) / (n - 3) + (((r12 + r13) ** 2) / 4) * ((1 - r23) ** 3))
        numerator = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
        t = numerator / denominator
        p = 1 - stats.t.cdf(t, df=n - 3)  # changed to n-3 on 30/11/14
        return t, p


def get_williams(corr_type: str = 'pearson'):
    # maps = {'BLEU4': 'B4', 'ROUGE-L': 'R-L', 'METEOR': 'MTR', 'BERTScore': 'BS', 'MoverScore': 'MS'}
    auto_auto = pd.read_csv('./correlation/corr.{}.flatten.auto_auto.csv'.format(corr_type), index_col=0)
    manual_auto = pd.read_csv('./correlation/corr.{}.flatten.manual_auto.csv'.format(corr_type), index_col=0)
    auto_metrics = list(auto_auto.columns)
    manual_metrics = list(manual_auto.index)
    p_values = {i: {} for i in manual_metrics}
    for m in manual_metrics:
        matrix = pd.DataFrame(np.zeros((len(auto_metrics), len(auto_metrics))), columns=auto_metrics, index=auto_metrics)
        for a, b in product(auto_metrics, auto_metrics):
            if a == b:
                matrix[a][b] = np.nan
                continue
            r12 = manual_auto[a][m]
            r13 = manual_auto[b][m]
            if r12 < r13:
                matrix[a][b] = np.nan
                continue
            r23 = auto_auto[a][b]
            _, p = williams_test(r12, r13, r23, 3000)
            p_values[m]['{}-{}'.format(a, b)] = p
            matrix[a][b] = round(p, 2)

        print(matrix)
        # for k, v in maps.items():
        #     matrix[v] = matrix.pop(k)
        # matrix = matrix.T
        # for k, v in maps.items():
        #     matrix[v] = matrix.pop(k)
        sns_plot = sns.heatmap(matrix, annot=True, fmt="g", vmax=0.5, cmap='BuGn', cbar=False, annot_kws={"size": 16})
        sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=0)
        sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=90)
        sns_plot.tick_params(labelsize=14)
        plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
        fig = sns_plot.get_figure()
        plt.tight_layout()
        fig.savefig('./correlation/williams.figure.{}.{}.png'.format(corr_type, m))
        fig.clear()

    pd.DataFrame(p_values).to_csv('./correlation/williams.{}.csv'.format(corr_type))


if __name__ == '__main__':
    get_williams('spearman')
    # get_williams('pearson')
