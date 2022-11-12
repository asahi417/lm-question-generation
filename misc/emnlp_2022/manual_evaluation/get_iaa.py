import json
import os
from itertools import chain, combinations
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

no_lstm = False
os.makedirs('iaa', exist_ok=True)


def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    _a = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    e = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        e += count

    return round((_a - e) / (1 - e + 1e-20), 4)


def fleiss_kappa(mat):
    """Computes Fleiss' kappa for group of annotators.
    :param mat: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'mat[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    n, k = mat.shape  # n is # of items, k is # of categories
    n_annotators = float(np.sum(mat[0, :]))  # # of annotators
    tot_annotations = n * n_annotators  # the total # of annotations
    category_sum = np.sum(mat, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    p_bar_e = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    p = (np.sum(mat * mat, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    p_bar = np.sum(p) / n  # add all observed agreement chances per item and divide by amount of items

    return round((p_bar - p_bar_e) / (1 - p_bar_e), 4)


with open('data/final_result.json') as f:
    tmp = json.load(f)
    # if no_lstm:
    #     tmp.pop('LSTM')
    tmp = list(chain(*[list(tmp[k]) for k in tmp.keys()]))

metric_types = ['grammaticality', 'understandability', 'correctness']
freq = None
fleiss_kappa_value = {}
for m in metric_types:
    mat_fless = []
    data = {}
    for _n, v in enumerate(tmp):
        tmp_mat = [0] * 3
        for _w, _s in zip(v['worker'], v[m]):
            tmp_mat[_s - 1] += 1
            if _w not in data:
                data[_w] = [[_n, _s]]
            else:
                data[_w].append([_n, _s])
        mat_fless.append(tmp_mat)
    if freq is None:
        freq = {k: len(data[k]) for k in sorted(data.keys())}
    fleiss_kappa_value[m] = fleiss_kappa(np.array(mat_fless))

    workers = sorted(data.keys())
    mat_cohen = np.zeros((len(workers), len(workers)))
    cofreq = np.zeros((len(workers), len(workers)))
    for a, b in combinations(workers, 2):
        index_a = [x for x, y in data[a]]
        index_b = [x for x, y in data[b]]
        intersection = list(set(index_a).intersection(set(index_b)))
        cofreq[b, a] = cofreq[a, b] = len(intersection)
        cofreq[a, a] = len(index_a)
        cofreq[b, b] = len(index_b)
        if len(intersection) == 0:
            continue
        data_a = sorted(data[a], key=lambda x: x[0])
        data_b = sorted(data[b], key=lambda x: x[0])
        data_a = [y for x, y in data_a if x in intersection]
        data_b = [y for x, y in data_b if x in intersection]
        mat_cohen[a, b] = cohen_kappa(data_a, data_b)
        mat_cohen[b, a] = cohen_kappa(data_a, data_b)
    sns_plot = sns.heatmap(pd.DataFrame(mat_cohen), annot=False, fmt="g", vmax=1, cmap='BuGn')
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('./iaa/cohen.{}.png'.format(m))
    fig.clear()

# plot frequency
ax = pd.DataFrame(sorted(freq.values(), reverse=True), columns=['number of annotation']).plot.bar(legend=False)
ax.set_xlabel("Annotators")
ax.set_ylabel("Frequency")
fig = ax.get_figure()
plt.tight_layout()
fig.savefig('./iaa/frequency.png')

# plot heatmap of worker co-occurrence

sns_plot = sns.heatmap(pd.DataFrame(cofreq), annot=False, fmt="g", cmap='BuGn')
fig = sns_plot.get_figure()
plt.tight_layout()
fig.savefig('./iaa/worker_cooccurrence.png')

with open('iaa/fleiss.json', 'w') as f:
    json.dump(fleiss_kappa_value, f)


