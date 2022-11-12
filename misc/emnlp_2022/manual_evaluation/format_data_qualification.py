import pandas as pd
import numpy as np
from itertools import chain


def byte_check(string):
    return ''.join(char for char in string if len(char.encode('utf-8')) < 3)


test_file = 'data/qualification_test.csv'
export_file = 'data/qualification_test.format.csv'
df = pd.read_csv(test_file)[['question (generated)', 'answer', 'passage', 'sentence']]
df = df[df.sentence.notnull()]
list_flat = []
for _, df_sub in df.iterrows():
    passage = df_sub['passage']
    sentence = df_sub['sentence']
    passage_split = passage.split(sentence)
    if len(passage_split) != 2:
        continue
    passage_before, passage_after = passage_split
    answer = df_sub['answer']
    question = df_sub['question (generated)']
    list_flat.append(
        [
            byte_check(passage_before),
            byte_check(sentence),
            byte_check(passage_after),
            byte_check(answer),
            byte_check(question)
        ]
    )


array = np.array(list_flat).reshape(-1, 25)
df = pd.DataFrame(array, columns=list(chain(*[[
    'passage_{}_before'.format(n),
    'sentence_{}'.format(n),
    'passage_{}_after'.format(n),
    'answer_{}'.format(n),
    'question_{}'.format(n)
] for n in range(1, 6)])))
df.to_csv(export_file, index=False)
