import os
import json
from random import seed, shuffle
from datasets import load_dataset

seed_size = 25
stats = {i: {s: len(load_dataset('lmqg/qa_squadshifts', i, split=s)) for s in ['train', 'validation']} for i in ['amazon', 'new_wiki', 'nyt', 'reddit']}
output_dir = 'qa_squadshifts_synthetic_random'
for lm in ['t5-small', 't5-base', 't5-large', 'bart-base', 'bart-large']:
    for anchor in ['amazon', 'new_wiki', 'nyt', 'reddit']:
        for split in ['train', 'validation']:
            for qag in ['end2end', 'pipeline', 'multitask']:
                if lm.startswith('bart') and qag == 'multitask':
                    continue
                dataset = [i for i in load_dataset('lmqg/qa_squadshifts_synthetic', f"{lm}-squad.{qag}.{anchor}", split=split)]
                for s in range(seed_size):
                    if os.path.exists(f"{output_dir}/seed_{s}/{lm}-squad.{qag}.{anchor}/{split}.jsonl"):
                        continue
                    seed(s)
                    shuffle(dataset)
                    os.makedirs(f"{output_dir}/seed_{s}/{lm}-squad.{qag}.{anchor}", exist_ok=True)
                    with open(f"{output_dir}/seed_{s}/{lm}-squad.{qag}.{anchor}/{split}.jsonl", 'w') as f:
                        f.write("\n".join([json.dumps(i) for i in dataset[:stats[anchor][split]]]))


# from datasets import load_dataset
# seed_size = 10
# for lm in ['t5-large']:
#     for anchor in ['amazon', 'new_wiki', 'nyt', 'reddit']:
#         for split in ['train', 'validation', 'test']:
#             for qag in ['pipeline']:
#                 if lm.startswith('bart') and qag == 'multitask':
#                     continue
#                 for s in range(seed_size):
#                     load_dataset('lmqg/qa_squadshifts_synthetic_random', f"{lm}-squad.{qag}.{anchor}.seed_{s}", split=split)
