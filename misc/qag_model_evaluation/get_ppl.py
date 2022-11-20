""" PPl for answer & question
python misc/qag_model_evaluation/get_ppl.py -m "t5-small-squad-multitask" -b 128
python misc/qag_model_evaluation/get_ppl.py -m "t5-base-squad-multitask" -b 32
python misc/qag_model_evaluation/get_ppl.py -m "t5-large-squad-multitask" -b 16
"""
import argparse
import os
import json
import logging
from lmqg import TransformersQG
from datasets import load_dataset
import pandas as pd


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
domains = ['amazon', 'new_wiki', 'nyt', 'reddit']


def main(m, batch, overwrite: bool = False):

    for d in domains:
        for split in ['train', 'validation']:
            dataset_gold = load_dataset("lmqg/qa_squadshifts", d, split=split)
            with open(f"qa_squadshifts_pseudo/{m}.{d}/{split}.jsonl") as f:
                dataset_split = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
            df = pd.DataFrame(dataset_split)
            for target in ['answer', 'question']:
                output_file = f"qa_squadshifts_pseudo/{m}.{d}/perplexity_{target}.{split}.json"

                if not os.path.exists(output_file) or overwrite:
                    model = TransformersQG(f"lmqg/{m}")

                    print(f"Computing perplexity for {target}: `{m}`, domain: `{d}`, split: `{split}`")
                    ppl = model.get_perplexity(
                        list_question=[i['question'] for i in dataset_split],
                        list_context=[i['context'] for i in dataset_split],
                        list_answer=[i['answers']['text'][0] for i in dataset_split],
                        target_output=target,
                        batch_size=batch
                    )
                    with open(output_file, "w") as f:
                        json.dump({f"perplexity_{target}": ppl}, f)

                with open(output_file) as f:
                    ppl = json.load(f)[f"perplexity_{target}"]
                assert len(dataset_split) == len(ppl), f"{len(dataset_split)} != {len(ppl)}"
                df[f'perplexity_{target}'] = ppl

            target_col = [i for i in df.columns if 'perplexity' not in i]
            # method 1: the same amount of data by ppl
            for target in ['answer', 'question']:
                filtered_1 = df.sort_values(by=f'perplexity_{target}').head(len(dataset_gold))[target_col]
                filtered_1_d = list(filtered_1.T.to_dict().values())
                with open(f"qa_squadshifts_pseudo/{m}.{d}/{split}.filtered.perplexity_{target}.jsonl", "w") as f:
                    f.write('\n'.join([json.dumps(i) for i in filtered_1_d]))
            #
            # # method 2: the same amount for each paragraph
            # for target in ['answer', 'question']:
            #     freq = {k: len(v) for k, v in dataset_gold.to_pandas().groupby('context')}
            #     df_list = []
            #     for c, _df in df.groupby('context'):
            #         assert len(_df) >= freq[c], f"{len(_df)} < {freq[c]}"
            #         df_list.append(_df.sort_values(by=f'perplexity_{target}').head(freq[c]))
            #     filtered_2 = pd.concat(df_list)[target_col]
            #     filtered_2_d = list(filtered_2.T.to_dict().values())
            #     with open(f"qa_squadshifts_pseudo/{m}.{d}/{split}.filtered.perplexity_{target}_partition.jsonl", "w") as f:
            #         f.write('\n'.join([json.dumps(i) for i in filtered_2_d]))

            # method 3: TODO combine two ppl?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get PPL')
    parser.add_argument('-b', '--batch', required=True, type=int)
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    opt = parser.parse_args()
    main(
        m=opt.model,
        batch=opt.batch,
        overwrite=opt.overwrite
    )

