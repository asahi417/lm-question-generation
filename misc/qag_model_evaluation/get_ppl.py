""" PPl for answer & question
python misc/qag_model_evaluation/get_ppl.py -m "t5-small-squad-multitask" -t "answer" -b 128
python misc/qag_model_evaluation/get_ppl.py -m "t5-small-squad-multitask" -t "question" -b 128
python misc/qag_model_evaluation/get_ppl.py -m "t5-base-squad-multitask" -t "answer" -b 32
python misc/qag_model_evaluation/get_ppl.py -m "t5-base-squad-multitask" -t "question" -b 32
python misc/qag_model_evaluation/get_ppl.py -m "t5-large-squad-multitask" -t "answer" -b 16
python misc/qag_model_evaluation/get_ppl.py -m "t5-large-squad-multitask" -t "question" -b 16
"""
import argparse
import os
import json
import logging
from lmqg import TransformersQG
from datasets import load_dataset


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
domains = ['amazon', 'new_wiki', 'nyt', 'reddit']


def main(m, batch, target: str = "answer", overwrite: bool = False):

    for d in domains:
        for split in ['train', 'validation']:
            output_file = f"qa_squadshifts_pseudo/{m}.{d}/perplexity_{target}.{split}.json"
            with open(f"qa_squadshifts_pseudo/{m}.{d}/{split}.jsonl") as f:
                dataset_split = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]

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

            # reference data
            # dataset_gold = load_dataset("lmqg/qa_squadshifts", d, split=split)

            # method 1: the same amount of data by ppl

            # for _ppl, _d in zip(ppl, dataset_split):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get PPL')
    parser.add_argument('-b', '--batch', required=True, type=int)
    parser.add_argument('-t', '--target', required=True, type=str)
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    opt = parser.parse_args()
    main(
        m=opt.model,
        batch=opt.batch,
        target=opt.target,
        overwrite=opt.overwrite
    )

