""" PPl for answer & question """
import argparse
import os
import json
from lmqg import TransformersQG


domains = ['amazon', 'new_wiki', 'nyt', 'reddit']


def main(m, batch, target: str = "answer"):
    model = TransformersQG(f"lmqg/{m}")
    for d in domains:
        for split in ['train', 'validation']:
            output_file = f"qa_squadshifts_pseudo/{m}.{d}/perplexity_{target}.{split}.json"
            if os.path.exists(output_file):
                continue

            with open(f"qa_squadshifts_pseudo/{m}.{d}/{split}.jsonl") as f:
                dataset_split = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]

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


if __name__ == '__main__':
    # models = ["t5-large-squad-multitask", "t5-base-squad-multitask", "t5-small-squad-multitask"]

    parser = argparse.ArgumentParser(description='Get PPL')
    parser.add_argument('-b', '--batch', required=True, type=int)
    parser.add_argument('-t', '--target', required=True, type=str)
    parser.add_argument('-m', '--model', required=True, type=str)
    opt = parser.parse_args()
    main(
        m=opt.model,
        batch=opt.batch,
        target=opt.target
    )

