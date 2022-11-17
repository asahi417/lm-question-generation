""" PPl for answer & question """
import json
from lmqg import TransformersQG

models = ["t5-large-squad-multitask", "t5-base-squad-multitask", "t5-small-squad-multitask"]
domains = ['amazon', 'new_wiki', 'nyt', 'reddit']
batch = 512
for m in models:
    model = TransformersQG(f"lmqg/{m}")
    for d in domains:
        dataset_name = f"{d}.{m}"
        for split in ['train', 'validation']:

            with open(f"qa_squadshifts_pseudo/{m}.{d}/{split}.jsonl") as f:
                dataset_split = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]

            # print(f"Computing perplexity for answer: `{m}`, domain: `{d}`, split: `{split}`")
            # ppl_answer = model.get_perplexity(
            #     list_question=[i['question'] for i in dataset_split],
            #     list_context=[i['context'] for i in dataset_split],
            #     list_answer=[i['answers']['text'][0] for i in dataset_split],
            #     target_output='answer',
            #     batch_size=batch
            # )
            # with open(f"qa_squadshifts_pseudo/{m}.{d}/perplexity_answer.{split}.json", "w") as f:
            #     json.dump({"perplexity_answer": ppl_answer}, f)

            print(f"Computing perplexity for question: `{m}`, domain: `{d}`, split: `{split}`")
            ppl_question = model.get_perplexity(
                list_question=[i['question'] for i in dataset_split],
                list_context=[i['context'] for i in dataset_split],
                list_answer=[i['answers']['text'][0] for i in dataset_split],
                target_output='question',
                batch_size=batch
            )
            with open(f"qa_squadshifts_pseudo/{m}.{d}/perplexity_question.{split}.json", "w") as f:
                json.dump({"perplexity_question": ppl_question}, f)

