import logging
import os
import json
from os.path import join as pj
from tqdm import tqdm
from datasets import load_dataset
from ..language_model import TransformersQG


def generate_qa_pairs(
        qg_model: str = 'lmqg/t5-small-squad-multitask',
        language: str = 'en',
        anchor_data: str = 'lmqg/qa_squadshifts',
        anchor_data_name: str = 'new_wiki',
        answer_extraction: bool = None,
        answer_model: str = None,  # language_model or keyword_extraction
        batch_size: int = 256,
        export_dir: str = None,
        overwrite: bool = False):
    logging.info(f'generate QA pairs from {anchor_data} with {qg_model}')
    if anchor_data_name is None:
        data = load_dataset(anchor_data)
    else:
        data = load_dataset(anchor_data, anchor_data_name)

    if answer_extraction is None:
        answer_extraction = True if qg_model.endswith('multitask') else False

    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)

    full_output = {}
    if answer_extraction:
        model = TransformersQG(model=qg_model,
                               language=language,
                               drop_answer_error_text=True,
                               drop_overflow_error_text=True,
                               drop_highlight_error_text=True)
        if answer_model is None and answer_extraction:
            answer_model = 'language_model' if model.add_prefix else 'keyword_extraction'
        for _split in data:
            logging.info(f'running prediction on {_split}')
            if export_dir is not None:
                if not overwrite and os.path.exists(pj(export_dir, f'{_split}.jsonl')):
                    with open(pj(export_dir, f'{_split}.jsonl')) as f:
                        full_output[_split] = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                    continue
            output = []
            for tmp_data in tqdm(data[_split]):
                out = model.generate_qa(
                    context=tmp_data['context'],
                    answer_model=answer_model,
                    batch_size=batch_size
                )
                if out is None or len(out) == 0:
                    continue
                for q, a in out:
                    output.append(
                        {
                            'id': tmp_data['id'],
                            'title': tmp_data['title'],
                            'context': tmp_data['context'],
                            'question': q,
                            'answers': {
                                'text': [a],
                                'answer_start': [tmp_data['context'].index(a)]
                            }
                        }
                    )
            full_output[_split] = output
    else:
        model = TransformersQG(model=qg_model, language=language, skip_overflow_error=True)
        for _split in data:
            logging.info(f'running prediction on {_split}')

            if export_dir is not None:
                if not overwrite and os.path.exists(pj(export_dir, f'{_split}.jsonl')):
                    with open(pj(export_dir, f'{_split}.jsonl')) as f:
                        full_output[_split] = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                    continue

            raw_context = data[_split]['context']
            raw_answer = data[_split]['answers']
            list_context = []
            list_answer = []
            for c, a in zip(raw_context, raw_answer):
                for _a in a['text']:
                    list_answer.append(_a)
                    list_context.append(c)
            question = model.generate_q(
                list_context=list_context,
                list_answer=list_answer,
                batch_size=batch_size
            )
            if question is None or len(question) == 0:
                continue
            output = []
            for tmp_data, q in zip(data[_split], question):
                output.append(
                    {
                        'id': tmp_data['id'],
                        'title': tmp_data['title'],
                        'context': tmp_data['context'],
                        'question': q,
                        'answers': tmp_data['answers']
                    }
                )
            full_output[_split] = output
    if export_dir is not None:
        for k, v in full_output.items():
            if overwrite or not os.path.exists(pj(export_dir, f'{k}.jsonl')):
                with open(pj(export_dir, f'{k}.jsonl'), 'w') as f:
                    f.write('\n'.join([json.dumps(i) for i in v]))
    return full_output
