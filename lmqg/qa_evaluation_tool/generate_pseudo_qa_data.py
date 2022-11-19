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
    data = load_dataset(anchor_data) if anchor_data_name is None else load_dataset(anchor_data, anchor_data_name)

    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)


    model = TransformersQG(model=qg_model, language=language, skip_overflow_error=True, drop_answer_error_text=True)
    if answer_extraction is None:
        answer_extraction = True if model.multitask_model else False

    logging.info(f"Export dir: {export_dir}")
    full_output = {}
    for _split in data:

        logging.info(f'running prediction on {_split}')
        if export_dir is not None:
            if not overwrite and os.path.exists(pj(export_dir, f'{_split}.jsonl')):
                with open(pj(export_dir, f'{_split}.jsonl')) as f:
                    full_output[_split] = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                continue

        if answer_extraction:
            if answer_model is None and answer_extraction:
                answer_model = 'language_model' if model.multitask_model else 'keyword_extraction'
            output = []
            for c, df in tqdm(data[_split].to_pandas().groupby('context')):
                out = model.generate_qa(
                    df['context'].values[0],
                    answer_model=answer_model,
                    batch_size=batch_size
                )
                if out is None or len(out) == 0:
                    continue
                output += [{
                    'id': df['id'].values[0],
                    'title': df['title'].values[0],
                    'context': df['context'].values[0],
                    'question': q,
                    'answers': {'text': [a], 'answer_start': [df['context'].values[0].index(a)]}
                } for q, a in out]

        else:
            question = model.generate_q(
                list_context=data[_split]['context'],
                list_answer=[a['text'][0] for a in data[_split]['answers']],
                batch_size=batch_size
            )
            if question is None or len(question) == 0:
                continue
            assert len(data[_split]) == len(question), f"{len(data[_split])} != {len(question)}"
            output = [{
                'id': tmp_data['id'],
                'title': tmp_data['title'],
                'context': tmp_data['context'],
                'question': q,
                'answers': tmp_data['answers']
            } for tmp_data, q in zip(data[_split], question)]

        full_output[_split] = output
        if export_dir is not None:
            logging.info(f"saving {_split} at {pj(export_dir, f'{_split}.jsonl')}")
            with open(pj(export_dir, f'{_split}.jsonl'), 'w') as f:
                f.write('\n'.join([json.dumps(i) for i in output]))

    return full_output
