import logging
import os
import json
from os.path import join as pj
from tqdm import tqdm
from datasets import load_dataset
from ..language_model import TransformersQG


def generate_qa_pairs(
        model_qg: str = 'lmqg/t5-small-squad-qg-ae',
        model_ae: str = None,
        language: str = 'en',
        anchor_data: str = 'lmqg/qa_squadshifts',
        anchor_data_name: str = 'new_wiki',
        use_reference_answer: bool = False,
        batch_size: int = 256,
        export_dir: str = None,
        overwrite: bool = False,
        max_length: int = 512,
        max_length_output: int = 256,
        use_auth_token: bool = False,
        torch_dtype=None,
        device_map: str = None,
        low_cpu_mem_usage: bool = False):

    logging.info(f'generate QA pairs from {anchor_data} with {model_qg}')
    data = load_dataset(anchor_data) if anchor_data_name is None else load_dataset(anchor_data, anchor_data_name)

    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)

    model = TransformersQG(
        model=model_qg,
        model_ae=model_ae,
        language=language,
        skip_overflow_error=True,
        drop_answer_error_text=True,
        max_length=max_length,
        max_length_output=max_length_output,
        use_auth_token=use_auth_token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage)

    logging.info(f"Export dir: {export_dir}")
    full_output = {}
    for _split in data:

        logging.info(f'running prediction on {_split}')
        if export_dir is not None:
            if not overwrite and os.path.exists(pj(export_dir, f'{_split}.jsonl')):
                with open(pj(export_dir, f'{_split}.jsonl')) as f:
                    full_output[_split] = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                continue

        if use_reference_answer:
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

        else:
            list_context = []
            list_data = []
            for c, df in tqdm(data[_split].to_pandas().groupby('context')):
                list_context.append(c)
                list_data.append(df)
            qa_pairs = model.generate_qa(
                list_context=list_context,
                batch_size=batch_size
            )
            output = []
            _id = 0
            for _qa_pairs, _context, df in zip(qa_pairs, list_context, list_data):
                if _qa_pairs is None:
                    continue
                for q, a in _qa_pairs:
                    context = df['context'].values[0]
                    if a not in context:
                        continue
                    _id += 1
                    output.append({
                        'id': str(_id),
                        'title': df['title'].values[0],
                        'context': context,
                        'question': q,
                        'answers': {'text': [a], 'answer_start': [context.index(a)]}
                    })

        full_output[_split] = output
        if export_dir is not None:
            logging.info(f"saving {_split} at {pj(export_dir, f'{_split}.jsonl')}")
            with open(pj(export_dir, f'{_split}.jsonl'), 'w') as f:
                f.write('\n'.join([json.dumps(i) for i in output]))

    return full_output
