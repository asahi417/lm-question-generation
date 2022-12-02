""" Data utility """
import logging
import os
import requests
from os.path import join as pj
from glob import glob

from datasets import load_dataset
from .language_model import internet_connection


__all__ = ('get_dataset', 'get_reference_files', 'DEFAULT_CACHE_DIR')
DEFAULT_CACHE_DIR = pj(os.path.expanduser('~'), '.cache', 'lmqg')


def get_dataset(path: str = 'lmqg/qg_squad',
                name: str = 'default',
                split: str = 'train',
                input_type: str = 'paragraph_answer',
                output_type: str = 'question',
                use_auth_token: bool = False):
    """ Get question generation input/output list of texts. """
    name = None if name == 'default' else name
    dataset = load_dataset(path, name, split=split, use_auth_token=use_auth_token)
    return dataset[input_type], dataset[output_type]


def get_reference_files(path: str = 'lmqg/qg_squad', name: str = 'default', cache_dir: str = None):
    """ Get reference files for automatic evaluation """
    url = f'https://huggingface.co/datasets/{path}/raw/main/reference_files'
    name = None if name == 'default' else name
    local_files_only = not internet_connection()
    if cache_dir is None:
        cache_dir = pj(DEFAULT_CACHE_DIR, 'reference_files', path)
    output = {}
    for split in ['test', 'validation']:
        dataset = load_dataset(path, name, split=split)
        for feature in ['answer', 'question', 'paragraph', 'sentence', 'questions_answers']:
            if feature not in dataset.features:
                continue
            if name is None:
                filename = f'{feature}-{split}.txt'
            else:
                filename = f'{feature}-{split}.{name}.txt'
            ref_path = pj(cache_dir, filename)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    line_length = len(f.read().split('\n'))
                if line_length < 20:
                    os.remove(ref_path)
            if not os.path.exists(ref_path):
                if local_files_only:
                    logging.info(f'network is not reachable, could not download the file from {url}/{filename}')
                    continue
                os.makedirs(os.path.dirname(ref_path), exist_ok=True)
                try:
                    r = requests.get(f'{url}/{filename}')
                    content = r.content
                    assert "Entry not found" not in str(content) and content != b'', content
                    with open(ref_path, "wb") as f:
                        f.write(content)
                except Exception:
                    logging.warning(f'can not get file from {url}/{filename}. generate reference file instead')
                    generate_reference_files(dataset=dataset, ref_path=ref_path, feature=feature)

            with open(ref_path) as f:
                line_length = len(f.read().split('\n'))
            if line_length < 20:
                os.remove(ref_path)
                continue
            assert os.path.exists(ref_path)
            output[f'{feature}-{split}'] = ref_path
    return output


def generate_reference_files(dataset, ref_path: str, feature: str):
    with open(ref_path, 'w') as f:
        if feature == 'paragraph':
            tmp_data = dataset['paragraph_id']
        else:
            tmp_data = dataset[feature]
        f.write('\n'.join([i.replace('\n', '.') for i in tmp_data]))
