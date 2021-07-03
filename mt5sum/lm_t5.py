import os
import logging
import pickle
from typing import List, Dict
from multiprocessing import Pool

import torch
from .util import Dataset, load_language_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self, tokenizer, max_length: int = 512, max_length_output: int = 128, task_prefix: str = 'summarize:'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_output = max_length_output
        self.task_prefix = task_prefix

    def __call__(self, inputs):
        """ encode_plus wrapper for multiprocessing """
        return self.encode_plus(*inputs)

    def encode_plus(self, input_sequence: str, output_sequence: str = None):
        param_input = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        param_output = {'max_length': self.max_length_output, 'truncation': True, 'padding': 'max_length'}
        encode = self.tokenizer.encode_plus(' '.join([self.task_prefix, input_sequence]), **param_input)
        if output_sequence is not None:
            encode['labels'] = self.tokenizer.encode(output_sequence, **param_output)
        return encode


class T5Summarizer:
    """  T5 summarization model. """

    def __init__(self,
                 model: str,
                 max_length: int = 128,
                 max_length_output: int = 128,
                 task_prefix: str = 'summarize:',
                 cache_dir: str = None):
        """ T5 summarization model. """
        self.model_name = model
        self.max_length = max_length
        self.max_length_output = max_length_output
        self.task_prefix = task_prefix
        logging.info('initialize T5Summarizer with `{}`'.format(self.model_name))
        self.tokenizer, self.model, _ = load_language_model(self.model_name, cache_dir=cache_dir)
        self.t5_encoder_config = {
            'tokenizer': self.tokenizer,
            'max_length': self.max_length,
            'max_length_output': self.max_length_output,
            'task_prefix': self.task_prefix
        }

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.info('{} GPUs are in use'.format(torch.cuda.device_count()))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def get_prediction(self, list_input: List, batch_size: int = None, num_workers: int = 0):
        assert type(list_input) == list, list_input
        self.eval()
        loader = self.get_data_loader(list_input, batch_size=batch_size, num_workers=num_workers)
        outputs = []
        for encode in loader:
            with torch.no_grad():
                outputs += self.generate(encode)
        return outputs

    def generate(self, encode: Dict):
        encode = {k: v.to(self.device) for k, v in encode.items()}
        if self.parallel:
            tensor = self.model.module.generate(**encode, max_length=self.max_length_output)
        else:
            tensor = self.model.generate(**encode, max_length=self.max_length_output)
        return self.tokenizer.batch_decode(tensor)

    def __call__(self, encode: Dict):
        loss = self.model(**{k: v.to(self.device) for k, v in encode.items()})['loss']
        return loss.mean() if self.parallel else loss

    def get_data_loader(self,
                        inputs,
                        outputs: List = None,
                        batch_size: int = None,
                        num_workers: int = 0,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        cache_path: str = None):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader. """
        if outputs is not None:
            assert len(outputs) == len(inputs), '{} != {}'.format(len(outputs), len(inputs))
            data = list(zip(inputs, outputs))
        else:
            data = [(i,) for i in inputs]
        features = self.__preprocess(data, cache_path)
        batch_size = len(features) if batch_size is None else batch_size
        return torch.utils.data.DataLoader(
            Dataset(features), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    def __preprocess(self, data, cache_path: str = None):
        """ Encoding list of sentence or (sentence, label) """
        assert type(data) == list, data
        if cache_path is not None and os.path.exists(cache_path):
            return pickle_load(cache_path)
        pool = Pool()
        out = pool.map(EncodePlus(**self.t5_encoder_config), data)
        pool.close()
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pickle_save(out, cache_path)
        return out

    def save(self, save_dir):
        if self.parallel:
            self.model.module.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

