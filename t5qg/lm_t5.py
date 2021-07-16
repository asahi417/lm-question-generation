""" T5 model. """
import os
import logging
import pickle
from typing import List, Dict
from multiprocessing import Pool

import torch
import transformers
from .exceptions import ExceedMaxLengthError, HighlightNotFoundError

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
TASK_PREFIX = {
    "ans_ext": "extract answers",
    "e2e_qg": "generate questions",
    "qa": "question",
    "qg": "generate question"
}
ADDITIONAL_SP_TOKENS = {'sep': '<sep>', 'hl': '<hl>'}
__all__ = ('T5', 'ADDITIONAL_SP_TOKENS', 'TASK_PREFIX')


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def load_language_model(model_name, cache_dir: str = None):
    """ load language model from huggingface model hub """
    # tokenizer
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except ValueError:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    try:
        config = transformers.AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    except ValueError:
        config = transformers.AutoConfig.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)

    # model
    if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
        model_class = transformers.T5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'mt5':
        model_class = transformers.MT5ForConditionalGeneration.from_pretrained
    else:
        raise ValueError('unsupported model type: {}'.format(config.model_type))
    try:
        model = model_class(model_name, config=config, cache_dir=cache_dir)
    except ValueError:
        model = model_class(model_name, config=config, cache_dir=cache_dir, local_files_only=True)
    # add new special tokens to the tokenizer and the model if they don't have it
    tokenizer.add_special_tokens({'additional_special_tokens': list(ADDITIONAL_SP_TOKENS.values())})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model, config


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset wrapper converting into tensor """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 max_length_output: int = 34,
                 drop_overflow_text: bool = False,
                 skip_overflow_error: bool = False,
                 task_prefix: str = None,
                 padding: bool = True):
        """ Wrapper of encode_plus for multiprocessing.

        @param tokenizer: transforms.Tokenizer
        @param max_length: Input max length.
        @param max_length_output: Output max length.
        @param drop_overflow_text: Return None if the input sentence exceeds the max token length.
        @param skip_overflow_error: Raise error if the input sentence exceeds the max token length.
        @param task_prefix: Either of `qg`, `ans_ext`, `qa`.
        @param padding: Pad the sequence to the max length.
        """
        assert task_prefix is None or task_prefix in TASK_PREFIX
        self.task_prefix = task_prefix
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_output = max_length_output
        # for model training, we should drop the exceeded input but not for the evaluation
        self.drop_overflow_text = drop_overflow_text
        self.skip_overflow_error = skip_overflow_error

        # truncation should be true for the batch process, but not necessary to process single input
        self.param_in = {'truncation': True, 'max_length': self.max_length}
        self.param_out = {'truncation': True, 'max_length': self.max_length_output}
        self.padding = padding
        if self.padding:
            self.param_in['padding'] = 'max_length'
            self.param_out['padding'] = 'max_length'

    def __call__(self, inputs):
        return self.encode_plus(*inputs)

    def encode_plus(self, input_sequence: str, output_sequence: str = None, input_highlight: str = None):

        # add highlight to the input
        if input_highlight is not None:
            position = input_sequence.find(input_highlight)
            if position == -1:
                raise HighlightNotFoundError(input_highlight, input_sequence)
            input_sequence = '{0}{1} {2} {1}{3}'.format(
                input_sequence[:position], ADDITIONAL_SP_TOKENS['hl'], input_highlight,
                input_sequence[position+len(input_highlight):])

        if self.task_prefix is not None:
            input_sequence = '{}: {}'.format(TASK_PREFIX[self.task_prefix], input_sequence)

        # remove sentence that exceeds the max_length
        if self.drop_overflow_text or not self.skip_overflow_error:
            if len(self.tokenizer.encode(input_sequence)) > self.max_length:
                if self.drop_overflow_text:
                    return None
                raise ExceedMaxLengthError(self.max_length)
            if output_sequence is not None:
                if len(self.tokenizer.encode(output_sequence)) > self.max_length_output:
                    if self.drop_overflow_text:
                        return None
                    raise ExceedMaxLengthError(self.max_length)
        encode = self.tokenizer.encode_plus(input_sequence, **self.param_in)
        if output_sequence is not None:
            encode['labels'] = self.tokenizer.encode(output_sequence, **self.param_out)
        return encode


class T5:
    """ T5 model. """

    def __init__(self, model: str, max_length: int = 512, max_length_output: int = 32, cache_dir: str = None):
        """ T5 model.

        @param model: path to the checkpoint or alias on huggingface modelhub.
        @param max_length: Max sequence length for the input.
        @param max_length_output: Max sequence length for the output.
        @param cache_dir:
        """
        self.model_name = model
        self.max_length = max_length
        self.max_length_output = max_length_output
        logging.info('instantiate T5 model class with `{}`'.format(self.model_name))
        self.tokenizer, self.model, _ = load_language_model(self.model_name, cache_dir=cache_dir)

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

    def get_prediction(self,
                       list_input: List,
                       list_highlight: List = None,
                       drop_overflow_text: bool = False,
                       skip_overflow_error: bool = False,
                       task_prefix: str = None,
                       batch_size: int = None,
                       num_beams: int = 4,
                       num_workers: int = 0,
                       cache_path: str = None):
        """ Get model prediction

        @param list_input: List of input sentences.
        @param list_highlight: List of highlight phrases.
        @param task_prefix: Either of `qg`, `ans_ext`, `qa`.
        @param drop_overflow_text: Return None if the input sentence exceeds the max token length.
        @param skip_overflow_error: Raise error if the input sentence exceeds the max token length.
        @param batch_size: Batch size.
        @param num_beams: Number of beam for model generation.
        @param num_workers:
        @param cache_path:
        @return: List of generated sentences.
        """
        assert type(list_input) == list, list_input
        self.eval()
        loader = self.get_data_loader(list_input,
                                      highlights=list_highlight,
                                      task_prefix=task_prefix,
                                      drop_overflow_text=drop_overflow_text,
                                      skip_overflow_error=skip_overflow_error,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      cache_path=cache_path)
        outputs = []

        for encode in loader:
            with torch.no_grad():
                encode = {k: v.to(self.device) for k, v in encode.items()}
                encode['max_length'] = self.max_length_output
                encode['num_beams'] = num_beams
                tensor = self.model.module.generate(**encode) if self.parallel else self.model.generate(**encode)
                outputs += self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return outputs

    def encode_to_loss(self, encode: Dict):
        loss = self.model(**{k: v.to(self.device) for k, v in encode.items()})['loss']
        return loss.mean() if self.parallel else loss

    def get_data_loader(self,
                        inputs,
                        outputs: List = None,
                        highlights: List = None,
                        task_prefix: str = None,
                        batch_size: int = None,
                        num_workers: int = 0,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        cache_path: str = None,
                        drop_overflow_text: bool = False,
                        skip_overflow_error: bool = False):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader.

        @param inputs: List of input sentences.
        @param outputs: List of output sentences.
        @param highlights: List of highlight phrases.
        @param task_prefix: Either of `qg`, `ans_ext`, `qa`.
        @param batch_size: Batch size.
        @param num_workers:
        @param shuffle:
        @param drop_last:
        @param cache_path:
        @param drop_overflow_text: Return None if the input sentence exceeds the max token length.
        @param skip_overflow_error: Raise error if the input sentence exceeds the max token length.
        @return: torch.utils.data.DataLoader
        """
        if outputs is not None:
            assert len(outputs) == len(inputs), '{} != {}'.format(len(outputs), len(inputs))
            data = list(zip(inputs, outputs))
        else:
            data = [(i, None) for i in inputs]

        if highlights is not None:
            assert len(highlights) == len(inputs), '{} != {}'.format(len(highlights), len(inputs))
            data = [tuple(list(d) + [h]) for d, h in zip(data, highlights)]

        if cache_path is not None and os.path.exists(cache_path):
            logging.info('loading preprocessed feature from {}'.format(cache_path))
            out = pickle_load(cache_path)
        else:
            # process in parallel
            pool = Pool()
            config = {'tokenizer': self.tokenizer, 'max_length': self.max_length,
                      'max_length_output': self.max_length_output, 'drop_overflow_text': drop_overflow_text,
                      'task_prefix': task_prefix, 'skip_overflow_error': skip_overflow_error}
            if len(data) == 1:
                config['padding'] = False

            out = pool.map(EncodePlus(**config), data)
            pool.close()

            # remove overflow text
            logging.info('encode all the data       : {}'.format(len(out)))
            out = list(filter(None, out))
            logging.info('after remove the overflow : {}'.format(len(out)))

            # cache the encoded data
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                pickle_save(out, cache_path)
                logging.info('preprocessed feature is saved at {}'.format(cache_path))

        batch_size = len(out) if batch_size is None else batch_size
        return torch.utils.data.DataLoader(
            Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    def save(self, save_dir):
        if self.parallel:
            self.model.module.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

