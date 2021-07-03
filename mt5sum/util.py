import random
import os
import tarfile
import zipfile
import gzip
import json
import requests
import logging
import shutil
import subprocess

from typing import List
from glob import glob
import gdown
import numpy as np
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset

# default_cache_dir = '{}/.cache/mt5sum'.format(os.path.expanduser('~'))
# os.makedirs(default_cache_dir, exist_ok=True)


def get_dataset(name, argument, split):
    dataset = load_dataset(name, argument, split=split)
    if name == 'cnn_dailymail':
        inputs = dataset['article']
        outputs = dataset['highlights']
    elif name == 'mlsum':
        inputs = dataset['text']
        outputs = dataset['summary']
    else:
        raise ValueError('undefined data: {}'.format(name))
    return inputs, outputs


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
        model_class = transformers.AutoModelForSequenceClassification.from_pretrained
    try:
        model = model_class(model_name, config=config, cache_dir=cache_dir)
    except ValueError:
        model = model_class(model_name, config=config, cache_dir=cache_dir, local_files_only=True)
    return tokenizer, model, config


def wget(url, cache_dir: str = './cache', gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
        filename = gdrive_filename
    else:
        filename = os.path.basename(url)
        with open('{}/{}'.format(cache_dir, filename), "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    path = '{}/{}'.format(cache_dir, filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=None, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        current_step += 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if num_training_steps is None:
            return 1
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


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


class Config:
    """ Model checkpoint managing class. """

    def __init__(self, checkpoint_dir: str, **kwargs):
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir):
            logging.info('load config from existing checkpoint at {}'.format(self.checkpoint_dir))
            self.config = self.safe_open('{}/trainer_config.json'.format(self.checkpoint_dir))
        else:
            logging.info('initialize checkpoint at {}'.format(self.checkpoint_dir))
            self.config = kwargs
            configs = {i: self.safe_open(i) for i in glob(
                '{}/*/trainer_config.json'.format(os.path.dirname(self.checkpoint_dir)))}
            configs = list(filter(lambda x: x[1] == self.config, configs.items()))
            if len(configs) != 0:
                input('\ncheckpoint with same config already exists: {}\n enter to overwrite >>>'.format(configs[0]))
                for _p, _ in configs:
                    shutil.rmtree(os.path.dirname(_p))
            self.__initialize_checkpoint()

        self.__dict__.update(self.config)
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info('\t * {}: {}'.format(k, str(v)[:min(100, len(str(v)))]))

    def __initialize_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if not os.path.exists('{}/trainer_config.json'.format(self.checkpoint_dir)):
            with open('{}/trainer_config.json'.format(self.checkpoint_dir), 'w') as f:
                json.dump(self.config, f)

    @staticmethod
    def safe_open(_file):
        with open(_file, 'r') as f:
            return json.load(f)


def exe_shell(command: str, exported_file: str = None):
    """ Execute shell command """
    logging.info("execute `{}`".format(command))
    try:
        args = dict(stderr=subprocess.STDOUT, shell=True, timeout=600, universal_newlines=True)
        log = subprocess.check_output(command, **args)
        logging.info("log\n{}".format(log))
    except subprocess.CalledProcessError as exc:
        if exported_file and os.path.exists(exported_file):
            # clear possibly broken file out
            os.system('rm -rf {}'.format(exported_file))
        raise ValueError("fail to execute command `{}`:\n {}\n {}".format(command, exc.returncode, exc.output))
