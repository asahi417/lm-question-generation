""" Training model. """
import os
import json
import logging
import shutil
import random
from os.path import join as pj
from glob import glob
from typing import List

import torch

from .language_model import TransformersQG
from .data import get_dataset, DEFAULT_CACHE_DIR

__all__ = ('to_list', 'Trainer')

OPTIMIZER_ON_CPU = bool(int(os.getenv('OPTIMIZER_ON_CPU', '0')))


def to_list(_val, sorting=True):
    if type(_val) != list:
        return [_val]
    if sorting:
        return sorted(_val, reverse=True)
    return _val


class Config:
    """ Model checkpoint managing class. """

    def __init__(self, checkpoint_dir: str, config_file: str = 'trainer_config.json', **kwargs):
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir):
            logging.info(f'load config from existing checkpoint at {self.checkpoint_dir}')
            self.config = self.safe_open(pj(self.checkpoint_dir, config_file))
        else:
            logging.info(f'initialize checkpoint at {self.checkpoint_dir}')
            self.config = kwargs
            configs = {i: self.safe_open(i) for i in glob(pj(os.path.dirname(self.checkpoint_dir), '*', config_file))}
            configs = list(filter(lambda x: x[1] == self.config, configs.items()))
            if len(configs) != 0:
                input(f'\ncheckpoint with same config already exists: {configs[0]}\n enter to overwrite >>>')
                for _p, _ in configs:
                    shutil.rmtree(os.path.dirname(_p))
            self.__initialize_checkpoint(config_file)

        self.__dict__.update(self.config)
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info(f'\t * {k}: {str(v)[:min(100, len(str(v)))]}')

    def __initialize_checkpoint(self, config_file):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if not os.path.exists(pj(self.checkpoint_dir, config_file)):
            with open(pj(self.checkpoint_dir, config_file), 'w') as f:
                json.dump(self.config, f)

    @staticmethod
    def safe_open(_file):
        with open(_file, 'r') as f:
            return json.load(f)


class Trainer:

    def __init__(self,
                 checkpoint_dir: str,
                 dataset_path: str = "asahi417/qg_squad",
                 dataset_name: str = 'default',
                 input_types: List or str = 'paragraph_answer',
                 output_types: List or str = 'question',
                 prefix_types: List or str = 'qg',
                 model: str = 't5-small',
                 max_length: int = 512,
                 max_length_output: int = 32,
                 epoch: int = 10,
                 batch: int = 128,
                 lr: float = 1e-4,
                 fp16: bool = False,
                 random_seed: int = 42,
                 gradient_accumulation_steps: int = 4,
                 label_smoothing: float = None,
                 disable_log: bool = False,
                 config_file: str = 'trainer_config.json'):
        logging.info('initialize model trainer')

        # config
        self.config = Config(
            config_file=config_file, checkpoint_dir=checkpoint_dir, dataset_path=dataset_path, dataset_name=dataset_name,
            input_types=input_types, output_types=output_types, prefix_types=prefix_types, model=model,
            max_length=max_length, max_length_output=max_length_output, epoch=epoch, batch=batch, lr=lr, fp16=fp16,
            random_seed=random_seed, gradient_accumulation_steps=gradient_accumulation_steps,
            label_smoothing=label_smoothing)

        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if not disable_log:
            # add file handler
            logger = logging.getLogger()
            file_handler = logging.FileHandler(pj(self.config.checkpoint_dir, 'training.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
            logger.addHandler(file_handler)

        # load model
        add_prefix = False if self.config.prefix_types is None else True
        ckpts = glob(pj(self.config.checkpoint_dir, 'epoch_*'))
        ckpts = [i for i in ckpts if os.path.exists(
            pj(self.config.checkpoint_dir, 'optimizers', f"optimizer.{i.split('epoch_')[-1]}.pt"))]
        flag = False
        if len(ckpts):
            epochs = sorted([int(i.split('epoch_')[-1]) for i in ckpts], reverse=True)
            for epoch in epochs:
                try:
                    path = pj(self.config.checkpoint_dir, f"epoch_{epoch}")
                    logging.info(f'load checkpoint from {path}')
                    self.model = TransformersQG(
                        model=path, max_length=self.config.max_length, max_length_output=self.config.max_length_output,
                        label_smoothing=self.config.label_smoothing, add_prefix=add_prefix,
                        drop_overflow_text=True
                    )
                    self.optimizer = self.setup_optimizer(epoch)
                    self.current_epoch = epoch
                    assert self.current_epoch <= self.config.epoch, 'model training is done'
                    flag = True
                except Exception:
                    logging.exception(f'error at loading checkpoint {ckpts}')
                if flag:
                    break
        if not flag:
            logging.info(f'initialize checkpoint with {self.config.model}')
            self.model = TransformersQG(
                model=self.config.model, max_length=self.config.max_length,
                max_length_output=self.config.max_length_output, add_prefix=add_prefix,
                drop_overflow_text=True)
            self.optimizer = self.setup_optimizer()
            self.current_epoch = 0

        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)

        # cached data folder
        input_types = to_list(self.config.input_types, sorting=False)
        output_types = to_list(self.config.output_types, sorting=False)
        assert len(input_types) == len(output_types)
        if prefix_types is None:
            prefix_types = [None] * len(input_types)
        else:
            prefix_types = to_list(self.config.prefix_types, sorting=False)
        prefix = pj(
            DEFAULT_CACHE_DIR,
            "encoded_feature"
            f"{self.config.dataset_path}{'.' + self.config.dataset_name if self.config.dataset_name != 'default' else ''}",
            f"{self.config.model}.{self.config.max_length}.{self.config.max_length_output}"
        )

        self.data_cache_paths = [[(i, o, p), f'{prefix}.{i}.{o}.train.{p}.pkl']
                                 for i, o, p in zip(input_types, output_types, prefix_types)]

    def setup_optimizer(self, epoch: int = None):
        # optimizer
        optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.config.lr)
        if epoch is not None:
            # load from existing config
            path = pj(self.config.checkpoint_dir, "optimizers", f'optimizer.{epoch}.pt')
            logging.info(f'load optimizer from {path}')
            device = 'cpu' if OPTIMIZER_ON_CPU else self.model.device
            logging.info(f'optimizer is loading on {device}')
            optimizer_stat = torch.load(path, map_location=torch.device(device))
            optimizer.load_state_dict(optimizer_stat['optimizer_state_dict'])
            del optimizer_stat
        return optimizer

    def save(self, current_epoch):
        # save model
        save_dir = pj(self.config.checkpoint_dir, f'epoch_{current_epoch + 1}')
        os.makedirs(save_dir, exist_ok=True)
        logging.info('saving model related files')
        self.model.save(save_dir)
        with open(pj(save_dir, 'trainer_config.json'), 'w') as f:
            tmp = self.config.config.copy()
            tmp['epoch'] = current_epoch + 1
            json.dump(obj=tmp, fp=f)

        # save optimizer
        save_dir_opt = pj(self.config.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch + 1}.pt')
        os.makedirs(os.path.dirname(save_dir_opt), exist_ok=True)
        logging.info('saving optimizer')
        torch.save({'optimizer_state_dict': self.optimizer.state_dict()}, save_dir_opt)

        logging.info('remove old optimizer files')
        path = pj(self.config.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch}.pt')
        if os.path.exists(path):
            os.remove(path)

    def train(self, epoch_save: None or int = 1, interval: int = 25, epoch_partial: int = None,
              use_auth_token: bool = False):
        """ Train model.

        @param epoch_save: Save the model every this epoch.
        @param interval:
        @param epoch_partial:
        """
        self.model.train()

        if self.current_epoch == self.config.epoch:
            logging.info('training is completed')
            return None

        logging.info('dataset preprocessing')
        encode_list = []
        for (i, o, p), cache_path in self.data_cache_paths:
            text_input, text_output = get_dataset(
                self.config.dataset_path, self.config.dataset_name, split='train', input_type=i, output_type=o,
                use_auth_token=use_auth_token)
            encode_list += self.model.text_to_encode(text_input, text_output, prefix_type=p, cache_path=cache_path)
        loader = self.model.get_data_loader(encode_list, batch_size=self.config.batch, shuffle=True, drop_last=True)

        logging.info('start model training')
        global_step = 0
        saved_checkpoints = []
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.current_epoch, self.config.epoch):  # loop over the epoch
                mean_loss, global_step = self.train_single_epoch(loader, global_step, interval)
                logging.info(f"[epoch {e}/{self.config.epoch}] average loss: {round(mean_loss, 3)}, "
                             f"lr: {self.optimizer.param_groups[0]['lr']}")
                if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != 0:
                    self.save(e)
                    saved_checkpoints.append(e)
                if epoch_partial is not None and (e + 1) == epoch_partial:
                    break
        if e not in saved_checkpoints:
            self.save(e)
        logging.info(f'complete training: model ckpt was saved at {self.config.checkpoint_dir}')

    def train_single_epoch(self, data_loader, global_step: int, interval):
        total_loss = []
        self.optimizer.zero_grad()
        for n, encode in enumerate(data_loader):

            loss = self.model.encode_to_loss(encode)
            self.scaler.scale(loss).backward()
            total_loss.append(loss.cpu().item())
            if (n + 1) % self.config.gradient_accumulation_steps != 0:
                continue

            global_step += 1
            _total_loss = total_loss[-self.config.gradient_accumulation_steps:]
            inst_loss = sum(_total_loss)/len(_total_loss)

            # optimizer update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if global_step % interval == 0:
                logging.info(f"\t * (global step {global_step}: loss: {inst_loss}, "
                             f"lr: {self.optimizer.param_groups[0]['lr']}")
        self.optimizer.zero_grad()
        return sum(total_loss)/len(total_loss), global_step
