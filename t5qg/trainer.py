""" Training model. """
import os
import json
import logging
import shutil
import random
from glob import glob
from typing import List

import torch
from transformers import get_linear_schedule_with_warmup

from .lm_t5 import T5
from .data import get_dataset, DEFAULT_CACHE_DIR


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


class Trainer:

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: str = "squad",
                 task_type: List or str = 'qg',
                 language: List or str = 'en',
                 model: str = 'google/mt5-small',
                 max_length: int = 512,
                 max_length_output: int = 32,
                 epoch: int = 10,
                 batch: int = 128,
                 lr: float = 1e-4,
                 lr_warmup: int = 100,
                 fp16: bool = False,
                 random_seed: int = 42,
                 gradient_accumulation_steps: int = 4):

        logging.info('initialize model trainer')
        # config
        self.config = Config(
            checkpoint_dir=checkpoint_dir,
            dataset=dataset,
            language=language,
            task_type=task_type,
            model=model,
            max_length=max_length,
            max_length_output=max_length_output,
            epoch=epoch,
            lr_warmup=lr_warmup,
            batch=batch,
            lr=lr,
            fp16=fp16,
            random_seed=random_seed,
            gradient_accumulation_steps=gradient_accumulation_steps)

        # add file handler
        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        logger = logging.getLogger()
        file_handler = logging.FileHandler('{}/training.log'.format(self.config.checkpoint_dir))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)

        # load model
        ckpts = glob('{}/epoch_*'.format(self.config.checkpoint_dir))
        if len(ckpts):
            epoch = sorted([int(i.split('epoch_')[-1]) for i in ckpts], reverse=True)[0]
            path = '{}/epoch_{}'.format(self.config.checkpoint_dir, epoch)
            logging.info('load checkpoint from {}'.format(path))
            self.model = T5(model=path,
                            max_length=self.config.max_length,
                            max_length_output=self.config.max_length_output)
            self.optimizer, self.scheduler = self.setup_optimizer(epoch)
            self.current_epoch = epoch
            assert self.current_epoch <= self.config.epoch, 'model training is done'
        else:
            logging.info('initialize checkpoint with {}'.format(self.config.model))
            self.model = T5(model=self.config.model,
                            max_length=self.config.max_length,
                            max_length_output=self.config.max_length_output)
            self.optimizer, self.scheduler = self.setup_optimizer()
            self.current_epoch = 0

        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)

        # cached data folder
        self.data_cache_dir = '{}/data_{}_encoded/{}.{}.{}.pkl'.format(
            DEFAULT_CACHE_DIR,
            self.config.dataset,
            self.config.model,
            self.config.max_length,
            self.config.max_length_output
        )
        if self.config.dataset == 'tydiqa':
            self.data_cache_dir = self.data_cache_dir.replace('.pkl', '') \
                                  + '.' + '_'.join(sorted(self.config.language)) + '.pkl'

        os.makedirs(os.path.dirname(self.data_cache_dir), exist_ok=True)

    def setup_optimizer(self, epoch: int = None):
        # optimizer
        optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.config.lr)
        # scheduler
        scheduler = None
        if self.config.lr_warmup is not None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.config.lr_warmup, num_training_steps=self.config.epoch)
        if epoch is not None:
            # load from existing config
            path = '{}/optimizers/optimizer.{}.pt'.format(self.config.checkpoint_dir, epoch)
            logging.info('load optimizer/scheduler from {}'.format(path))
            optimizer_stat = torch.load(path, map_location=torch.device('cpu'))
            optimizer.load_state_dict(optimizer_stat['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(optimizer_stat['scheduler_state_dict'])
        return optimizer, scheduler

    def save(self, current_epoch):
        # save model
        save_dir = '{}/epoch_{}'.format(self.config.checkpoint_dir, current_epoch + 1)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save(save_dir)

        # save optimizer
        save_dir_opt = '{}/optimizers/optimizer.{}.pt'.format(self.config.checkpoint_dir, current_epoch + 1)
        os.makedirs(os.path.dirname(save_dir_opt), exist_ok=True)
        if self.scheduler is None:
            torch.save({'optimizer_state_dict': self.optimizer.state_dict()}, save_dir_opt)
        else:
            torch.save({'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict()}, save_dir_opt)

    def train(self,
              num_workers: int = 0,
              epoch_save: int = 1,
              interval: int = 50,
              activate_tensorboard: bool = False):
        """ Train model.

        :param num_workers: Workers for DataLoader.
        :param epoch_save: Save the model every this epoch.
        :param interval:
        :param activate_tensorboard:
        """

        logging.info('dataset preprocessing')
        raw_input, raw_output = get_dataset(
            self.config.dataset, split='train', language=self.config.language, task_type=self.config.task_type)
        loader = self.model.get_data_loader(
            raw_input,
            raw_output,
            batch_size=self.config.batch,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            cache_path=self.data_cache_dir,
            drop_overflow_text=True
        )
        self.model.train()

        logging.info('start model training')
        global_step = 0
        writer = None
        if activate_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=self.config.checkpoint_dir)

        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.current_epoch, self.config.epoch):  # loop over the epoch
                mean_loss, global_step = self.train_single_epoch(loader, global_step, writer, interval)
                logging.info('[epoch {}/{}] average loss: {}, lr: {}'.format(
                    e, self.config.epoch, round(mean_loss, 3), self.optimizer.param_groups[0]['lr']))
                if (e + 1) % epoch_save == 0 and (e + 1) != 0:
                    self.save(e)
        if writer is not None:
            writer.close()
        self.save(e)
        logging.info('complete training: model ckpt was saved at {}'.format(self.config.checkpoint_dir))

    def train_single_epoch(self, data_loader, global_step: int, writer, interval):
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
            if writer is not None:
                writer.add_scalar('train/loss', inst_loss, global_step)
                writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            # optimizer update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            if global_step % interval == 0:
                logging.debug('\t * (global step {}: loss: {}, lr: {}'.format(
                    global_step, inst_loss, self.optimizer.param_groups[0]['lr']))

                # if n == 0:
                #     self.model.eval()
                #     sentence = self.model.tokenizer.batch_decode(encode['input_ids'], skip_special_tokens=True)
                #     label = self.model.tokenizer.batch_decode(encode['labels'], skip_special_tokens=True)
                #     out, _, _, _ = self.model.get_prediction(sentence)
                #     for _n, i in enumerate(zip(out, label, sentence)):
                #         logging.debug('\t * {}'.format(i))
                #         if _n > 5:
                #             break
                #     self.model.train()

        self.optimizer.zero_grad()
        return sum(total_loss)/len(total_loss), global_step
