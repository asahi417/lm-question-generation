""" Training model. """
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from .lm_t5 import T5Summarizer
from .util import get_linear_schedule_with_warmup, fix_seed, Config, get_dataset


class Trainer:

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: str = "cnn_dailymail",
                 dataset_argument: str = "3.0.0",
                 task_prefix: str = 'summarize:',
                 model: str = 'google/mt5-small',
                 max_length: int = 512,
                 max_length_output: int = 128,
                 epoch: int = 5,
                 batch: int = 32,
                 lr: float = 0.00005,
                 lr_decay: bool = False,
                 lr_warmup: int = 100,
                 weight_decay: float = 0,
                 optimizer: str = 'adam',
                 momentum: float = 1.0,
                 fp16: bool = False,
                 random_seed: int = 0,
                 cache_dir: str = None,
                 gradient_accumulation_steps: int = 1):

        logging.info('initialize model trainer')
        # config
        self.data_cache_dir = '{}/.data_cache'.format(checkpoint_dir)
        self.config = Config(
            checkpoint_dir=checkpoint_dir,
            dataset=dataset,
            dataset_argument=dataset_argument,
            task_prefix=task_prefix,
            model=model,
            max_length=max_length,
            max_length_output=max_length_output,
            epoch=epoch,
            lr_warmup=lr_warmup,
            batch=batch,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            optimizer=optimizer,
            momentum=momentum,
            fp16=fp16,
            random_seed=random_seed,
            gradient_accumulation_steps=gradient_accumulation_steps)
        self.cache_dir = cache_dir
        self.model = T5Summarizer(model=self.config.model,
                                  max_length=self.config.max_length,
                                  max_length_output=self.config.max_length_output,
                                  task_prefix=self.config.task_prefix,
                                  cache_dir=self.cache_dir)
        # setup
        fix_seed(self.config.random_seed)

        # setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        model_parameters = [
            {"params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        if self.config.optimizer == 'adamax':
            self.optimizer = torch.optim.Adamax(model_parameters, lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model_parameters, lr=self.config.lr, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.AdamW(model_parameters, lr=self.config.lr)
        else:
            raise ValueError('unknown optimizer: {}'.format(self.config.optimizer))

        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.lr_warmup,
            num_training_steps=self.config.epoch if self.config.lr_decay else None)

        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)

    def save(self, current_epoch):
        save_dir = '{}/epoch_{}'.format(self.config.checkpoint_dir, current_epoch + 1)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save(save_dir)

    def train(self, num_workers: int = 0, epoch_save: int = 1, interval = 10):
        """ Train model.

        :param num_workers: Workers for DataLoader.
        :param epoch_save: Save the model every this epoch.
        """

        logging.info('dataset preprocessing')
        raw_input, raw_output = get_dataset(self.config.dataset, self.config.dataset_argument, split='train')
        loader = self.model.get_data_loader(
            raw_input, raw_output, batch_size=self.config.batch, shuffle=True, drop_last=True, num_workers=num_workers,
            cache_path='{}/data.{}.{}.{}.pkl'.format(
                self.data_cache_dir, self.config.max_length, self.config.max_length_output, self.config.task_prefix))
        self.model.train()

        logging.info('start model training')
        global_step = 0
        writer = SummaryWriter(log_dir=self.config.checkpoint_dir)
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.config.epoch):  # loop over the epoch
                mean_loss, global_step = self.train_single_epoch(loader, global_step, writer, interval)
                logging.info('[epoch {}/{}] average loss: {}, lr: {}'.format(
                    e, self.config.epoch, round(mean_loss, 3), self.optimizer.param_groups[0]['lr']))
                if (e + 1) % epoch_save == 0 and (e + 1) != 0:
                    self.save(e)
        writer.close()
        self.save(e)
        logging.info('complete training: model ckpt was saved at {}'.format(self.config.checkpoint_dir))

    def train_single_epoch(self, data_loader, global_step: int, writer, interval):
        total_loss = []
        self.optimizer.zero_grad()
        for n, encode in enumerate(data_loader):

            loss = self.model(encode)
            self.scaler.scale(loss).backward()
            total_loss.append(loss.cpu().item())
            if (n + 1) % self.config.gradient_accumulation_steps != 0:
                continue

            global_step += 1
            inst_loss = sum(total_loss[-self.config.gradient_accumulation_steps:])/self.config.gradient_accumulation_steps
            writer.add_scalar('train/loss', inst_loss, global_step)
            writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            # optimizer update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if global_step % interval == 0:
                logging.info('\t * (global step {}: loss: {}, lr: {}'.format(
                    global_step, inst_loss, self.optimizer.param_groups[0]['lr']))

                if n == 0:
                    self.model.eval()
                    sentence = self.model.tokenizer.batch_decode(encode['input_ids'], skip_special_tokens=True)
                    label = self.model.tokenizer.batch_decode(encode['labels'], skip_special_tokens=True)
                    out, _, _, _ = self.model.get_prediction(sentence)
                    for _n, i in enumerate(zip(out, label, sentence)):
                        logging.debug('\t * {}'.format(i))
                        if _n > 5:
                            break
                    self.model.train()

        self.optimizer.zero_grad()

        return sum(total_loss)/len(total_loss), global_step
