import glob
import json
import os
import logging
from typing import List
from itertools import product

from .trainer import Trainer
from .evaluator import evaluate_qg


class GridSearcher:
    """ Grid search (epoch, batch, lr, random_seed, label_smoothing) """

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: str = "squad",
                 model: str = 't5-small',
                 task_type: List or str = 'qg',
                 language: List or str = 'en',
                 lr_warmup: int = 100,
                 fp16: bool = False,
                 gradient_accumulation_steps: int = 4,
                 epoch: int = 10,
                 epoch_partial: int = 2,
                 n_max_config: int = 5,
                 max_length: List or int = 512,
                 max_length_output: List or int = 32,
                 batch: List or int = 128,
                 batch_eval: List or int = 128,
                 n_beams_eval: List or int = 4,
                 lr: List or float = 1e-4,
                 label_smoothing: List or float = None,
                 random_seed: List or int = 42,
                 metric: str = 'dev/Bleu_4'):

        # initialize searcher
        assert not os.path.exists(checkpoint_dir), 'work directory `{}` is not empty'.format(checkpoint_dir)

        # static configs
        self.static_config = {
            # 'checkpoint_dir': checkpoint_dir,
            'dataset': dataset,
            'model': model,
            'task_type': task_type,
            'language': language,
            'lr_warmup': lr_warmup,
            'fp16': fp16,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'epoch': epoch,
            'disable_log': True
        }

        # dynamic config
        assert epoch_partial < epoch
        self.epoch = epoch
        self.epoch_partial = epoch_partial
        self.batch_eval = batch_eval
        self.n_beams_eval = n_beams_eval
        self.checkpoint_dir = checkpoint_dir
        self.n_max_config = n_max_config
        self.split, self.metric = metric.split('/')

        def to_list(_val):
            if type(_val) != list:
                return [_val]
            return sorted(_val, reverse=True)

        self.dynamic_config = {
            'max_length': to_list(max_length),
            'max_length_output': to_list(max_length_output),
            'batch': to_list(batch),
            'lr': to_list(lr),
            'label_smoothing': to_list(label_smoothing),
            'random_seed': to_list(random_seed)
        }

        self.all_dynamic_configs = list(product(
            self.dynamic_config['max_length'],
            self.dynamic_config['max_length_output'],
            self.dynamic_config['batch'],
            self.dynamic_config['lr'],
            self.dynamic_config['label_smoothing'],
            self.dynamic_config['random_seed']))

    def initialize_searcher(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if os.path.exists('{}/config_static.json'):
            with open('{}/config_static.json') as f:
                tmp = json.load(f)
            assert tmp == self.static_config

        if os.path.exists('{}/config_dynamic.json'):
            with open('{}/config_dynamic.json') as f:
                tmp = json.load(f)
            assert tmp == self.dynamic_config

        with open('{}/config_static.json', 'w') as f:
            json.dump(self.static_config, f)
        with open('{}/config_dynamic.json', 'w') as f:
            json.dump(self.dynamic_config, f)

        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler('{}/grid_search.log'.format(self.checkpoint_dir))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)

        logging.info('INITIALIZE GRID SEARCHER: {} configs to try'.format(len(self.all_dynamic_configs)))

    def run(self):

        self.initialize_searcher()

        ###########
        # 1st RUN #
        ###########

        checkpoints = []
        for n, dynamic_config in enumerate(self.all_dynamic_configs):
            logging.info('## 1st RUN: Configuration {}/{} ##'.format(n, len(dynamic_config)))
            config = self.static_config.copy()
            config.update({
                'max_length': dynamic_config[0], 'max_length_output': dynamic_config[1], 'batch': dynamic_config[2],
                'lr': dynamic_config[3], 'label_smoothing': dynamic_config[4], 'random_seed': dynamic_config[5],
            })
            checkpoint_dir = '{}/model_{}'.format(self.checkpoint_dir, n)
            if not os.path.exists('{}/epoch_{}'.format(checkpoint_dir, self.epoch_partial)):
                trainer = Trainer(checkpoint_dir=checkpoint_dir, **config)
                trainer.train(epoch_partial=self.epoch_partial, epoch_save=None)
            checkpoints.append(checkpoint_dir)

        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info('## 1st RUN (EVAL): Configuration {}/{} ##'.format(n, len(checkpoints)))
            checkpoint_dir_model = '{}/epoch_{}'.format(checkpoint_dir, self.epoch_partial)
            if not os.path.exists('{}/eval/metric.json'.format(checkpoint_dir_model)):
                evaluate_qg(model=checkpoint_dir_model, export_dir='{}/eval'.format(checkpoint_dir_model))

            with open('{}/eval/metric.json'.format(checkpoint_dir_model), 'r') as f:
                metric = json.load(f)
                metrics[checkpoint_dir_model] = metric[self.split][self.metric]

        metric = sorted(metric.items(), key=lambda x: x[1], reverse=True)
        logging.info('1st RUN RESULTS ({}/{})'.format(self.split, self.metric))
        for n, (k, v) in enumerate(metric):
            logging.info(' rank: {} | metric: {} | model: {} |'.format(n, round(v, 3), k))
        with open('{}/metric.1st.json', 'w') as f:
            json.dump(metric, f)

        ###########
        # 2nd RUN #
        ###########

        metric = metric[:min(len(metric), self.n_max_config)]
        checkpoints = []
        for n, (model_num, _metric) in enumerate(metric):
            logging.info('## 2nd RUN: Configuration {}/{}: {}/{} = {}'.format(
                n, len(metric), self.split, self.metric, _metric))
            checkpoint_dir = '{}/model_{}'.format(self.checkpoint_dir, n)
            if os.path.exists('{}/epoch_{}'.format(checkpoint_dir, self.epoch)):
                trainer = Trainer(checkpoint_dir=checkpoint_dir)
                trainer.train()

            checkpoints.append(checkpoint_dir)

        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info('## 2nd RUN (EVAL): Configuration {}/{} ##'.format(n, len(checkpoints)))
            for checkpoint_dir_model in glob.glob('{}/epoch_*'.format(checkpoint_dir)):

                if not os.path.exists('{}/eval/metric.json'.format(checkpoint_dir_model)):
                    evaluate_qg(model=checkpoint_dir, export_dir='{}/eval'.format(checkpoint_dir_model))

                evaluate_qg(model=checkpoint_dir_model, export_dir='{}/eval'.format(checkpoint_dir_model))
                with open('{}/eval/metric.json'.format(checkpoint_dir_model), 'r') as f:
                    metric = json.load(f)
                    metrics[checkpoint_dir_model] = metric[self.split][self.metric]

        metric = sorted(metric.items(), key=lambda x: x[1], reverse=True)
        logging.info('2nd RUN RESULTS: \n{}'.format(str(metric)))
        for n, (k, v) in enumerate(metric):
            logging.info(' rank: {} | metric: {} | model: {} |'.format(n, round(v, 3), k))

        with open('{}/metric.2nd.json', 'w') as f:
            json.dump(metric, f)

