import glob
import json
import os
import logging
import string
import random
from os.path import join as pj
from typing import List
from itertools import product
from distutils.dir_util import copy_tree

from .trainer import Trainer, to_list
from .automatic_evaluation import evaluate
from .data import DEFAULT_CACHE_DIR


__all__ = 'GridSearcher'


def get_random_string(length: int = 6, exclude: List = None):
    tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    if exclude:
        while tmp in exclude:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    return tmp


class GridSearcher:
    """ Grid search (epoch, batch, lr, random_seed, label_smoothing) """

    def __init__(self, checkpoint_dir: str, dataset_path: str = "asahi417/qg_squad", dataset_name: str = "default",
                 input_types: List or str = 'paragraph_answer', output_types: List or str = 'question',
                 prefix_types: List or str = 'qg', model: str = 't5-small', fp16: bool = False,
                 gradient_accumulation_steps: List or int = 4, metric: str = 'validation/Bleu_4',
                 epoch: int = 10, epoch_partial: int = 2, n_max_config: int = 5, max_length: int = 512,
                 max_length_eval: int = None, max_length_output: int = 32, max_length_output_eval: int = None,
                 prediction_aggregation: str = 'first', prediction_level: str = 'sentence',
                 batch: int = 128, batch_eval: int = 32, n_beams_eval: int = 4, lr: List or float = 1e-4,
                 label_smoothing: List or float = None, random_seed: List or int = 42, language: str = 'en',
                 normalize: bool = True):

        # evaluation configs
        max_length_eval = max_length if max_length_eval is None else max_length_eval
        max_length_output_eval = max_length_output if max_length_output_eval is None else max_length_output_eval
        self.eval_config = {
            'max_length_eval': max_length_eval, 'max_length_output_eval': max_length_output_eval,
            'n_beams_eval': n_beams_eval, 'prediction_aggregation': prediction_aggregation,
            'prediction_level': prediction_level, 'language': language, 'normalize': normalize
        }

        # static configs
        self.static_config = {
            'dataset_path': dataset_path, 'dataset_name': dataset_name, 'input_types': input_types,
            'output_types': output_types, 'model': model, 'fp16': fp16, 'batch': batch, 'epoch': epoch,
            'max_length': max_length, 'max_length_output': max_length_output, 'prefix_types': prefix_types
        }

        # dynamic config
        self.epoch = epoch
        self.epoch_partial = epoch_partial
        self.batch_eval = batch_eval
        self.checkpoint_dir = checkpoint_dir
        self.n_max_config = n_max_config
        self.split, self.metric = metric.split('/')

        self.dynamic_config = {
            'lr': to_list(lr),
            'label_smoothing': to_list(label_smoothing),
            'random_seed': to_list(random_seed),
            'gradient_accumulation_steps': to_list(gradient_accumulation_steps),
        }

        self.all_dynamic_configs = list(product(
            self.dynamic_config['lr'],
            self.dynamic_config['label_smoothing'],
            self.dynamic_config['random_seed'],
            self.dynamic_config['gradient_accumulation_steps'],
        ))

    def initialize_searcher(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path_to_config = pj(self.checkpoint_dir, 'config_static.json')
        if os.path.exists(path_to_config):
            with open(path_to_config) as f:
                tmp = json.load(f)
            tmp_v = [tmp[k] for k in sorted(tmp.keys())]
            static_tmp_v = [self.static_config[k] for k in sorted(tmp.keys())]
            assert tmp_v == static_tmp_v, f'{str(tmp_v)}\n not matched \n{str(static_tmp_v)}'
        path_to_d_config = pj(self.checkpoint_dir, 'config_dynamic.json')
        if os.path.exists(path_to_d_config):
            with open(path_to_d_config) as f:
                tmp = json.load(f)

            tmp_v = [tmp[k] for k in sorted(tmp.keys())]
            dynamic_tmp_v = [self.dynamic_config[k] for k in sorted(tmp.keys())]

            assert tmp_v == dynamic_tmp_v
        path_to_e_config = pj(self.checkpoint_dir, 'config_eval.json')
        if os.path.exists(path_to_e_config):
            with open(path_to_e_config) as f:
                tmp = json.load(f)
            tmp_v = [tmp[k] for k in sorted(tmp.keys())]
            eval_tmp_v = [self.eval_config[k] for k in sorted(tmp.keys())]
            assert tmp_v == eval_tmp_v, f'{str(tmp_v)}\n not matched \n{str(eval_tmp_v)}'

        with open(path_to_config, 'w') as f:
            json.dump(self.static_config, f)
        with open(path_to_d_config, 'w') as f:
            json.dump(self.dynamic_config, f)
        with open(path_to_e_config, 'w') as f:
            json.dump(self.eval_config, f)

        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler(pj(self.checkpoint_dir, 'grid_search.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)
        logging.info(f'INITIALIZE GRID SEARCHER: {len(self.all_dynamic_configs)} configs to try')

    def get_evaluator(self, overwrite: bool):
        # configure evaluation data types
        input_types = to_list(self.static_config['input_types'], sorting=False)
        output_types = to_list(self.static_config['output_types'], sorting=False)
        assert len(input_types) == len(output_types)
        if self.static_config['prefix_types'] is None:
            prefix_types = [None] * len(input_types)
        else:
            prefix_types = to_list(self.static_config['prefix_types'], sorting=False)
        tmp = [(i, o, p) for i, o, p in zip(input_types, output_types, prefix_types) if o == 'question']
        assert len(tmp) == 1
        i, o, p = tmp[0]
        prefix = pj(
            DEFAULT_CACHE_DIR,
            "encoded_feature",
            f"{self.static_config['dataset_path']}{'.' + self.static_config['dataset_name'] if self.static_config['dataset_name'] != 'default' else ''}"
            f"{self.static_config['model']}.{self.eval_config['max_length_eval']}.{self.eval_config['max_length_output_eval']}.{i}.{o}"
        )
        data_cache_paths = {split: f"{prefix}.{split}.{p}.pkl" for split in ['test', 'validation']}

        def get_evaluation(_checkpoint_dir_model, use_auth_token: bool = False):
            return evaluate(
                export_dir=pj(_checkpoint_dir_model, 'eval'),
                batch_size=self.batch_eval,
                n_beams=self.eval_config['n_beams_eval'],
                model=_checkpoint_dir_model,
                max_length=self.eval_config['max_length_eval'],
                overwrite=overwrite,
                max_length_output=self.eval_config['max_length_output_eval'],
                bleu_only=True,
                dataset_path=self.static_config['dataset_path'],
                dataset_name=self.static_config['dataset_name'],
                input_type=i,
                output_type=o,
                prediction_aggregation=self.eval_config['prediction_aggregation'],
                prediction_level=self.eval_config['prediction_level'],
                language=self.eval_config['language'],
                use_auth_token=use_auth_token,
                data_caches=data_cache_paths)
        return get_evaluation

    def run(self, interval: int = 25, overwrite: bool = False, use_auth_token: bool = False):

        self.initialize_searcher()

        # instantiate evaluator
        evaluator = self.get_evaluator(overwrite)
        ###########
        # 1st RUN #
        ###########
        checkpoints = []
        ckpt_exist = {}
        for trainer_config in glob.glob(pj(self.checkpoint_dir, 'model_*', 'trainer_config.json')):
            with open(trainer_config, 'r') as f:
                ckpt_exist[os.path.dirname(trainer_config)] = json.load(f)

        for n, dynamic_config in enumerate(self.all_dynamic_configs):
            logging.info(f'## 1st RUN: Configuration {n}/{len(self.all_dynamic_configs)} ##')
            config = self.static_config.copy()
            tmp_dynamic_config = {
                'lr': dynamic_config[0],
                'label_smoothing': dynamic_config[1],
                'random_seed': dynamic_config[2],
                'gradient_accumulation_steps': dynamic_config[3]
            }
            config.update(tmp_dynamic_config)
            ex_dynamic_config = [(k_, [v[k] for k in sorted(tmp_dynamic_config.keys())]) for k_, v in ckpt_exist.items()]
            tmp_dynamic_config = [tmp_dynamic_config[k] for k in sorted(tmp_dynamic_config.keys())]
            duplicated_ckpt = [k for k, v in ex_dynamic_config if v == tmp_dynamic_config]

            if len(duplicated_ckpt) == 1:
                logging.info(f'skip as the config exists at {duplicated_ckpt} \n{config}')
                checkpoint_dir = duplicated_ckpt[0]
            elif len(duplicated_ckpt) == 0:
                ckpt_name_exist = [os.path.basename(k).replace('model_', '') for k in ckpt_exist.keys()]
                ckpt_name_made = [os.path.basename(c).replace('model_', '') for c in checkpoints]
                model_ckpt = get_random_string(exclude=ckpt_name_exist + ckpt_name_made)
                checkpoint_dir = pj(self.checkpoint_dir, f'model_{model_ckpt}')
            else:
                raise ValueError(f'duplicated checkpoints are found: \n {duplicated_ckpt}')

            if not os.path.exists(pj(checkpoint_dir, f'epoch_{self.epoch_partial}')):
                trainer = Trainer(checkpoint_dir=checkpoint_dir, disable_log=True, **config)
                trainer.train(
                    epoch_partial=self.epoch_partial, epoch_save=1, interval=interval, use_auth_token=use_auth_token)

            checkpoints.append(checkpoint_dir)

        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info(f'## 1st RUN (EVAL): Configuration {n}/{len(checkpoints)} ##')
            checkpoint_dir_model = pj(checkpoint_dir, f'epoch_{self.epoch_partial}')
            metric = evaluator(checkpoint_dir_model, use_auth_token=use_auth_token)
            # except Exception:
            metrics[checkpoint_dir_model] = metric[self.split][self.metric]

        metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        with open(pj(self.checkpoint_dir, f"metric.{self.eval_config['prediction_level']}.1st.json"), 'w') as f:
            json.dump(metrics, f)

        logging.info(f'1st RUN RESULTS ({self.split}/{self.metric})')
        for n, (k, v) in enumerate(metrics):
            logging.info(f'\t * rank: {n} | metric: {round(v, 3)} | model: {k} |')

        if self.epoch_partial == self.epoch:
            logging.info('No 2nd phase as epoch_partial == epoch')
            return

        ###########
        # 2nd RUN #
        ###########
        metrics = metrics[:min(len(metrics), self.n_max_config)]
        checkpoints = []
        for n, (checkpoint_dir_model, _metric) in enumerate(metrics):
            logging.info(f'## 2nd RUN: Configuration {n}/{len(metrics)}: {self.split}/{self.metric} = {_metric}')
            model_ckpt = os.path.dirname(checkpoint_dir_model)
            if not os.path.exists(pj(model_ckpt, f'epoch_{self.epoch}')):
                trainer = Trainer(checkpoint_dir=model_ckpt, disable_log=True)
                trainer.train(epoch_save=1, interval=interval, use_auth_token=use_auth_token)

            checkpoints.append(model_ckpt)

        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info(f'## 2nd RUN (EVAL): Configuration {n}/{len(checkpoints)} ##')
            for checkpoint_dir_model in sorted(glob.glob(pj(checkpoint_dir, 'epoch_*'))):
                if int(checkpoint_dir_model.split(os.path.sep)[-1].replace('epoch_', '')) > self.static_config['epoch']:
                    continue
                metric = evaluator(checkpoint_dir_model, use_auth_token=use_auth_token)
                metrics[checkpoint_dir_model] = metric[self.split][self.metric]

        metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        logging.info(f'2nd RUN RESULTS: \n{str(metrics)}')
        for n, (k, v) in enumerate(metrics):
            logging.info(f'\t * rank: {n} | metric: {round(v, 3)} | model: {k} |')

        with open(pj(self.checkpoint_dir, f"metric.{self.eval_config['prediction_level']}.2nd.json"), 'w') as f:
            json.dump(metrics, f)

        best_model_ckpt, best_metric_score = metrics[0]
        epoch = int(best_model_ckpt.split(os.path.sep)[-1].replace('epoch_', ''))
        if epoch == self.static_config['epoch']:
            ###########
            # 3rd RUN #
            ###########
            best_model_dir = os.path.dirname(best_model_ckpt)
            logging.info(f'## 3rd RUN: target model: {best_model_dir} (metric: {best_metric_score}) ##')
            metric_list = [[epoch, best_metric_score]]
            with open(pj(best_model_dir, 'trainer_config.json')) as f:
                config = json.load(f)
            while True:
                epoch += 1
                logging.info(f'## 3rd RUN (TRAIN): epoch {epoch} ##')
                config['epoch'] = epoch
                with open(pj(best_model_dir, 'trainer_config.additional_training.json'), 'w') as f:
                    json.dump(config, f)
                checkpoint_dir_model = pj(best_model_dir, f'epoch_{epoch}')
                if not os.path.exists(checkpoint_dir_model):
                    trainer = Trainer(
                        checkpoint_dir=best_model_dir, config_file='trainer_config.additional_training.json', disable_log=True)
                    trainer.train(epoch_save=1, interval=interval, use_auth_token=use_auth_token)
                logging.info(f'## 3rd RUN (EVAL): epoch {epoch} ##')
                metric = evaluator(checkpoint_dir_model, use_auth_token=use_auth_token)
                tmp_metric_score = metric[self.split][self.metric]
                metric_list.append([epoch, tmp_metric_score])
                logging.info(f'\t tmp metric: {tmp_metric_score}')
                if best_metric_score > tmp_metric_score:
                    logging.info('\t finish 3rd phase (no improvement)')
                    break
                else:
                    logging.info(f'\t tmp metric improved the best model ({best_metric_score} --> {tmp_metric_score})')
                    best_metric_score = tmp_metric_score
            logging.info(f'3rd RUN RESULTS: {best_model_dir}')
            for k, v in metric_list:
                logging.info(f'\t epoch {k}: {v}')

            metric_list = sorted(metric_list, key=lambda x: x[1], reverse=True)
            with open(pj(self.checkpoint_dir, f"metric.{self.eval_config['prediction_level']}.3rd.json"), 'w') as f:
                json.dump(metric_list, f)
            best_model_ckpt = pj(best_model_dir, f'epoch_{metric_list[0][0]}')

        copy_tree(best_model_ckpt, pj(self.checkpoint_dir, 'best_model'))
