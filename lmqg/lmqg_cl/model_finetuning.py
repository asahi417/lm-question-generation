""" Parameter optimization of Question Generation Finetuning. """
import argparse
import logging

from lmqg import GridSearcher, Trainer


def arguments(parser):
    parser.add_argument('-c', '--checkpoint-dir', help='directory to save checkpoint', required=True, type=str)
    parser.add_argument('-d', '--dataset-path', help='huggingface datasets alias', default='lmqg/qg_squad', type=str)
    parser.add_argument('--dataset-name', help='huggingface datasets name', default='default', type=str)
    parser.add_argument('-i', '--input-types',
                        help='input to the model (`paragraph_answer`/`sentence_answer`/`paragraph_sentence`)',
                        nargs='+', default='paragraph_answer', type=str)
    parser.add_argument('-o', '--output-types',
                        help='output of the model (`answer`/`question`)',
                        nargs='+', default='question', type=str)
    parser.add_argument('-p', '--prefix-types',
                        help='prefix type (`qg`/`ae`) which should be specified for T5 but not for mT5, BART, or mBART',
                        nargs='+', default=None, type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', default='facebook/bart-large', type=str)
    parser.add_argument('-e', '--epoch', help='epoch', default=8, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('--use-auth-token', help='', action='store_true')
    parser.add_argument('--max-length', default=512, type=int, help='max sequence length for input sequence')
    parser.add_argument('--max-length-output', default=32, type=int, help='max sequence length for output sequence')
    parser.add_argument('--interval', default=50, type=int)
    return parser


def arguments_training(parser):
    # training config
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--label-smoothing', help='label smoothing', default=0.3, type=float)
    parser.add_argument('-l', '--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='number of gradient accumulation', default=1, type=int)
    # monitoring parameter
    parser.add_argument('--epoch-save', default=None, type=int)
    return parser


def arguments_training_search(parser):
    parser.add_argument('--n-beams-eval', help='number of beam at evaluation', default=4, type=int)
    parser.add_argument('--batch-eval', help='batch size at evaluation', default=8, type=int)
    parser.add_argument('--prediction-aggregation', help='`first`/`short`/`last`/`long`/`middle`', default='first', type=str)
    parser.add_argument('--prediction-level', help="`sentence`/`paragraph`/`answer`", default='answer', type=str)
    parser.add_argument('--metric', help='evaluation metric for grid search', default='validation/Bleu_4', type=str)
    parser.add_argument('--n-max-config', help='number of maximum configuration to select for the second stage',
                        default=5, type=int)
    parser.add_argument('--epoch-partial', help='epoch', default=2, type=int)
    parser.add_argument('--max-length-eval', help='max sequence length for input sequence at evaluation',
                        default=512, type=int)
    parser.add_argument('--max-length-output-eval', help='max sequence length for output sequence at evaluation',
                        default=64, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', nargs='+', default=[1e-05, 5e-05, 1e-04], type=float)
    parser.add_argument('--label-smoothing', help='label smoothing', nargs='+', default=[0.0, 0.15], type=float)
    parser.add_argument('--random-seed', help='random seed', nargs='+', default=[1], type=int)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='number of gradient accumulation',
                        nargs='+',  default=[1, 2, 4, 8], type=int)
    parser.add_argument('--overwrite', help='overwrite metric already computed', action='store_true')
    parser.add_argument('--language', help='language needs to be specified for evaluation mainly',
                        default='en', type=str)
    return parser


def main_training():
    parser = argparse.ArgumentParser(description='Fine-tuning on QG.')
    parser = arguments(parser)
    parser = arguments_training(parser)
    opt = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    trainer = Trainer(
        checkpoint_dir=opt.checkpoint_dir,
        dataset_path=opt.dataset_path,
        dataset_name=opt.dataset_name,
        input_types=opt.input_types,
        output_types=opt.output_types,
        prefix_types=opt.prefix_types if opt.prefix_types is not None else None,
        random_seed=opt.random_seed,
        model=opt.model,
        epoch=opt.epoch,
        lr=opt.lr,
        batch=opt.batch,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        fp16=opt.fp16,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        label_smoothing=opt.label_smoothing)
    trainer.train(interval=opt.interval, epoch_save=opt.epoch_save, use_auth_token=opt.use_auth_token)


def main_training_search():
    parser = argparse.ArgumentParser(description='Fine-tuning on QG with Grid Search.')
    parser = arguments(parser)
    parser = arguments_training_search(parser)
    opt = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    trainer = GridSearcher(
        checkpoint_dir=opt.checkpoint_dir,
        dataset_path=opt.dataset_path,
        dataset_name=opt.dataset_name,
        input_types=opt.input_types,
        output_types=opt.output_types,
        prefix_types=opt.prefix_types if opt.prefix_types is not None else None,
        model=opt.model,
        fp16=opt.fp16,
        epoch=opt.epoch,
        metric=opt.metric,
        epoch_partial=opt.epoch_partial,
        batch=opt.batch,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        n_max_config=opt.n_max_config,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        lr=opt.lr,
        label_smoothing=opt.label_smoothing,
        random_seed=opt.random_seed,
        batch_eval=opt.batch_eval,
        n_beams_eval=opt.n_beams_eval,
        prediction_aggregation=opt.prediction_aggregation,
        prediction_level=opt.prediction_level,
        max_length_output_eval=opt.max_length_output_eval,
        max_length_eval=opt.max_length_eval,
        language=opt.language
    )
    trainer.run(interval=opt.interval, overwrite=opt.overwrite, use_auth_token=opt.use_auth_token)

