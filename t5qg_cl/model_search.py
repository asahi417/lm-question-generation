""" Fine-tune T5. """
import argparse
import logging

from t5qg import GridSearcher


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune T5 (with grid search).')
    # model training configuration
    parser.add_argument('-c', '--checkpoint-dir', help='directory to save checkpoint', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='dataset', default='squad', type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', default='t5-small', type=str)
    parser.add_argument('-e', '--epoch', help='epoch', default=10, type=int)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='', default=4, type=int)
    parser.add_argument('--task-type', help='task type', default='qg', type=str)
    parser.add_argument('--language', help='language', default='en', type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument("--lr-warmup", help="linear warmup of lr", default=None, type=int)

    # monitoring parameter
    parser.add_argument('--debug', help='log mode', action='store_true')

    # grid search
    parser.add_argument('--n-beams-eval', help='', default=4, type=int)
    parser.add_argument('--batch-eval', help='', default=128, type=int)
    parser.add_argument('--metric', default='dev/BLeu_4', type=str)
    parser.add_argument('--n-max-config', help='', default=5, type=int)
    parser.add_argument('--epoch-partial', help='epoch', default=2, type=int)
    parser.add_argument('--max-length', default='512', type=str, help='max sequence length for input sequence')
    parser.add_argument('--max-length-output', default='32', type=str, help='max sequence length for output sequence')
    parser.add_argument('-l', '--lr', help='learning rate', default='1e-5,5e-5,1e-4,5e-4,1e-3', type=str)
    parser.add_argument('--label-smoothing', help='label smoothing', default='0.0,0.1,0.2', type=str)
    parser.add_argument('--random-seed', help='random seed', default='1234', type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    trainer = GridSearcher(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        task_type=opt.task_type.split(','),
        language=opt.language.split(','),
        model=opt.model,
        lr_warmup=opt.lr_warmup,
        fp16=opt.fp16,
        epoch=opt.epoch,
        metric=opt.metric,
        epoch_partial=opt.epoch_partial,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        batch=opt.batch,
        max_length=[int(i) for i in opt.max_length.split(',')],
        max_length_output=[int(i) for i in opt.max_length_output.split(',')],
        n_max_config=opt.n_max_config,
        lr=[float(i) for i in opt.lr.split(',')],
        label_smoothing=[float(i) for i in opt.label_smoothing.split(',')],
        random_seed=[int(i) for i in opt.random_seed.split(',')],
        batch_eval=opt.batch_eval,
        n_beams_eval=opt.n_beams_eval
    )
    trainer.run()


if __name__ == '__main__':
    main()
