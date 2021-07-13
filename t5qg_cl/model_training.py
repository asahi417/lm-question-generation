""" Fine-tune T5. """
import argparse
import logging

from t5qg import Trainer


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune T5.')
    # model training configuration
    parser.add_argument('-c', '--checkpoint-dir', help='directory to save checkpoint', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='dataset', default='squad', type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', default='google/mt5-small', type=str)
    parser.add_argument('-e', '--epoch', help='epoch', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=64, type=int)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='', default=8, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--task-type', help='task type', default='ans_ext,qa,qg', type=str)
    parser.add_argument('--language', help='language', default='en', type=str)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument("--lr-warmup", help="linear warmup of lr", default=None, type=int)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--max-length', default=512, type=int, help='max sequence length for input sequence')
    parser.add_argument('--max-length-output', default=32, type=int, help='max sequence length for output sequence')
    # monitoring parameter
    parser.add_argument('--debug', help='log mode', action='store_true')
    parser.add_argument('--activate-tensorboard', help='log mode', action='store_true')
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--epoch-save', default=1, type=int)
    parser.add_argument('--interval', default=50, type=int)
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    trainer = Trainer(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        task_type=opt.task_type.split(','),
        language=opt.language.split(','),
        random_seed=opt.random_seed,
        model=opt.model,
        epoch=opt.epoch,
        lr=opt.lr,
        lr_warmup=opt.lr_warmup,
        batch=opt.batch,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        fp16=opt.fp16,
        gradient_accumulation_steps=opt.gradient_accumulation_steps
    )
    trainer.train(
        epoch_save=opt.epoch_save,
        interval=opt.interval,
        num_workers=opt.num_workers,
        activate_tensorboard=opt.activate_tensorboard)


if __name__ == '__main__':
    main()
