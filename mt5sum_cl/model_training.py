""" Fine-tune transformers on counterfactual sentence classification. """
import argparse
import logging

from mt5sum import Trainer


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on counterfactual sentence classification.')
    parser.add_argument('-c', '--checkpoint-dir', help='directory to save checkpoint', required=True, type=str)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('-d', '--dataset', help='dataset', default='cnn_dailymail', type=str)
    parser.add_argument('-a', '--dataset-argument', help='dataset argument', default='3.0.0', type=str)
    parser.add_argument('--task-prefix', help='task prefix', default='summarize:', type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', default='google/mt5-base', type=str)
    parser.add_argument('-e', '--epoch', help='epoch', default=10, type=int)
    parser.add_argument('--lr', help='learning rate', default=5e-4, type=float)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('--lr-decay', help='linear decay of learning rate after warmup', action='store_true')
    parser.add_argument("--lr-warmup", help="linear warmup of lr", default=100, type=int)
    parser.add_argument('--optimizer', help='optimizer `adam`/`adamax`/`adam`', default='adam', type=str)
    parser.add_argument("--momentum", help="sgd momentum", default=1.0, type=float)
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='', default=8, type=int)
    parser.add_argument('--max-length', default=512, type=int, help='max sequence length for input sequence')
    parser.add_argument('--max-length-out', default=128, type=int, help='max sequence length for output sequence')
    parser.add_argument('--weight-decay', help='weight decay', default=0, type=float)
    parser.add_argument('--debug', help='log mode', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    trainer = Trainer(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        dataset_argument=opt.dataset_argument,
        task_prefix=opt.task_prefix,
        random_seed=opt.random_seed,
        model=opt.model,
        epoch=opt.epoch,
        lr=opt.lr,
        lr_decay=opt.lr_decay,
        lr_warmup=opt.lr_warmup,
        optimizer=opt.optimizer,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
        batch=opt.batch,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        fp16=opt.fp16,
        gradient_accumulation_steps=opt.gradient_accumulation_steps
    )
    trainer.train()


if __name__ == '__main__':
    main()
