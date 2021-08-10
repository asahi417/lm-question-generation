""" evaluate T5. """
import argparse
import logging
from t5qg import evaluate_qg


def get_options():
    parser = argparse.ArgumentParser(description='evaluate T5.')
    # model training configuration
    parser.add_argument('-d', '--dataset', help='dataset', default='squad', type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
    parser.add_argument('-e', '--export-dir', help='export dir', required=True, type=str)
    parser.add_argument('--num-beams', help='n  beams', default=4, type=int)
    parser.add_argument('--language', help='language', default='en', type=str)
    parser.add_argument('--max-length', default=512, type=int, help='max sequence length for input sequence')
    parser.add_argument('--max-length-output', default=32, type=int, help='max sequence length for output sequence')
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    # monitoring parameter
    parser.add_argument('--debug', help='log mode', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    evaluate_qg(
        model=opt.model,
        export_dir=opt.export_dir,
        dataset=opt.dataset,
        language=opt.language.split(','),
        batch=opt.batch,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        num_beams=opt.num_beams,
        random_seed=opt.random_seed
    )


if __name__ == '__main__':
    main()
