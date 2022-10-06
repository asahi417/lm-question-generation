""" SQuAD QG evaluation (sentence/answer level) """
import argparse
import json

from lmqg import evaluate


def get_options():
    parser = argparse.ArgumentParser(description='QG evaluation on SQuAD.')
    parser.add_argument('-m', '--model-checkpoint', default=None, type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--prediction-aggregation', help='', default='first', type=str)
    parser.add_argument('--prediction-level', help="'sentence', 'context', 'answer'", default='sentence', type=str)
    parser.add_argument('--n-beams', default=4, type=int)
    parser.add_argument('-e', '--export-dir', required=True, type=str)
    parser.add_argument('--hyp-test', default=None, type=str)
    parser.add_argument('--hyp-dev', default=None, type=str)
    parser.add_argument('--max-length', default=512, type=int, help='')
    parser.add_argument('--max-length-output', default=64, type=int, help='')
    parser.add_argument('-d', '--dataset-path', help='huggingface datasets alias', default='lmqg/qg_squad', type=str)
    parser.add_argument('--dataset-name', help='huggingface datasets name', default='default', type=str)
    parser.add_argument('-i', '--input-type', help='', default='paragraph_answer', type=str)
    parser.add_argument('-o', '--output-type', help='', default='question', type=str)
    parser.add_argument('--language', help='', default='en', type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    parser.add_argument('--bleu-only', help='', action='store_true')
    parser.add_argument('--use-auth-token', help='', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    assert opt.model_checkpoint or opt.hyp_test or opt.hyp_dev
    assert opt.prediction_aggregation in ['first', 'last', 'long', 'short', 'middle']
    assert opt.prediction_level in ['answer', 'sentence', 'context']
    metric = evaluate(
        export_dir=opt.export_dir,
        batch_size=opt.batch_size,
        n_beams=opt.n_beams,
        hypothesis_file_dev=opt.hyp_dev,
        hypothesis_file_test=opt.hyp_test,
        model=opt.model_checkpoint,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        dataset_path=opt.dataset_path,
        dataset_name=opt.dataset_name,
        input_type=opt.input_type,
        output_type=opt.output_type,
        prediction_aggregation=opt.prediction_aggregation,
        prediction_level=opt.prediction_level,
        overwrite=opt.overwrite,
        language=opt.language,
        bleu_only=opt.bleu_only,
        use_auth_token=opt.use_auth_token
    )
    print(json.dumps(metric, indent=4, sort_keys=True))

