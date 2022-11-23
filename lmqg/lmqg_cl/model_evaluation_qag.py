""" Evaluate Multitask QAG Model with QAG metric (eg. QA Aligned F1). """
import json
import logging
import argparse
import os
from itertools import chain
from statistics import mean

from datasets import load_dataset
from lmqg import TransformersQG
from lmqg.automatic_evaluation_tool import QAAlignedF1Score


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='QG evaluation on SQuAD.')
    parser.add_argument('-m', '--model-checkpoint', default=None, type=str)
    parser.add_argument('--max-length', default=512, type=int, help='')
    parser.add_argument('--max-length-output', default=256, type=int, help='')
    parser.add_argument('-d', '--dataset-path', help='huggingface datasets alias', default='lmqg/qg_squad', type=str)
    parser.add_argument('--dataset-name', help='huggingface datasets name', default='default', type=str)
    parser.add_argument('--test-split', help='the name of test split', default='test', type=str)
    parser.add_argument('--validation-split', help='the name of validation split', default='validation', type=str)
    parser.add_argument('--n-beams', default=4, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('-l', '--language', help='', default='en', type=str)
    parser.add_argument('--use-auth-token', help='', action='store_true')
    parser.add_argument('-e', '--export-dir', required=True, type=str)
    parser.add_argument('--hyp-test', default=None, type=str)
    parser.add_argument('--hyp-dev', default=None, type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    os.makedirs(opt.export_dir, exist_ok=True)
    metrics = [
        (QAAlignedF1Score(base_metric='bertscore', language=opt.language), "QAAlignedF1Score (BERTScore)"),
        (QAAlignedF1Score(base_metric='moverscore', language=opt.language), "QAAlignedF1Score (MoverScore)")
    ]

    def load_model():
        if opt.model_checkpoint is not None:
            _model = TransformersQG(opt.model_checkpoint,
                                   skip_overflow_error=True,
                                   drop_answer_error_text=True,
                                   language=opt.language,
                                   max_length=opt.max_length,
                                   max_length_output=opt.max_length_output,
                                   use_auth_token=opt.use_auth_token)
            _model.eval()
            return _model
        raise ValueError(f"require `-m` or `--model-checkpoint`")

    metric_file = f"{opt.export_dir}/metric.first.answer.paragraph.questions_answers." \
                  f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}.json"
    if os.path.exists(metric_file):
        with open(metric_file) as f:
            output = json.load(f)
    else:
        output = {}

    for _split, _file in zip([opt.test_split, opt.validation_split], [opt.hyp_test, opt.hyp_dev]):
        _file = f"{opt.export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}.txt" if _file is None else _file
        logging.info(f'generate qa for split {_split}')
        if _split not in output:
            output[_split] = {}

        if opt.dataset_name == 'default' or opt.dataset_name is None:
            dataset = load_dataset(opt.dataset_path, split=_split, use_auth_token=opt.use_auth_token)
        else:
            dataset = load_dataset(opt.dataset_path, opt.dataset_name, split=_split, use_auth_token=opt.use_auth_token)
        df = dataset.to_pandas()

        # formatting data into qag format
        model_input = []
        gold_reference = []
        model_highlight = []
        for paragraph, g in df.groupby("paragraph"):
            model_input.append(paragraph)
            model_highlight.append(g['answer'].tolist())
            gold_reference.append(' | '.join([
                f"question: {i['question']}, answer: {i['answer']}" for _, i in g.iterrows()
            ]))
        prediction = None
        if not opt.overwrite and os.path.exists(_file):
            with open(_file) as f:
                _prediction = f.read().split('\n')
            if len(_prediction) != len(gold_reference):
                logging.warning(f"found prediction file at {_file} but length not match "
                                f"({len(_prediction)} != {len(gold_reference)})")
            else:
                prediction = _prediction
        if prediction is None:
            model = load_model()
            # model prediction
            if model.multitask_model:
                logging.info("model prediction: (multitask model)")
                prediction = model.generate_qa(
                    list_context=model_input,
                    num_beams=opt.n_beams,
                    batch_size=opt.batch_size)
            else:
                logging.info("model prediction: (qg model, answer fixed by reference)")
                model_input_flat = list(chain(*[[i] * len(h) for i, h in zip(model_input, model_highlight)]))
                model_highlight_flat = list(chain(*model_highlight))
                prediction_flat = model.generate_q(
                    list_context=model_input_flat,
                    list_answer=model_highlight_flat,
                    num_beams=opt.n_beams,
                    batch_size=opt.batch_size)
                _index = 0
                prediction = []
                for h in model_highlight:
                    prediction.append(prediction_flat[_index:_index+len(h)])
                    _index += len(h)

            # formatting prediction
            prediction = [' | '.join([f"question: {q}, answer: {a}" for q, a in p]) if p is not None else "" for p in prediction]
            assert len(prediction) == len(model_input), f"{len(prediction)} != {len(model_input)}"
            with open(_file, 'w') as f:
                f.write('\n'.join(prediction))

        for metric, metric_name in metrics:
            if opt.overwrite or metric_name not in output[_split]:
                scores = metric.get_score(prediction, gold_reference)
                output[_split][metric_name] = mean([i['f1'] for i in scores])

    with open(metric_file, "w") as f:
        json.dump(output, f)

