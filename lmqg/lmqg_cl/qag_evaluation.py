""" Evaluate Multitask QAG Model with QAG metric (eg. QA Aligned F1). """
import json
import logging
import argparse
import os
from itertools import chain
from statistics import mean

from datasets import load_dataset
from lmqg import TransformersQG
from lmqg.automatic_evaluation_tool import QAAlignedF1Score, Bleu, Meteor, Rouge, BERTScore, MoverScore
from lmqg.spacy_module import SpacyPipeline
from lmqg.automatic_evaluation import LANG_NEED_TOKENIZATION
import pandas as pd


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='QAG evaluation.')
    parser.add_argument('-m', '--model', default=None, type=str)
    parser.add_argument('--model-ae', default=None, type=str)
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
    parser.add_argument('--device-map', help='', default=None, type=str)
    parser.add_argument('--low-cpu-mem-usage', help='', action='store_true')
    parser.add_argument('-e', '--export-dir', required=True, type=str)
    parser.add_argument('--hyp-test', default=None, type=str)
    parser.add_argument('--hyp-dev', default=None, type=str)
    parser.add_argument('--overwrite-prediction', help='', action='store_true')
    parser.add_argument('--overwrite-metric', help='', action='store_true')
    parser.add_argument('--use-reference-answer', action='store_true')
    return parser


def evaluate_qag(model, model_ae, max_length, max_length_output, dataset_path, dataset_name, test_split, validation_split, n_beams, batch_size, language, use_auth_token, device_map, low_cpu_mem_usage, export_dir, hyp_test, hyp_dev, overwrite_prediction, overwrite_metric, use_reference_answer):
    os.makedirs(export_dir, exist_ok=True)
    metrics = [
        (QAAlignedF1Score(target_metric='f1', base_metric='bertscore', language=language),
         "QAAlignedF1Score (BERTScore)"),
        (QAAlignedF1Score(target_metric='recall', base_metric='bertscore', language=language),
         "QAAlignedRecall (BERTScore)"),
        (QAAlignedF1Score(target_metric='precision', base_metric='bertscore', language=language),
         "QAAlignedPrecision (BERTScore)"),
        (QAAlignedF1Score(target_metric='f1', base_metric='moverscore', language=language),
         "QAAlignedF1Score (MoverScore)"),
        (QAAlignedF1Score(target_metric='recall', base_metric='moverscore', language=language),
         "QAAlignedRecall (MoverScore)"),
        (QAAlignedF1Score(target_metric='precision', base_metric='moverscore', language=language),
         "QAAlignedPrecision (MoverScore)"),
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #(Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (BERTScore(language=language), "BERTScore"),
        (MoverScore(language=language), 'MoverScore')
    ]

    def load_model():
        if model is not None:

            _model = TransformersQG(model,
                                    # is_ae=None if is_ae else True,
                                    # is_qg=None if is_qg else True,
                                    is_qag=True,
                                    model_ae=model_ae,
                                    skip_overflow_error=True,
                                    drop_answer_error_text=True,
                                    language=language,
                                    max_length=max_length,
                                    max_length_output=max_length_output,
                                    use_auth_token=use_auth_token,
                                    device_map=device_map,
                                    low_cpu_mem_usage=low_cpu_mem_usage)
            _model.eval()
            return _model
        raise ValueError(f"require `-m` or `--model`")

    if model_ae is not None:
        metric_file = f"{export_dir}/metric.first.answer.paragraph.questions_answers." \
                      f"{dataset_path.replace('/', '_')}.{dataset_name}." \
                      f"{model_ae.replace('/', '_')}.json"
    else:
        metric_file = f"{export_dir}/metric.first.answer.paragraph.questions_answers." \
                      f"{dataset_path.replace('/', '_')}.{dataset_name}.json"
    if os.path.exists(metric_file):
        with open(metric_file) as f:
            output = json.load(f)
    else:
        output = {}
    spacy_model = SpacyPipeline(language=language) if language in LANG_NEED_TOKENIZATION else None
    for _split, _file in zip([test_split, validation_split], [hyp_test, hyp_dev]):
        if _file is None:
            if model_ae is not None:
                _file = f"{export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                        f"{dataset_path.replace('/', '_')}.{dataset_name}." \
                        f"{model_ae.replace('/', '_')}.txt"
            else:
                _file = f"{export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                        f"{dataset_path.replace('/', '_')}.{dataset_name}.txt"

        logging.info(f'generate qa for split {_split}')
        if _split not in output:
            output[_split] = {}

        # dataset = load_dataset(dataset_path, None if dataset_name == 'default' else dataset_name,
        #                        split=_split, use_auth_token=use_auth_token)
        # df = dataset.to_pandas()
        df = pd.read_csv(dataset_path).rename(columns={"context":"paragraph"})
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
        if not overwrite_prediction and os.path.exists(_file):
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
            if not use_reference_answer:
                logging.info("model prediction: (qag model)")
                prediction = model.generate_qa(
                    list_context=model_input,
                    num_beams=n_beams,
                    batch_size=batch_size)
            else:
                logging.info("model prediction: (qg model, answer fixed by reference)")
                model_input_flat = list(chain(*[[i] * len(h) for i, h in zip(model_input, model_highlight)]))
                model_highlight_flat = list(chain(*model_highlight))
                prediction_flat = model.generate_q(
                    list_context=model_input_flat,
                    list_answer=model_highlight_flat,
                    num_beams=n_beams,
                    batch_size=batch_size)
                _index = 0
                prediction = []
                for h in model_highlight:
                    questions = prediction_flat[_index:_index+len(h)]
                    answers = model_highlight_flat[_index:_index+len(h)]
                    prediction.append(list(zip(questions, answers)))
                    _index += len(h)

            # formatting prediction
            prediction = [' | '.join([f"question: {q}, answer: {a}" for q, a in p]) if p is not None else "" for p in prediction]
            assert len(prediction) == len(model_input), f"{len(prediction)} != {len(model_input)}"
            with open(_file, 'w') as f:
                f.write('\n'.join(prediction))

        for metric, metric_name in metrics:
            metric_name_list = [metric_name] if type(metric_name) is str else metric_name
            if overwrite_metric or any(m not in output[_split] for m in metric_name_list):
                if spacy_model is not None and (type(metric_name) is list or not metric_name.startswith("QAAligned")):
                    prediction = [' '.join(spacy_model.token(i)) for i in prediction]
                scores = metric.get_score(prediction, gold_reference)
                if type(metric_name) is list:
                    for _metric_name, _score in zip(metric_name, scores):
                        output[_split][_metric_name] = mean(_score.tolist())
                else:
                    output[_split][metric_name] = mean(scores.tolist())

    with open(metric_file, "w") as f:
        json.dump(output, f)
def run_evaluation(args_list=None):
    parser = create_parser()
    args = parser.parse_args(args_list)

    evaluate_qag(args.model, args.model_ae, args.max_length, args.max_length_output, args.dataset_path, args.dataset_name, args.test_split, args.validation_split, args.n_beams, args.batch_size, args.language, args.use_auth_token, args.device_map, args.low_cpu_mem_usage, args.export_dir, args.hyp_test, args.hyp_dev, args.overwrite_prediction, args.overwrite_metric, args.use_reference_answer)

# Example usage in a Jupyter Notebook
# args_list = ["-m", "lmqg/t5-large-squad-qag", "-e", "./weird_dir", "-d", "lmqg/qg_squad", "-l", "en"]
# run_evaluation(args_list)
