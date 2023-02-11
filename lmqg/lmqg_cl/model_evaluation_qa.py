""" Evaluate Multitask QAG Model with QAG metric (eg. QA Aligned F1). """
import json
import logging
import argparse
import os

from evaluate import load
from lmqg import TransformersQG, get_dataset


squad_metric = load("squad")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='QA evaluation (F1 & Exact Match)')
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
    parser.add_argument('--device-map', help='', default=None, type=str)
    parser.add_argument('--low-cpu-mem-usage', help='', action='store_true')
    parser.add_argument('-e', '--export-dir', required=True, type=str)
    parser.add_argument('--hyp-test', default=None, type=str)
    parser.add_argument('--hyp-dev', default=None, type=str)
    parser.add_argument('--overwrite-prediction', help='', action='store_true')
    parser.add_argument('--overwrite-metric', help='', action='store_true')
    parser.add_argument('-i', '--input-type', help='', default='paragraph_question', type=str)
    parser.add_argument('-o', '--output-type', help='', default='answer', type=str)
    return parser.parse_args()


def main():

    opt = get_options()
    os.makedirs(opt.export_dir, exist_ok=True)

    def load_model():
        if opt.model_checkpoint is not None:
            _model = TransformersQG(opt.model_checkpoint,
                                    skip_overflow_error=True,
                                    drop_answer_error_text=True,
                                    language=opt.language,
                                    max_length=opt.max_length,
                                    max_length_output=opt.max_length_output,
                                    use_auth_token=opt.use_auth_token,
                                    device_map=opt.device_map,
                                    low_cpu_mem_usage=opt.low_cpu_mem_usage)
            _model.eval()
            return _model
        raise ValueError(f"require `-m` or `--model-checkpoint`")

    metric_file = f"{opt.export_dir}/metric.first.answer.{opt.input_type}.{opt.output_type}." \
                  f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}.json"
    if os.path.exists(metric_file):
        with open(metric_file) as f:
            output = json.load(f)
    else:
        output = {}

    for _split, _file in zip([opt.test_split, opt.validation_split], [opt.hyp_test, opt.hyp_dev]):
        _file = f"{opt.export_dir}/samples.{_split}.hyp.{opt.input_type}.{opt.output_type}." \
                f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}.txt" if _file is None else _file
        logging.info(f'generate qa for split {_split}')
        if _split not in output:
            output[_split] = {}

        data_in, gold_reference = get_dataset(
            opt.dataset_path,
            opt.dataset_name,
            input_type=opt.input_type,
            output_type=opt.output_type,
            split=_split,
            use_auth_token=opt.use_auth_token)
        references = [{"answers": {"answer_start": [100], "text": [r]}, "id": str(n)} for n, r in
                      enumerate(gold_reference)]
        prediction = None
        if not opt.overwrite_prediction and os.path.exists(_file):
            with open(_file) as f:
                _prediction = f.read().split('\n')
            if len(_prediction) != len(gold_reference):
                logging.warning(f"found prediction file at {_file} but length not match "
                                f"({len(_prediction)} != {len(gold_reference)})")
            else:
                prediction = _prediction
        if prediction is None:
            model = load_model()
            prediction = model.generate_prediction(
                inputs=data_in,
                prefix_type="qa" if model.add_prefix else None,
                num_beams=opt.n_beams,
                batch_size=opt.batch_size)
            # formatting prediction
            assert len(prediction) == len(data_in), f"{len(prediction)} != {len(data_in)}"
            with open(_file, 'w') as f:
                f.write('\n'.join(prediction))

        prediction = [{"prediction_text": p, "id": str(n)} for n, p in enumerate(prediction)]

        if opt.overwrite_metric or 'F1 (QA)' not in output[_split] or 'Exact Match (QA)' not in output[_split]:
            scores = squad_metric.compute(predictions=prediction, references=references)
            output[_split]['AnswerF1Score'] = scores['f1']
            output[_split]['AnswerExactMatch'] = scores['exact_match']

    with open(metric_file, "w") as f:
        json.dump(output, f)

