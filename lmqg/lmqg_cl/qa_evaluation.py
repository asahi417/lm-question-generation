import argparse
import logging

from lmqg.qa_evaluation_tool import generate_qa_pairs, run_qa_evaluation

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main_qa_model_training():
    parser = argparse.ArgumentParser(description='QA model training.')
    parser.add_argument('-m', '--model', default='distilbert-base-uncased', type=str)
    parser.add_argument('-d', '--dataset', default='lmqg/qa_squadshifts', type=str)
    parser.add_argument('-n', '--dataset-name', default='new_wiki', type=str)
    parser.add_argument('--eval-step', default=50, type=int)
    parser.add_argument('--random-seed', default=42, type=int)
    parser.add_argument('--split-train', default='train', type=str)
    parser.add_argument('--split-validation', default='validation', type=str)
    parser.add_argument('--split-test', default='test', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-trials', default=10, type=int)
    parser.add_argument('--ray-result-dir', default='ray_result', type=str)
    parser.add_argument('--output-dir', default='qa_eval_output', type=str)
    parser.add_argument('--overwrite', action='store_true')
    opt = parser.parse_args()

    run_qa_evaluation(
        dataset=opt.dataset,
        dataset_name=opt.dataset_name,
        language_model=opt.model,
        eval_step=opt.eval_step,
        random_seed=opt.random_seed,
        split_train=opt.split_train,
        split_validation=opt.split_validation,
        split_test=opt.split_test,
        parallel=opt.parallel,
        n_trials=opt.n_trials,
        ray_result_dir=opt.ray_result_dir,
        output_dir=opt.output_dir,
        overwrite=opt.overwrite)


def main_generate_qa_pair():
    parser = argparse.ArgumentParser(description='Generate QA pseudo dataset.')
    parser.add_argument('-m', '--qg-model', default='lmqg/t5-small-squad-multitask', type=str)
    parser.add_argument('-l', '--language', default='en', type=str)
    parser.add_argument('-d', '--anchor-data', default='lmqg/qa_squadshifts', type=str)
    parser.add_argument('-n', '--anchor-data-name', default='new_wiki', type=str)
    parser.add_argument('--answer-model', default=None, type=str)
    parser.add_argument('-a', '--answer-extraction', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('-e', '--export-dir', default='tmp_output', type=str)
    opt = parser.parse_args()
    generate_qa_pairs(
        qg_model=opt.qg_model,
        language=opt.language,
        anchor_data=opt.anchor_data,
        anchor_data_name=opt.anchor_data_name,
        answer_extraction=opt.answer_extraction,
        answer_model=opt.answer_model,
        batch_size=opt.batch_size,
        export_dir=opt.export_dir,
        overwrite=opt.overwrite
    )
