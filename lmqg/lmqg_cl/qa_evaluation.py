import argparse
import logging

from lmqg.qa_evaluation_tool import generate_qa_pairs, qa_trainer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


# def main_qa_model_training():
#     parser = argparse.ArgumentParser(description='QA model training.')
#     parser.add_argument('-m', '--model', default='lmqg/distilbert-base-uncased', type=str)
#     parser.add_argument('-d', '--dataset', default='lmqg/qa_squadshifts', type=str)
#     parser.add_argument('-n', '--dataset-name', default='new_wiki', type=str)
#     parser.add_argument('-b', '--batch-size', default=256, type=int)
#
#     parser.add_argument('--answer-model', default=None, type=str)
#     parser.add_argument('-a', '--answer-extraction', action='store_true')
#     parser.add_argument('--overwrite', action='store_true')
#     parser.add_argument('-b', '--batch-size', default=256, type=int)
#     parser.add_argument('-e', '--export-dir', default='tmp_output', type=str)
#     opt = parser.parse_args()
#
#     qa_trainer(
#         dataset=opt.dataset,
#         dataset_name=opt.dataset_name,
#         language_model=opt.model,
#         eval_step: int = 50,
#         random_seed: int = 42,
#         split_train: str = 'train',
#         split_validation: str = 'validation',
#         split_test: str = 'test',
#         parallel: bool = False,
#         n_trials: int = 10,
#         ray_result_dir: str = 'ray_result',
#         output_dir: str = 'qa_eval_output',
#         overwrite: bool = False
#             )


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
