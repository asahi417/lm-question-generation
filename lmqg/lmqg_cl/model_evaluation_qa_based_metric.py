import argparse
import logging
import json
import os
import pandas as pd
from datasets import load_dataset
from lmqg import TransformersQG
from lmqg.qa_evaluation_tool import generate_qa_pairs, run_qa_evaluation

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main_qa_model_training():
    parser = argparse.ArgumentParser(description='QA model training.')
    parser.add_argument('-m', '--model', default='distilbert-base-uncased', type=str)
    parser.add_argument('-d', '--dataset', default='lmqg/qa_squadshifts', type=str)
    parser.add_argument('-n', '--dataset-name', default=None, type=str)
    parser.add_argument('--dataset-train', default=None, type=str)
    parser.add_argument('--dataset-validation', default=None, type=str)
    parser.add_argument('--dataset-test', default=None, type=str)
    parser.add_argument('--eval-step', default=50, type=int)
    parser.add_argument('--random-seed', default=42, type=int)
    parser.add_argument('--split-train', default='train', type=str)
    parser.add_argument('--split-validation', default='validation', type=str)
    parser.add_argument('--split-test', default='test', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-trials', default=10, type=int)
    parser.add_argument('--max-seq-length', default=384, type=int)
    parser.add_argument('--ray-result-dir', default='ray_results', type=str)
    parser.add_argument('--output-dir', default='qa_eval_output', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--down-sample-size-train', default=None, type=int)
    parser.add_argument('--down-sample-size-validation', default=None, type=int)
    parser.add_argument('--num-cpus', default=4, type=int)

    opt = parser.parse_args()
    if opt.dataset_train is not None and opt.dataset_validation is not None and opt.dataset_test is not None:
        dataset_files = {
            'train': opt.dataset_train,
            'validation': opt.dataset_validation,
            'test': opt.dataset_test
        }
    else:
        dataset_files = None

    run_qa_evaluation(
        dataset=opt.dataset,
        dataset_name=opt.dataset_name,
        dataset_files=dataset_files,
        language_model=opt.model,
        eval_step=opt.eval_step,
        random_seed=opt.random_seed,
        max_seq_length=opt.max_seq_length,
        split_train=opt.split_train,
        split_validation=opt.split_validation,
        split_test=opt.split_test,
        parallel=opt.parallel,
        n_trials=opt.n_trials,
        ray_result_dir=opt.ray_result_dir,
        output_dir=opt.output_dir,
        overwrite=opt.overwrite,
        skip_training=opt.skip_training,
        down_sample_size_train=opt.down_sample_size_train,
        down_sample_size_validation=opt.down_sample_size_validation,
        num_cpus=opt.num_cpus
    )


def main_generate_qa_pair():
    parser = argparse.ArgumentParser(description='Generate QA pseudo dataset.')
    parser.add_argument('-m', '--model-qg', default='lmqg/t5-small-squad-qg', type=str)
    parser.add_argument('--model-ae', default=None, type=str)
    parser.add_argument('-l', '--language', default='en', type=str)
    parser.add_argument('-d', '--anchor-data', default='lmqg/qa_squadshifts', type=str)
    parser.add_argument('-n', '--anchor-data-name', default='new_wiki', type=str)
    parser.add_argument('--use-reference-answer', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('-e', '--export-dir', default='tmp_output', type=str)
    parser.add_argument('--max-length', default=512, type=int, help='')
    parser.add_argument('--max-length-output', default=64, type=int, help='')
    parser.add_argument('--compute-ppl', action='store_true')
    opt = parser.parse_args()
    qa_pairs = generate_qa_pairs(
        model_qg=opt.model_qg,
        model_ae=opt.model_ae,
        language=opt.language,
        anchor_data=opt.anchor_data,
        anchor_data_name=opt.anchor_data_name,
        use_reference_answer=opt.use_reference_answer,
        batch_size=opt.batch_size,
        export_dir=opt.export_dir,
        overwrite=opt.overwrite,
        max_length=opt.max_length,
        max_length_output=opt.max_length)
    if opt.compute_ppl:
        logging.info(f"Perplexity: `{opt.model_qg}`, data: `{opt.anchor_data}`, data_name: `{opt.anchor_data_name}`")
        anchor_data = load_dataset(opt.anchor_data, opt.anchor_data_name)
        model = TransformersQG(opt.model_qg, max_length=opt.max_length, max_length_output=opt.max_length)
        if model.is_qag:
            target_outputs = ['questions_answers']
        else:
            assert model.is_qg
            target_outputs = ['answer', 'question'] if model.is_ae else ['question']

        for _split, dataset_split in qa_pairs.items():
            df = pd.DataFrame(dataset_split)
            for target in target_outputs:
                output_file = f"{opt.export_dir}/perplexity_{target}.{_split}.json"
                if os.path.exists(output_file) and not opt.overwrite:
                    continue
                ppl = model.get_perplexity(
                    list_question=[i['question'] for i in dataset_split],
                    list_context=[i['context'] for i in dataset_split],
                    list_answer=[i['answers']['text'][0] for i in dataset_split],
                    target_output=target,
                    batch_size=opt.batch_size
                )
                assert len(dataset_split) == len(ppl), f"{len(dataset_split)} != {len(ppl)}"
                with open(output_file, "w") as f:
                    json.dump({f"perplexity_{target}": ppl}, f)
                df['perplexity'] = ppl
                target_col = [i for i in df.columns if 'perplexity' not in i]
                filtered_1 = df.sort_values(by=f'perplexity_{target}').head(len(anchor_data))[target_col]
                filtered_1_d = list(filtered_1.T.to_dict().values())
                with open(f"{opt.export_dir}/{_split}.filtered.perplexity_{target}.jsonl", "w") as f:
                    f.write('\n'.join([json.dumps(i) for i in filtered_1_d]))
