import argparse
import logging
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
    parser.add_argument('--use-auth-token', help='', action='store_true')
    parser.add_argument('--device-map', help='', default=None, type=str)
    parser.add_argument('--low-cpu-mem-usage', help='', action='store_true')
    opt = parser.parse_args()
    generate_qa_pairs(
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
        max_length_output=opt.max_length,
        use_auth_token=opt.use_auth_token,
        device_map=opt.device_map,
        low_cpu_mem_usage=opt.low_cpu_mem_usage)
