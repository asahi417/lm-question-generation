"""https://huggingface.co/docs/transformers/tasks/question_answering"""
import logging
import multiprocessing
import os
import json
import urllib
from os.path import join as pj
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer, EvalPrediction
from ray import tune


def internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


local_files_only = not internet_connection()


def qa_trainer(dataset,
               dataset_name: str = None,
               language_model: str = "distilbert-base-uncased",
               eval_step: int = 50,
               random_seed: int = 42,
               split_train: str = 'train',
               split_validation: str = 'validation',
               split_test: str = 'test',
               parallel: bool = False,
               n_trials: int = 10,
               ray_result_dir: str = 'ray_result',
               output_dir: str = 'qa_eval_output',
               overwrite: bool = False):

    def preprocess_function(examples):
        tokenizer = AutoTokenizer.from_pretrained(language_model, local_files_only=local_files_only)
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    os.environ["WANDB_DISABLED"] = "true"

    print(local_files_only)
    metric = evaluate.load("squad", local_files_only=local_files_only)

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    if dataset_name is not None:
        data = load_dataset(dataset, dataset_name)
    else:
        data = load_dataset(dataset)
    tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)
    best_model_path = pj(output_dir, 'best_model')
    if overwrite or not os.path.exists(best_model_path):
        logging.info(f'QA model training with {language_model}')

        model = AutoModelForQuestionAnswering.from_pretrained(language_model, local_files_only=local_files_only)

        training_args = TrainingArguments(
            report_to=None,
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=eval_step,
            seed=random_seed
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data[split_train],
            eval_dataset=tokenized_data[split_validation],
            compute_metrics=compute_metrics,
            model_init=lambda x: AutoModelForQuestionAnswering.from_pretrained(language_model, return_dict=True)
        )
        if parallel:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=ray_result_dir, direction="maximize", backend="ray", n_trials=n_trials,
                resources_per_trial={'cpu': multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()},

            )
        else:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=ray_result_dir, direction="maximize", backend="ray", n_trials=n_trials
            )
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        trainer.train()
        trainer.save_model(best_model_path)

    logging.info(f'QA model evaluation of {best_model_path}')

    model = AutoModelForQuestionAnswering.from_pretrained(best_model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(report_to=None, output_dir=output_dir, evaluation_strategy="no", seed=random_seed),
        eval_dataset=tokenized_data[split_test],
        compute_metrics=compute_metrics
    )
    summary_file = pj(output_dir, 'test_result.json')
    result = {k: v for k, v in trainer.evaluate().items()}
    logging.info(json.dumps(result, indent=4))
    with open(summary_file, 'w') as f:
        json.dump(result, f)

