#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import urllib
import json
import multiprocessing
import torch
from typing import Dict
from os.path import join as pj
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from ray import tune

from .trainer_qa import QuestionAnsweringTrainer
from .utils_qa import postprocess_qa_predictions

os.environ["WANDB_DISABLED"] = "true"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")


def internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


def run_qa_evaluation(dataset: str,
                      dataset_name: str = None,
                      dataset_files: Dict = None,
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
                      overwrite: bool = False,
                      skip_training: bool = False,
                      doc_stride: int = 128,
                      max_seq_length: int = 384,
                      max_answer_length: int = 30,
                      question_column_name: str = "question",
                      context_column_name: str = "context",
                      answer_column_name: str = "answers"):
    summary_file = pj(output_dir, 'test_result.json')
    if os.path.exists(summary_file) and not overwrite:
        return

    os.makedirs(output_dir, exist_ok=True)
    # Set seed before initializing model.
    set_seed(random_seed)
    local_file_only = not internet_connection()
    if dataset_files is None:
        raw_datasets = load_dataset(dataset) if dataset_name is None else load_dataset(dataset, dataset_name)
    else:
        raw_datasets = load_dataset(dataset, data_files=dataset_files)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(language_model, local_files_only=local_file_only)
    tokenizer = AutoTokenizer.from_pretrained(language_model, use_fast=True, local_files_only=local_file_only)
    model = AutoModelForQuestionAnswering.from_pretrained(language_model, config=config, local_files_only=local_file_only)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    if max_seq_length > tokenizer.model_max_length:
        logging.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Create train feature from dataset
    train_example = raw_datasets[split_train]
    # train_example = raw_datasets[split_train].select(list(range(50)))
    train_dataset = train_example.map(
        prepare_train_features, batched=True, num_proc=None,
        remove_columns=train_example.column_names, desc="Running tokenizer on train dataset"
    )
    # Validation Feature Creation
    validation_example = raw_datasets[split_validation]
    # validation_example = raw_datasets[split_validation].select(list(range(50)))
    validation_dataset = validation_example.map(
        prepare_validation_features, batched=True, num_proc=None,
        remove_columns=validation_example.column_names, desc="Running tokenizer on validation dataset",
    )
    # Predict Feature Creation
    if split_test in raw_datasets:
        test_example = raw_datasets[split_test]
        test_dataset = test_example.map(
            prepare_validation_features, batched=True, num_proc=None,
            remove_columns=test_example.column_names, desc="Running tokenizer on prediction dataset"
        )
    else:
        test_example = test_dataset = None

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=output_dir,
            prefix=stage
        )

        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": _v} for k, _v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    best_model_path = pj(output_dir, 'best_model')
    if not skip_training:
        # Initialize our Trainer
        training_args = TrainingArguments(
            report_to=None,
            output_dir=output_dir,
            eval_steps=eval_step,
            seed=random_seed,
            evaluation_strategy="steps"
        )
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            eval_examples=validation_example,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
            model_init=lambda x: AutoModelForQuestionAnswering.from_pretrained(language_model, return_dict=True, local_files_only=local_file_only)
        )

        logging.info("*** Finetuning ***")
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

    # Evaluation
    if test_example is not None and test_dataset is not None:
        logging.info("*** Evaluate ***")
        trainer = QuestionAnsweringTrainer(
            model=AutoModelForQuestionAnswering.from_pretrained(best_model_path, local_files_only=local_file_only),
            args=TrainingArguments(report_to=None, output_dir=output_dir, seed=random_seed, evaluation_strategy="no"),
            eval_dataset=test_dataset,
            eval_examples=test_example,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )
        metric = trainer.evaluate()
        result = {k: v for k, v in metric.items()}
        logging.info(json.dumps(result, indent=4))
        with open(summary_file, 'w') as f:
            json.dump(result, f)
