# QA-based Evaluation
This is a guideline to run QA-based Evaluation (QAE), proposed in [Zhang and Bansal, 2019](https://aclanthology.org/D19-1253/).
QAE first generates a synthetic QA dataset over 1 million paragraph and answer pairs collected in [Du and Cardie, 2018](https://aclanthology.org/P18-1177/) with
target QG model. Then, a QA model is trained on the synthetic dataset, and the accuracy of the model on the test set (the validation set of SQuAD is commonly used)
becomes the QAE metric. Higher QAE means the QG model can produce questions, that is as valuable as human-made questions.


## Step1: Create Synthetic Dataset with Target QG Model
First, download `train.json` file from [https://drive.google.com/drive/folders/1E6Cg7c0XkWBOszMgHq_gHNRZVnAW2mEa](https://drive.google.com/drive/folders/1E6Cg7c0XkWBOszMgHq_gHNRZVnAW2mEa),
and locate it under `./data/`, then run
```shell
python get_synthetic_data.py -m [MODEL] -e [EXPORT_FILE] -b [BATCH]
```

To run it on our models,
```shell
python get_synthetic_data.py -m 'lmqg/lmqg-t5-large-squad' -e './data/t5_large.jsonl' -b 64
python get_synthetic_data.py -m 'lmqg/lmqg-t5-base-squad' -e './data/t5_base.jsonl' -b 64
python get_synthetic_data.py -m 'lmqg/lmqg-t5-small-squad' -e './data/t5_small.jsonl' -b 64
python get_synthetic_data.py -m 'lmqg/lmqg-bart-large-squad' -e './data/bart_large.jsonl' -b 64
python get_synthetic_data.py -m 'lmqg/lmqg-bart-base-squad' -e './data/bart_base.jsonl' -b 64
```

## Step2: Upload Synthetic Data to DatasetHub
Once the synthetic QA data is created, we upload it to the huggingface dataset hub. 
See [lmqg/qa_harvesting_from_wikipedia_pseudo](https://huggingface.co/datasets/lmqg/qa_harvesting_from_wikipedia_pseudo) for more detail.

## Step3: Fine-tune & Evaluate QA model 
QA model fine-tuning follows the [huggingface's example script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering).
We forked a few scripts that are needed to run QA training([run_qa.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py),
[trainer_qa.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py),
[utils_qa.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py)).
Only difference we made in the configuration is that we use `bert-base-cased` instead of `bert-base-uncased`.
To run QA model fine-tuning/evaluation on our datasets, 

```shell
python run_qa.py --model_name_or_path bert-base-cased --do_train --do_eval --per_device_train_batch_size 16 --learning_rate 1e-6 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 --dataset_name lmqg/qa_harvesting_from_wikipedia_pseudo --output_dir qa/bert_base_cased_bart_base --dataset_config_name bart_base
python run_qa.py --model_name_or_path bert-base-cased --do_train --do_eval --per_device_train_batch_size 16 --learning_rate 1e-6 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 --dataset_name lmqg/qa_harvesting_from_wikipedia_pseudo --output_dir qa/bert_base_cased_bart_large --dataset_config_name bart_large
python run_qa.py --model_name_or_path bert-base-cased --do_train --do_eval --per_device_train_batch_size 16 --learning_rate 1e-6 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 --dataset_name lmqg/qa_harvesting_from_wikipedia_pseudo --output_dir qa/bert_base_cased_t5_small --dataset_config_name t5_small
python run_qa.py --model_name_or_path bert-base-cased --do_train --do_eval --per_device_train_batch_size 16 --learning_rate 1e-6 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 --dataset_name lmqg/qa_harvesting_from_wikipedia_pseudo --output_dir qa/bert_base_cased_t5_base --dataset_config_name t5_base
python run_qa.py --model_name_or_path bert-base-cased --do_train --do_eval --per_device_train_batch_size 16 --learning_rate 1e-6 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 --dataset_name lmqg/qa_harvesting_from_wikipedia_pseudo --output_dir qa/bert_base_cased_t5_large --dataset_config_name t5_large
```