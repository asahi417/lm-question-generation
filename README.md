[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/lmqg/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/lmqg.svg)](https://badge.fury.io/py/lmqg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lmqg.svg)](https://pypi.python.org/pypi/lmqg/)
[![PyPI status](https://img.shields.io/pypi/status/lmqg.svg)](https://pypi.python.org/pypi/lmqg/)

# Generative Language Models for Paragraph-Level Question Generation

<p align="center">
  <img src="https://raw.githubusercontent.com/asahi417/lm-question-generation/master/assets/qg_diagram.png" width="500">
</p>

This is the official repository of the paper
"Generative Language Models for Paragraph-Level Question Generation:
A Unified Benchmark and Evaluation, EMNLP 2022 main conference".
This repository includes following contents:
- ***QG-Bench***, the first ever multilingual/multidomain QG benchmark.
- ***Multilingual/multidomain QG models*** fine-tuned on QG-Bench.
- A python library ***`lmqg`*** developed for question generation in python as well as QG model fine-tuning/evaluation.
- ***AutoQG***, a web application hosting QG models where user can test the model output interactively. 

### Table of Contents  
1. **[QG-Bench: multilingual & multidomain QG datasets (+ fine-tuned models)](https://github.com/asahi417/lm-question-generation/blob/master/QG_BENCH.md)**
2. **[LMQG: python library to fine-tune/evaluate QG model](#lmqg-language-model-for-question-generation-)**
3. **[AutoQG: web application hosting multilingual QG models](#autoqg)**
4. **[RestAPI: run model prediction via restAPI](#rest-api-with-huggingface-inference-api)**

Please cite following paper if you use any resource:
```
@inproceedings{ushio-etal-2022-generative,
    title = "{G}enerative {L}anguage {M}odels for {P}aragraph-{L}evel {Q}uestion {G}eneration: {A} {U}nified {B}enchmark and {E}valuation",
    author = "Ushio, Asahi  and
        Alva-Manchego, Fernando  and
        Camacho-Collados, Jose",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, U.A.E.",
    publisher = "Association for Computational Linguistics",
}
```

## LMQG: Language Model for Question Generation 🚀
The `lmqg` is a python library to fine-tune seq2seq language models ([T5](https://arxiv.org/pdf/1910.10683.pdf), [BART](https://arxiv.org/pdf/1910.13461.pdf)) 
on the question generation task and provide an API to host the model prediction via [huggingface](https://huggingface.co/).
Let's install `lmqg` via pip first.
```shell
pip install lmqg
```

### Generate Question & Answer

- ***Generate Question on Answers:*** This is a basic usecase of our QG models, where user provides a paragraph and an answer to generate a question that is answerable by the answer given the paragraph.
See [MODEL CARD](https://github.com/asahi417/lm-question-generation/blob/master/QG_BENCH.md#qg-models) for the available models.

```python
from lmqg import TransformersQG
# initialize model
model = TransformersQG(language='en', model='lmqg/t5-large-squad-multitask')
# a list of paragraph
context = [
    "William Turner was an English painter who specialised in watercolour landscapes",
    "William Turner was an English painter who specialised in watercolour landscapes"
]
# a list of answer (same size as the context)
answer = [
    "William Turner",
    "English"
]
# model prediction
question = model.generate_q(list_context=context, list_answer=answer)
print(question)
[
    'Who was an English painter who specialised in watercolour landscapes?',
    'What nationality was William Turner?'
]
```

- ***Generate Question & Answer Pairs:*** Instead of specifying an answer, user can let QG model to generate an answer on the paragraph, and generate question on it sequentially.
This functionality is only available for the QG models fine-tuned with answer extraction see [MODEL CARD](https://github.com/asahi417/lm-question-generation/blob/master/QG_BENCH.md#models-with-answer-extraction) for the full list of models with answer extraction (model alias usually has a suffix of `-multitask`).
  
```python
from lmqg import TransformersQG
# initialize model
model = TransformersQG(language='en', model='lmqg/t5-large-squad-multitask')
# paragraph to generate pairs of question and answer
context = "William Turner was an English painter who specialised in watercolour landscapes. He is often known as William Turner of Oxford or just Turner of Oxford to distinguish him from his contemporary, J. M. W. Turner. Many of Turner's paintings depicted the countryside around Oxford. One of his best known pictures is a view of the city of Oxford from Hinksey Hill."
# model prediction
question_answer = model.generate_qa(context)
# the output is a list of tuple (question, answer)
print(question_answer)
[
    ('Who was an English painter who specialised in watercolour landscapes?', 'William Turner'),
    ("What was William Turner's nickname?", 'William Turner of Oxford'),
    ("What did many of Turner's paintings depict around Oxford?", 'countryside'),
    ("What is one of William Turner's best known paintings?", 'a view of the city of Oxford')
]
```


### Model Evaluation
The evaluation tool reports `BLEU4`, `ROUGE-L`, `METEOR`, `BERTScore`, and `MoverScore` following [QG-Bench](https://github.com/asahi417/lm-question-generation/blob/master/QG_BENCH.md).
From command line, run following command 
```shell
lmqg-eval -m "lmqg/t5-large-squad" -e "./eval_metrics" -d "lmqg/qg_squad" -l "en"
```
where `-m` is a model alias on huggingface or path to local checkpoint, `-e` is the directly to export the metric file, `-d` is the dataset to evaluate, and `-l` is the language of the test set.
Instead of running model prediction, you can provide a prediction file instead to avoid computing it each time.
```shell
lmqg-eval --hyp-test '{your prediction file}' -e "./eval_metrics" -d "lmqg/qg_squad" -l "en"
```
The prediction file should be a text file of model generation in each line in the order of `test` split in the target dataset
([sample](https://huggingface.co/lmqg/t5-large-squad/raw/main/eval/samples.validation.hyp.paragraph_sentence.question.lmqg_qg_squad.default.txt)).
Check `lmqg-eval -h` to display all the options.

### Model Training
<p align="center">
  <img src="https://raw.githubusercontent.com/asahi417/lm-question-generation/master/assets/grid_search.png" width="650">
</p>

To fine-tune QG model, we employ a two-stage hyper-parameter optimization, described as above diagram.
Following command is to run the fine-tuning with parameter optimization.
```shell
lmqg-train-search -c "tmp_ckpt" -d "lmqg/qg_squad" -m "t5-small" -b 64 --epoch-partial 5 -e 15 --language "en" --n-max-config 1 \
  -g 2 4 --lr 1e-04 5e-04 1e-03 --label-smoothing 0 0.15
```
Check `lmqg-train-search -h` to display all the options.

Fine-tuning models in python follows below.  
```python
from lmqg import GridSearcher
trainer = GridSearcher(
    checkpoint_dir='tmp_ckpt',
    dataset_path='lmqg/qg_squad',
    model='t5-small',
    epoch=15,
    epoch_partial=5,
    batch=64,
    n_max_config=5,
    gradient_accumulation_steps=[2, 4], 
    lr=[1e-04, 5e-04, 1e-03],
    label_smoothing=[0, 0.15]
)
trainer.run()
```


## AutoQG

<p align="center">
  <img src="https://raw.githubusercontent.com/asahi417/lm-question-generation/master/assets/autoqg.gif" width="500">
</p>

***AutoQG ([https://autoqg.net](https://autoqg.net/))*** is a free web application hosting our QG models.
The QG models are listed at the [QG-Bench page](https://github.com/asahi417/lm-question-generation/blob/master/QG_BENCH.md).

## Rest API with huggingface inference API
<p align="center">
  <img src="https://raw.githubusercontent.com/asahi417/lm-question-generation/master/assets/api.png" width="600">
</p>

We provide a rest API which hosts the model inference through huggingface inference API. You need huggingface API token to run your own API and install dependencies as below.
```shell
pip install lmqg[api]
```
Swagger UI is available at [`http://127.0.0.1:8080/docs`](http://127.0.0.1:8080/docs), when you run the app locally (replace the address by your server address).

- Build/Run Local (command line):
```shell
export API_TOKEN={Your Huggingface API Token}
uvicorn app:app --reload --port 8088
uvicorn app:app --host 0.0.0.0 --port 8088
```

- Build/Run Local (docker):
```shell
docker build -t lmqg/app:latest . --build-arg api_token={Your Huggingface API Token}
docker run -p 8080:8080 lmqg/app:latest
```


## Misc
Following link is useful if you need to reproduce the results in our paper.
- [Model Fine-tuning/Evaluation](https://github.com/asahi417/lm-question-generation/tree/master/misc/qg_model_training)
- [QA based Evaluation](https://github.com/asahi417/lm-question-generation/tree/master/misc/qa_based_evaluation)
- [NQG model baseline](https://github.com/asahi417/lm-question-generation/tree/master/misc/nqg_baseline)