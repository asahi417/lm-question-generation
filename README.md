[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/lmqg/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/lmqg.svg)](https://badge.fury.io/py/lmqg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lmqg.svg)](https://pypi.python.org/pypi/lmqg/)
[![PyPI status](https://img.shields.io/pypi/status/lmqg.svg)](https://pypi.python.org/pypi/lmqg/)

# Question and Answer Generation with Language Models

<p align="center">
  <img src="https://raw.githubusercontent.com/asahi417/lm-question-generation/master/assets/qag.png" width="400">
  <br><em> Figure 1: Three distinct QAG approaches. </em>
</p>

The `lmqg` is a python library for question and answer generation (QAG) with language models (LMs). Here, we consider 
paragraph-level QAG, where user will provide a context (paragraph or document), and the model will generate a list of 
question and answer pairs on the context. With `lmqg`, you can do following things:
- ***QAG & QG in One Line of Code:*** Given a paragraph, generate a list of question and answer pairs (or specify answer and generate question on it) in *8* languages (en/fr/ja/ko/ru/it/es/de).
- ***QAG & QG Model Training:*** On a dataset containing triples of *paragraph*, *question*, and *answer*, fine-tune LMs to achieve your own QAG or QG model.
- ***QAG & QG Model Evaluation:*** Evaluate QAG or QG model.
- ***QAG & QG Model Hosting:*** Host your QAG or QG model on a web application or a restAPI server.
 
All the models are available at [https://autoqg.net/](https://autoqg.net/), an online demo service to play around with QAG models.

***Update May 2023:*** Two papers got accepted by ACL 2023 ([QAG at finding](https://asahiushio.com/files/paper_2023_acl_qag.pdf), [LMQG at system demonstration](https://asahiushio.com/files/paper_2023_acl_demo.pdf)).

***Update Oct 2022:*** Our [QG paper](https://aclanthology.org/2022.emnlp-main.42/) got accepted by EMNLP main 2022.

### A bit more about QAG models 📝
Our QAG models can be grouped into three types: **Pipeline**, **Multitask**, 
and **End2end** (see Figure 1). The **Pipeline** consists of question generation (QG) and answer extraction (AE) models independently,
where AE will parse all the sentences in the context to extract answers, and QG will generate questions on the answers.
The **Multitask** follows same architecture as the **Pipeline**, but the QG and AE models are shared model fine-tuned jointly.
Finally, **End2end** model will generate a list of question and answer pairs in an end-to-end manner.
In practice, **Pipeline** and **Multitask** generate more question and answer pairs, while **End2end** generates less but a few times faster, 
and the quality of the generated question and answer pairs depend on language.
All types are available in the *8* diverse languages (en/fr/ja/ko/ru/it/es/de) via `lmqg`, and the models are all shared on HuggingFace (link to the model card).
To know more about QAG, please check [our ACL 2023 paper](https://asahiushio.com/files/paper_2023_acl_qag.pdf) that describes the QAG models and reports a complete performance comparison of each QAG models in every language.

<p align="left">
  <img src="https://raw.githubusercontent.com/asahi417/lm-question-generation/master/assets/example.png" width="700">
  <em> Figure 2: An example of QAG (a) and QG (b). </em>
</p>

### Is QAG different from Question Generation (QG)? 🤔 
All the functionalities support question generation as well. Our QG model assumes user to specify an answer in addition to a context,
and the QG model will generate a question that is answerable by the answer given the context (see Figure 2).
To know more about QG, please check [our EMNLP 2022 paper](https://aclanthology.org/2022.emnlp-main.42/) that describes the QG models more in detail.


## Get Started 🚀
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13izkdp2l7G2oeh_fwL7xJdR_67HMK_hQ?usp=sharing)

Let's install `lmqg` via pip first.
```shell
pip install lmqg
```

## Generate Question & Answer

- ***QAG with End2end or Multitask Models:*** The end2end QAG models are fine-tuned to generate a list of QA pairs with a single inference, 
  so it is the fastest class among our QAG models. Meanwhile, multitask QAG models breakdown the QAG task into QG and AE, where they 
  parse each sentence to get the answer with AE, and generate question on each answer with QG. Multitask QAG potentially generate more QA pairs than end2end QAG, but 
  inference takes a few times more than end2end models. Both models can be used as following 
  
```python
from pprint import pprint
from lmqg import TransformersQG

# initialize model
model = TransformersQG('lmqg/t5-base-squad-qag') # or TransformersQG(model='lmqg/t5-base-squad-qg-ae') 
# paragraph to generate pairs of question and answer
context = "William Turner was an English painter who specialised in watercolour landscapes. He is often known as William Turner of Oxford or just Turner of Oxford to distinguish him from his contemporary, J. M. W. Turner. Many of Turner's paintings depicted the countryside around Oxford. One of his best known pictures is a view of the city of Oxford from Hinksey Hill."
# model prediction
question_answer = model.generate_qa(context)
# the output is a list of tuple (question, answer)
pprint(question_answer)
[
    ('Who was an English painter who specialised in watercolour landscapes?', 'William Turner'),
    ('What is William Turner often known as?', 'William Turner of Oxford or just Turner of Oxford'),
    ("What did many of Turner's paintings depict?", 'the countryside around Oxford'),
    ("What is one of Turner's best known pictures?", 'a view of the city of Oxford from Hinksey Hill')
]
```

- ***QAG with Pipeline Models:*** The pipeline QAG is similar to multitask QAG, but the QG and AE models are independently fine-tuned, unlike 
  multitask QAG that fine-tunes QG and AE jointly. Pipeline QAG can improve the performance in some cases, but it is as heavy as multitask QAG with 
  more storage consuming due to the two models loaded. The pipeline QAG can be used as following. The `model` and `model_ae` are the QG and AE models respectively.
  
```python
from pprint import pprint
from lmqg import TransformersQG

# initialize model
model = TransformersQG(model='lmqg/t5-base-squad-qg', model_ae='lmqg/t5-base-squad-ae')  
# paragraph to generate pairs of question and answer
context = "William Turner was an English painter who specialised in watercolour landscapes. He is often known as William Turner of Oxford or just Turner of Oxford to distinguish him from his contemporary, J. M. W. Turner. Many of Turner's paintings depicted the countryside around Oxford. One of his best known pictures is a view of the city of Oxford from Hinksey Hill."
# model prediction
question_answer = model.generate_qa(context)
# the output is a list of tuple (question, answer)
pprint(question_answer)
[
    ('Who was an English painter who specialised in watercolour landscapes?', 'William Turner'),
    ('What is another name for William Turner?', 'William Turner of Oxford'),
    ("What did many of William Turner's paintings depict around Oxford?", 'the countryside'),
    ('From what hill is a view of the city of Oxford taken?', 'Hinksey Hill.')
]
```

- ***QG only:*** The QG model can be used as following. The `model` is the QG model.
  
```python
from pprint import pprint
from lmqg import TransformersQG

# initialize model
model = TransformersQG(model='lmqg/t5-base-squad-qg')

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
pprint(question)
[
  'Who was an English painter who specialised in watercolour landscapes?',
  'What nationality was William Turner?'
]
``` 

- ***AE only:*** The QG model can be used as following. The `model` is the QG model.

```python
from pprint import pprint
from lmqg import TransformersQG

# initialize model
model = TransformersQG(model='lmqg/t5-base-squad-ae')
# model prediction
answer = model.generate_a("William Turner was an English painter who specialised in watercolour landscapes")
pprint(answer)
['William Turner']
```

## Rest API with huggingface inference API

We provide a rest API which hosts the model inference through huggingface inference API. You need huggingface API token to run your own API and install dependencies as below.
```shell
pip install lmqg[api]
```
Swagger UI is available at [`http://127.0.0.1:8088/docs`](http://127.0.0.1:8088/docs), when you run the app locally (replace the address by your server address).

### Build
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

### API Description
You must pass the huggingface api token via the environmental variable `API_TOKEN`.
The main endpoint is `question_generation`, which has following parameters,

| Parameter        | Description                                                                         |
| ---------------- | ----------------------------------------------------------------------------------- |
| **input_text**   | input text, a paragraph or a sentence to generate question |
| **language**     | language |
| **qg_model**     | question generation model |
| **answer_model** | answer extraction model |

and return a list of dictionaries with `question` and `answer`. 
```shell
{
  "qa": [
    {"question": "Who founded Nintendo Karuta?", "answer": "Fusajiro Yamauchi"},
    {"question": "When did Nintendo distribute its first video game console, the Color TV-Game?", "answer": "1977"}
  ]
}
```

## Misc
To reproduce the models used in our papers, please see the links.

- [***TBA, ACL 2022 Finding***](https://asahiushio.com/files/paper_2023_acl_qag.pdf): [https://github.com/asahi417/lm-question-generation/tree/master/misc/2023_acl_qag](https://github.com/asahi417/lm-question-generation/tree/master/misc/2023_acl_qag)
- [***A Practical Toolkit for Multilingual Question and Answer Generation, ACL 2022, System Demonstration***](https://asahiushio.com/files/paper_2023_acl_demo.pdf): [https://github.com/asahi417/lm-question-generation/tree/master/misc/2023_acl_qag](https://github.com/asahi417/lm-question-generation/tree/master/misc/2023_acl_qag)
- [***Generative Language Models for Paragraph-Level Question Generation, EMNLP 2022 Main***](https://aclanthology.org/2022.emnlp-main.42/):  [https://github.com/asahi417/lm-question-generation/tree/master/misc/2022_emnlp_qg](https://github.com/asahi417/lm-question-generation/tree/master/misc/2022_emnlp_qg)

## Citation
Please cite following paper if you use any resource.

- [***Generative Language Models for Paragraph-Level Question Generation, EMNLP 2022 Main***](https://aclanthology.org/2022.emnlp-main.42/): The QG models.
```
@inproceedings{ushio-etal-2022-generative,
    title = "{G}enerative {L}anguage {M}odels for {P}aragraph-{L}evel {Q}uestion {G}eneration",
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

- [***TBA, ACL 2022 Finding***](tba): The QAG models.
```
@inproceedings{ushio-etal-2023-tba,
    title = "TBA",
    author = "Ushio, Asahi  and
        Alva-Manchego, Fernando  and
        Camacho-Collados, Jose",
    booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics",
    month = Jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
}
```

- [***A Practical Toolkit for Multilingual Question and Answer Generation, ACL 2022, System Demonstration***](tba): The library and demo.

```
@inproceedings{ushio-etal-2023-a-practical-toolkit
    title = "A Practical Toolkit for Multilingual Question and Answer Generation, ACL 2022, System Demonstration",
    author = "Ushio, Asahi  and
        Alva-Manchego, Fernando  and
        Camacho-Collados, Jose",
    booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = Jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
}
```
