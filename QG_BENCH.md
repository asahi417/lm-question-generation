# QG-Bench and Fine-tuned Models
QG-Bench consists of question generation datasets in 8 different languages and 11 diverse domains.
The dataset is proposed in ["Generative Language Models for Paragraph-Level Question Generation, EMNLP 2022 main conference"](https://arxiv.org/abs/2210.03992),
and all the datasets are shared on huggingface via the [link below](#datasets).
To use the dataset, first install `datasets` library (`pip install datasets`) and load the dataset.

```python
from datasets import load_dataset
dataset = load_dataset("lmqg/qg_squad")
```

An example of the dataset instance looks as follows.
```
{
  "question": "What is heresy mainly at odds with?",
  "paragraph": "Heresy is any provocative belief or theory that is strongly at variance with established beliefs or customs. A heretic is a proponent of such claims or beliefs. Heresy is distinct from both apostasy, which is the explicit renunciation of one's religion, principles or cause, and blasphemy, which is an impious utterance or action concerning God or sacred things.",
  "answer": "established beliefs or customs",
  "sentence": "Heresy is any provocative belief or theory that is strongly at variance with established beliefs or customs .",
  "paragraph_sentence": "<hl> Heresy is any provocative belief or theory that is strongly at variance with established beliefs or customs . <hl> A heretic is a proponent of such claims or beliefs. Heresy is distinct from both apostasy, which is the explicit renunciation of one's religion, principles or cause, and blasphemy, which is an impious utterance or action concerning God or sacred things.",
  "paragraph_answer": "Heresy is any provocative belief or theory that is strongly at variance with <hl> established beliefs or customs <hl>. A heretic is a proponent of such claims or beliefs. Heresy is distinct from both apostasy, which is the explicit renunciation of one's religion, principles or cause, and blasphemy, which is an impious utterance or action concerning God or sacred things.",
  "sentence_answer": "Heresy is any provocative belief or theory that is strongly at variance with <hl> established beliefs or customs <hl> ."
}
```
Each feature contains following information:
- `question`: a `string` feature. 
- `paragraph`: a `string` feature.
- `answer`: a `string` feature.
- `sentence`: a `string` feature.
- `paragraph_answer`: a `string` feature, which is same as the paragraph but the answer is highlighted by a special token `<hl>`.
- `paragraph_sentence`: a `string` feature, which is same as the paragraph but a sentence containing the answer is highlighted by a special token `<hl>`.
- `sentence_answer`: a `string` feature, which is same as the sentence but the answer is highlighted by a special token `<hl>`.

Each of `paragraph_answer`, `paragraph_sentence`, and `sentence_answer` feature is assumed to be used to train a question generation model, but with different information.
The `paragraph_answer` and `sentence_answer` features are for answer-aware question generation and `paragraph_sentence` feature is for sentence-aware question generation.
See more detail at our paper.

## Datasets

- QG-Bench (multilingual): The multilingual subset of QG-Bench from Wikipedia in each language.

| Dataset                                                                       | Data size (train/valid/test) | Average character length (paragraph/sentence/question/answer) |
|-------------------------------------------------------------------------------|:----------------------------:|:-------------------------------------------------------------:|
| [English (`lmqg/qg_squad`)](https://huggingface.co/datasets/lmqg/qg_squad)    | 75,722/10,570/11,877         | 757/179/59/20                                                 |
| [French (`lmqg/qg_frquad`)](https://huggingface.co/datasets/lmqg/qg_frquad)   | 17,543/3,188/3,188           | 797/160/57/23                                                 |
| [Japanese (`lmqg/qg_jaquad`)](https://huggingface.co/datasets/lmqg/qg_jaquad) | 27,809/3,939/3,939           | 424/72/32/6                                                   |
| [Korean (`lmqg/qg_koquad`)](https://huggingface.co/datasets/lmqg/qg_koquad)   | 54,556/5,766/5,766           | 521/81/34/6                                                   |
| [Russian (`lmqg/qg_ruquad`)](https://huggingface.co/datasets/lmqg/qg_ruquad)  | 40,291/5,036/5,036           | 754/174/64/26                                                 |
| [Italian (`lmqg/qg_itquad`)](https://huggingface.co/datasets/lmqg/qg_itquad)  | 46,550/7,609/7,609           | 807/124/66/16                                                 |
| [Spanish (`lmqg/qg_esquad`)](https://huggingface.co/datasets/lmqg/qg_esquad)  | 77,025/10,570/10,570         | 781/122/64/21                                                 |
| [German (`lmqg/qg_dequad`)](https://huggingface.co/datasets/lmqg/qg_dequad)   | 9,314/2,204/2,204            | 1,577/165/59/66                                               |
  
***IMPORTANT:*** The `lmqg/qg_frquad` is private as the original FQuAD requires filling a form first, please see [`here`](https://huggingface.co/datasets/lmqg/qg_frquad_dummy/discussions/1).

- QG-Bench (multidomain): The multidomain subset of QG-Bench in English.

| Dataset                                                                                           | Data size (train/valid/test) | Average character length (paragraph/sentence/question/answer) |
|---------------------------------------------------------------------------------------------------|:----------------------------:|:-------------------------------------------------------------:|
| [SubjQA/Book (`lmqg/qg_subjqa`)](https://huggingface.co/datasets/lmqg/qg_subjqa)                  |          637/92/191          |                        1,514/146/28/83                        |
| [SubjQA/Elec (`lmqg/qg_subjqa`)](https://huggingface.co/datasets/lmqg/qg_subjqa)                  |          697/99/238          |                        1,282/129/26/66                        |
| [SubjQA/Grocery (`lmqg/qg_subjqa`)](https://huggingface.co/datasets/lmqg/qg_subjqa)               |          687/101/379         |                         896/107/25/49                         |
| [SubjQA/Movie (`lmqg/qg_subjqa`)](https://huggingface.co/datasets/lmqg/qg_subjqa)                 |          724/101/154         |                        1,746/146/27/72                        |
| [SubjQA/Restaurant (`lmqg/qg_subjqa`)](https://huggingface.co/datasets/lmqg/qg_subjqa)            |          823/129/136         |                        1,006/104/26/51                        |
| [SubjQA/Trip (`lmqg/qg_subjqa`)](https://huggingface.co/datasets/lmqg/qg_subjqa)                  |          875/143/397         |                        1,002/108/27/51                        |
| [SQuADShifts/Amazon (`lmqg/qg_squadshifts`)](https://huggingface.co/datasets/lmqg/qg_squadshifts) |       3,295/1,648/4,942      |                         773/111/43/18                         |
| [SQuADShifts/Wiki (`lmqg/qg_squadshifts`)](https://huggingface.co/datasets/lmqg/qg_squadshifts)   |       2,646/1,323/3,969      |                         773/184/58/26                         |
| [SQuADShifts/News (`lmqg/qg_squadshifts`)](https://huggingface.co/datasets/lmqg/qg_squadshifts)   |       3,355/1,678/5,032      |                         781/169/51/20                         |
| [SQuADShifts/Reddit (`lmqg/qg_squadshifts`)](https://huggingface.co/datasets/lmqg/qg_squadshifts) |       3,268/1,634/4,901      |                         774/116/45/19                         |


## QG Models
We release QG models fine-tuned on every dataset in QG-Bench. Following models are available via the transformers modelhub and can be used as below.
We recommend to use the models via [`lmqg`](https://github.com/asahi417/lm-question-generation#lmqg-language-model-for-question-generation-), but they are compatible with [`transformers`](https://github.com/huggingface/transformers) too.

- With `lmqg` library
```python
from lmqg import TransformersQG
# initialize model
model = TransformersQG(language='en', model='lmqg/t5-large-squad-qg')
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
    "Who was an English painter who specialised in watercolour landscapes?",
    "What nationality was William Turner?"
]
```

- With `transformers` library
```python
from transformers import pipeline

pipe = pipeline("text2text-generation", 'lmqg/t5-large-squad-qg')
# model prediction
input_text = 'generate question: <hl> Beyonce <hl> further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records.'
pipe(input_text)
[{'generated_text': 'Who starred as Etta James in Cadillac Records?'}]
```

### English QG Models Leaderboard
English QG model fine-tuned on [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad). The data split follows [`Du, et al 2017`](https://arxiv.org/pdf/1805.05942.pdf) and [`Du, et al 2018`](https://arxiv.org/pdf/1705.00106.pdf).

| Model                                                                         | LM                                                | Training Data                                                    | Test Data                                                        | BLEU4 | METEOR | ROUGE-L | BERTScore | MoverScore |
|-------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|-------|--------|---------|-----------|------------|
| [`UniLM`](https://arxiv.org/abs/1905.03197)                                     | UniLM (340M Parameter)                            | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 22.78 |  25.49 |   51.57 |     -     |      -     |
| [`UniLM-v2`](https://arxiv.org/abs/2002.12804)                                  | UniLM-v2 (110M Parameter)                         | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 24.70 |  26.33 |   52.13 |     -     |      -     |
| [`ProphetNet`](https://aclanthology.org/2020.findings-emnlp.217/)               | ProphetNet (340M Parameter)                       | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 23.91 |  26.60 |   52.26 |     -     |      -     |
| [`ERNIE-GEN`](https://arxiv.org/abs/2001.11314)                                 | ERNIE-GEN (340M Parameter)                        | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 25.40 |  26.92 |   52.84 |     -     |      -     |
| [`lmqg/t5-small-squad-qg`](https://huggingface.co/lmqg/t5-small-squad-qg)       | [`t5-small`](https://huggingface.co/t5-small)     | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 24.40 |  25.84 |   51.43 |     90.20 |      63.89 |
| [`lmqg/t5-base-squad-qg`](https://huggingface.co/lmqg/t5-base-squad-qg)         | [`t5-base`](https://huggingface.co/t5-base)       | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 26.13 |  26.97 |   53.33 |     90.60 |      64.74 |
| [`lmqg/t5-large-squad-qg`](https://huggingface.co/lmqg/t5-large-squad-qg)       | [`t5-large`](https://huggingface.co/t5-large)     | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 27.21 |  27.70 |   54.13 |     91.00 |      65.29 |
| [`lmqg/bart-base-squad-qg`](https://huggingface.co/lmqg/bart-base-squad-qg)     | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 24.68 |  26.05 |   52.66 |     90.87 |      64.47 |
| [`lmqg/bart-large-squad-qg`](https://huggingface.co/lmqg/bart-large-squad-qg)   | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | 26.17 |  27.07 |   53.85 |     91.00 |      64.99 |

The results of UniLM/UniLM-v2/ProphetNet/ERNIE-GEN are taken from their papers.

### Non-English QG Models Leaderboard
Non-English QG model fine-tuned on QG-Bench (multilingual).

| Model                                                                                     | LM                                                                            | Training Data                                                    | Test Data                                                        | BLEU4 | METEOR | ROUGE-L | BERTScore | MoverScore |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|-------|--------|---------|-----------|------------|
| [`lmqg/mt5-small-squad-qg`](https://huggingface.co/lmqg/mt5-small-squad-qg)                 | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)   | 21.65 |  23.83 |   48.95 |     90.01 |      62.75 |
| [`lmqg/mt5-small-ruquad-qg`](https://huggingface.co/lmqg/mt5-small-ruquad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_ruquad`](https://huggingface.co/datasets/lmqg/qg_ruquad) | [`lmqg/qg_ruquad`](https://huggingface.co/datasets/lmqg/qg_ruquad) | 16.31 |  26.39 |   31.39 |     84.27 |      62.49 |
| [`lmqg/mt5-small-jaquad-qg`](https://huggingface.co/lmqg/mt5-small-jaquad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_jaquad`](https://huggingface.co/datasets/lmqg/qg_jaquad) | [`lmqg/qg_jaquad`](https://huggingface.co/datasets/lmqg/qg_jaquad) | 30.49 |  29.03 |   50.88 |     80.87 |      58.67 |
| [`lmqg/mt5-small-itquad-qg`](https://huggingface.co/lmqg/mt5-small-itquad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_itquad`](https://huggingface.co/datasets/lmqg/qg_itquad) | [`lmqg/qg_itquad`](https://huggingface.co/datasets/lmqg/qg_itquad) |  7.37 |  17.57 |   21.93 |     80.80 |      56.79 |
| [`lmqg/mt5-small-koquad-qg`](https://huggingface.co/lmqg/mt5-small-koquad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_koquad`](https://huggingface.co/datasets/lmqg/qg_koquad) | [`lmqg/qg_koquad`](https://huggingface.co/datasets/lmqg/qg_koquad) | 10.57 |  27.52 |   25.64 |     82.89 |      82.49 |
| [`lmqg/mt5-small-esquad-qg`](https://huggingface.co/lmqg/mt5-small-esquad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_esquad`](https://huggingface.co/datasets/lmqg/qg_esquad) | [`lmqg/qg_esquad`](https://huggingface.co/datasets/lmqg/qg_esquad) |  9.61 |  22.71 |   24.62 |     84.07 |      59.06 |
| [`lmqg/mt5-small-dequad-qg`](https://huggingface.co/lmqg/mt5-small-dequad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_dequad`](https://huggingface.co/datasets/lmqg/qg_dequad) | [`lmqg/qg_dequad`](https://huggingface.co/datasets/lmqg/qg_dequad) |  0.43 |  11.47 |   10.08 |     79.90 |      54.64 |
| [`lmqg/mt5-small-frquad-qg`](https://huggingface.co/lmqg/mt5-small-frquad-qg)               | [`mt5-small`](https://huggingface.co/mt5-small)                                 | [`lmqg/qg_frquad`](https://huggingface.co/datasets/lmqg/qg_frquad) | [`lmqg/qg_frquad`](https://huggingface.co/datasets/lmqg/qg_frquad) |  8.55 |  17.51 |   28.56 |     80.71 |      56.50 |
| [`lmqg/mt5-base-squad-qg`](https://huggingface.co/lmqg/mt5-base-squad-qg)                   | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)   | 23.03 |  25.18 |   50.67 |     90.23 |      63.60 |
| [`lmqg/mt5-base-ruquad-qg`](https://huggingface.co/lmqg/mt5-base-ruquad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_ruquad`](https://huggingface.co/datasets/lmqg/qg_ruquad) | [`lmqg/qg_ruquad`](https://huggingface.co/datasets/lmqg/qg_ruquad) | 17.63 |  28.48 |   33.02 |     85.82 |      64.56 |
| [`lmqg/mt5-base-jaquad-qg`](https://huggingface.co/lmqg/mt5-base-jaquad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_jaquad`](https://huggingface.co/datasets/lmqg/qg_jaquad) | [`lmqg/qg_jaquad`](https://huggingface.co/datasets/lmqg/qg_jaquad) | 32.54 |  30.58 |   52.67 |     81.77 |      59.68 |
| [`lmqg/mt5-base-itquad-qg`](https://huggingface.co/lmqg/mt5-base-itquad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_itquad`](https://huggingface.co/datasets/lmqg/qg_itquad) | [`lmqg/qg_itquad`](https://huggingface.co/datasets/lmqg/qg_itquad) |  7.70 |  18.00 |   22.51 |     81.16 |      57.11 |
| [`lmqg/mt5-base-koquad-qg`](https://huggingface.co/lmqg/mt5-base-koquad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_koquad`](https://huggingface.co/datasets/lmqg/qg_koquad) | [`lmqg/qg_koquad`](https://huggingface.co/datasets/lmqg/qg_koquad) | 12.18 |  29.62 |   28.57 |     84.52 |      83.36 |
| [`lmqg/mt5-base-esquad-qg`](https://huggingface.co/lmqg/mt5-base-esquad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_esquad`](https://huggingface.co/datasets/lmqg/qg_esquad) | [`lmqg/qg_esquad`](https://huggingface.co/datasets/lmqg/qg_esquad) | 10.15 |  23.43 |   25.45 |     84.47 |      59.62 |
| [`lmqg/mt5-base-dequad-qg`](https://huggingface.co/lmqg/mt5-base-dequad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_dequad`](https://huggingface.co/datasets/lmqg/qg_dequad) | [`lmqg/qg_dequad`](https://huggingface.co/datasets/lmqg/qg_dequad) |  0.87 |  13.65 |   11.10 |     80.39 |      55.73 |
| [`lmqg/mt5-base-frquad-qg`](https://huggingface.co/lmqg/mt5-base-frquad-qg)                 | [`mt5-base`](https://huggingface.co/mt5-base)                                   | [`lmqg/qg_frquad`](https://huggingface.co/datasets/lmqg/qg_frquad) | [`lmqg/qg_frquad`](https://huggingface.co/datasets/lmqg/qg_frquad) |  6.14 |  15.55 |   25.88 |     77.81 |      54.58 |
| [`lmqg/mbart-large-cc25-squad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-squad-qg)   | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad)   | 23.03 |  25.10 |   50.58 |     90.36 |      63.63 |
| [`lmqg/mbart-large-cc25-ruquad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-ruquad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_ruquad`](https://huggingface.co/datasets/lmqg/qg_ruquad) | [`lmqg/qg_ruquad`](https://huggingface.co/datasets/lmqg/qg_ruquad) | 18.80 |  29.30 |   34.18 |     87.18 |      65.88 |
| [`lmqg/mbart-large-cc25-jaquad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-jaquad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_jaquad`](https://huggingface.co/datasets/lmqg/qg_jaquad) | [`lmqg/qg_jaquad`](https://huggingface.co/datasets/lmqg/qg_jaquad) | 32.16 |  29.97 |   52.95 |     82.26 |      59.88 |
| [`lmqg/mbart-large-cc25-itquad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-itquad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_itquad`](https://huggingface.co/datasets/lmqg/qg_itquad) | [`lmqg/qg_itquad`](https://huggingface.co/datasets/lmqg/qg_itquad) |  7.13 |  17.97 |   21.69 |     80.63 |      56.84 |
| [`lmqg/mbart-large-cc25-koquad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-koquad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_koquad`](https://huggingface.co/datasets/lmqg/qg_koquad) | [`lmqg/qg_koquad`](https://huggingface.co/datasets/lmqg/qg_koquad) | 10.92 |  30.23 |   27.76 |     83.89 |      82.95 |
| [`lmqg/mbart-large-cc25-esquad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-esquad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_esquad`](https://huggingface.co/datasets/lmqg/qg_esquad) | [`lmqg/qg_esquad`](https://huggingface.co/datasets/lmqg/qg_esquad) |  9.18 |  22.95 |   24.26 |     83.58 |      58.91 |
| [`lmqg/mbart-large-cc25-dequad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-dequad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_dequad`](https://huggingface.co/datasets/lmqg/qg_dequad) | [`lmqg/qg_dequad`](https://huggingface.co/datasets/lmqg/qg_dequad) |  0.75 |  13.71 |   11.19 |     80.77 |      55.88 |
| [`lmqg/mbart-large-cc25-frquad-qg`](https://huggingface.co/lmqg/mbart-large-cc25-frquad-qg) | [`facebook/mbart-large-cc25`](https://huggingface.co/facebook/mbart-large-cc25) | [`lmqg/qg_frquad`](https://huggingface.co/datasets/lmqg/qg_frquad) | [`lmqg/qg_frquad`](https://huggingface.co/datasets/lmqg/qg_frquad) |  9.47 |  19.80 |   30.62 |     81.75 |      57.96 |


## Citation
Please cite following paper if you use any resource:
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