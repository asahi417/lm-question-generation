# ACL 2023 QAG & Demo Papers
This repository contains the code to reproduce the models for the following papers:
- [***TBA, ACL 2022 Finding***](tba)
- [***A Practical Toolkit for Multilingual Question and Answer Generation, ACL 2022, System Demonstration***](tba)

## Contents
- [`nqg_baseline`](https://github.com/asahi417/lm-question-generation/blob/master/misc/2022_emnlp_qg/nqg_baseline): Get LSTM based QG baseline.
- [`qa_based_evaluation`](https://github.com/asahi417/lm-question-generation/blob/master/misc/2022_emnlp_qg/qa_based_evaluation): Run QA-based evaluation.
- [`qg_model_training`](https://github.com/asahi417/lm-question-generation/blob/master/misc/2022_emnlp_qg/qg_model_training): Train QG models.

## Citation
Please cite following paper if you use any resource:
- [***TBA, ACL 2022 Finding***](tba)
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

- [***A Practical Toolkit for Multilingual Question and Answer Generation, ACL 2022, System Demonstration***](tba)

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



# QAG
## Get Started
```shell
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download ko_core_news_sm
python -m spacy download ru_core_news_sm
python -m spacy download fr_core_news_sm
```


## QAG
End-to-end QAG model fine-tuning of encoder-decoder language models by [`lmqg`](https://github.com/asahi417/lm-question-generation).

- Usage
```python
from lmqg import TransformersQG
model = TransformersQG('lmqg/t5-base-tweetqa-qag')
qa_pairs = model.generate_qa(
    list_context="One of the dumber and least respected of the political pundits is Chris Cillizza of the Washington Post @TheFix. Moron hates my poll numbers.",
)
print(qa_pairs)
>>> [
    ('who hates my poll numbers?', 'moron'),
    ('who is one of the dumber and least respected of the political pundits?', 'chris cillizza')
]
```


- Leaderboard

| Model                                                                               | Data                                                                   |   BERTScore |   METEOR |   MoverScore |   QAAlignedF1Score (BERTScore) |   QAAlignedF1Score (MoverScore) |   QAAlignedPrecision (BERTScore) |   QAAlignedPrecision (MoverScore) |   QAAlignedRecall (BERTScore) |   QAAlignedRecall (MoverScore) |   BLEU-1 |   BLEU-2 |   BLEU-3 |   BLEU-4 |   ROUGE-L | Language Model                                                      |
|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|------------:|---------:|-------------:|-------------------------------:|--------------------------------:|---------------------------------:|----------------------------------:|------------------------------:|-------------------------------:|---------:|---------:|---------:|---------:|----------:|:--------------------------------------------------------------------|
| [`lmqg/t5-small-tweetqa-qag`](https://huggingface.co/lmqg/t5-small-tweetqa-qag)     | [`lmqg/qag_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |       89.64 |    28.02 |        60.47 |                          91.42 |                           63.08 |                            91.89 |                             64.08 |                         90.98 |                          62.16 |    35.53 |    22.94 |    15.11 |    10.08 |     34.19 | [`t5-small`](https://huggingface.co/t5-small)                       |
| [`lmqg/t5-base-tweetqa-qag`](https://huggingface.co/lmqg/t5-base-tweetqa-qag)       | [`lmqg/qag_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |       90.55 |    30.35 |        61.82 |                          92.37 |                           64.63 |                            92.75 |                             65.5  |                         92.01 |                          63.85 |    39.29 |    26.69 |    18.4  |    12.93 |     36.54 | [`t5-base`](https://huggingface.co/t5-base)                         |
| [`lmqg/t5-large-tweetqa-qag`](https://huggingface.co/lmqg/t5-large-tweetqa-qag)     | [`lmqg/qag_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |       91.09 |    31.61 |        62.77 |                          92.5  |                           65.05 |                            92.72 |                             65.58 |                         92.29 |                          64.59 |    41.33 |    28.37 |    19.68 |    13.76 |     37.24 | [`t5-large`](https://huggingface.co/t5-large)                       |
| [`lmqg/bart-base-tweetqa-qag`](https://huggingface.co/lmqg/bart-base-tweetqa-qag)   | [`lmqg/qag_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |       91.19 |    25.66 |        61.59 |                          91.5  |                           63.78 |                            91.9  |                             64.77 |                         91.11 |                          62.89 |    39.8  |    27.7  |    19.05 |    13.27 |     33.39 | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |
| [`lmqg/bart-large-tweetqa-qag`](https://huggingface.co/lmqg/bart-large-tweetqa-qag) | [`lmqg/qag_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |       91.27 |    27.91 |        62.25 |                          92.47 |                           64.66 |                            92.74 |                             65.39 |                         92.21 |                          64.03 |    44.55 |    31.15 |    21.58 |    15.18 |     34.99 | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |


## Benchmark 

- Answer Extraction

| Model                                                                           | Data                                                             | Type          | Language Model                                                      |   AnswerExactMatch |   AnswerF1Score |   BERTScore |   METEOR |   MoverScore |   BLEU-1 |   BLEU-2 |   BLEU-3 |   BLEU-4 |   ROUGE-L |
|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------|:--------------|:--------------------------------------------------------------------|-------------------:|----------------:|------------:|---------:|-------------:|---------:|---------:|---------:|---------:|----------:|
| [`lmqg/bart-base-squad-ae`](https://huggingface.co/lmqg/bart-base-squad-ae)     | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | AE            | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |              58.17 |           69.47 |       91.96 |    41.71 |        82.2  |    65.92 |    63.24 |    60.8  |    58.72 |     68.7  |
| [`lmqg/bart-large-squad-ae`](https://huggingface.co/lmqg/bart-large-squad-ae)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | AE            | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |              58.95 |           69.67 |       91.93 |    41.89 |        82.41 |    65.82 |    63.21 |    60.73 |    58.61 |     68.96 |
| [`lmqg/t5-base-squad-ae`](https://huggingface.co/lmqg/t5-base-squad-ae)         | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | AE            | [`t5-base`](https://huggingface.co/t5-base)                         |              59.48 |           70.32 |       91.87 |    43.62 |        82.69 |    64.27 |    60.78 |    57.35 |    54.28 |     69.72 |
| [`lmqg/t5-large-squad-ae`](https://huggingface.co/lmqg/t5-large-squad-ae)       | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | AE            | [`t5-large`](https://huggingface.co/t5-large)                       |              59.77 |           70.41 |       91.91 |    43.06 |        82.82 |    65.48 |    62.11 |    58.71 |    55.66 |     69.67 |
| [`lmqg/t5-small-squad-ae`](https://huggingface.co/lmqg/t5-small-squad-ae)       | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | AE            | [`t5-small`](https://huggingface.co/t5-small)                       |              56.15 |           68.06 |       91.2  |    42.5  |        80.92 |    52.42 |    47.81 |    43.22 |    39.23 |     67.58 |
| [`lmqg/t5-base-squad-qg-ae`](https://huggingface.co/lmqg/t5-base-squad-qg-ae)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-base`](https://huggingface.co/t5-base)                         |              58.9  |           70.18 |       91.57 |    43.94 |        82.16 |    56.96 |    52.57 |    48.21 |    44.33 |     69.62 |
| [`lmqg/t5-large-squad-qg-ae`](https://huggingface.co/lmqg/t5-large-squad-qg-ae) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-large`](https://huggingface.co/t5-large)                       |              59.26 |           70.3  |       91.63 |    44.46 |        82.48 |    60.87 |    56.96 |    53.12 |    49.73 |     69.82 |
| [`lmqg/t5-small-squad-qg-ae`](https://huggingface.co/lmqg/t5-small-squad-qg-ae) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-small`](https://huggingface.co/t5-small)                       |              54.17 |           66.92 |       90.77 |    40.9  |        79.49 |    40.81 |    35.84 |    31.06 |    27.06 |     66.52 | 


- Question Generation

| Model                                                                           | Data                                                             | Type          | Language Model                                                      |   BERTScore |   METEOR |   MoverScore |   BLEU-1 |   BLEU-2 |   BLEU-3 |   BLEU-4 |   ROUGE-L |
|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------|:--------------|:--------------------------------------------------------------------|------------:|---------:|-------------:|---------:|---------:|---------:|---------:|----------:|
| [`lmqg/t5-base-squad-qg-ae`](https://huggingface.co/lmqg/t5-base-squad-qg-ae)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-base`](https://huggingface.co/t5-base)                         |       90.58 |    27    |        64.72 |    58.59 |    42.6  |    32.91 |    26.01 |     53.4  |
| [`lmqg/t5-large-squad-qg-ae`](https://huggingface.co/lmqg/t5-large-squad-qg-ae) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-large`](https://huggingface.co/t5-large)                       |       90.69 |    27.81 |        65.29 |    59.93 |    43.98 |    34.19 |    27.2  |     54.23 |
| [`lmqg/t5-small-squad-qg-ae`](https://huggingface.co/lmqg/t5-small-squad-qg-ae) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-small`](https://huggingface.co/t5-small)                       |       90.18 |    25.58 |        63.72 |    56.54 |    40.31 |    30.8  |    24.18 |     51.12 |
| [`lmqg/bart-base-squad-qg`](https://huggingface.co/lmqg/bart-base-squad-qg)     | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |       90.87 |    26.05 |        64.47 |    56.92 |    40.98 |    31.44 |    24.68 |     52.66 |
| [`lmqg/bart-large-squad-qg`](https://huggingface.co/lmqg/bart-large-squad-qg)   | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |       91    |    27.07 |        64.99 |    58.79 |    42.79 |    33.11 |    26.17 |     53.85 |
| [`lmqg/t5-base-squad-qg`](https://huggingface.co/lmqg/t5-base-squad-qg)         | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`t5-base`](https://huggingface.co/t5-base)                         |       90.6  |    26.97 |        64.74 |    58.69 |    42.66 |    32.99 |    26.13 |     53.33 |
| [`lmqg/t5-large-squad-qg`](https://huggingface.co/lmqg/t5-large-squad-qg)       | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`t5-large`](https://huggingface.co/t5-large)                       |       91    |    27.7  |        65.29 |    59.54 |    43.79 |    34.14 |    27.21 |     54.13 |
| [`lmqg/t5-small-squad-qg`](https://huggingface.co/lmqg/t5-small-squad-qg)       | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`t5-small`](https://huggingface.co/t5-small)                       |       90.2  |    25.84 |        63.89 |    56.86 |    40.59 |    31.05 |    24.4  |     51.43 | 


- Question & Answer Pairs Generation

| Model                                                                                                                                                        | Data                                                             | Type          | Language Model                                                      |   QAAlignedF1Score (BERTScore) |   QAAlignedF1Score (MoverScore) |   QAAlignedPrecision (BERTScore) |   QAAlignedPrecision (MoverScore) |   QAAlignedRecall (BERTScore) |   QAAlignedRecall (MoverScore) |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|:--------------|:--------------------------------------------------------------------|-------------------------------:|--------------------------------:|---------------------------------:|----------------------------------:|------------------------------:|-------------------------------:|
| [`lmqg/bart-base-squad-qag`](https://huggingface.co/lmqg/bart-base-squad-qag)                                                                                | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | End2end QAG   | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |                          84.49 |                           57.46 |                            85.64 |                             60.01 |                         83.38 |                          55.26 |
| [`lmqg/bart-large-squad-qag`](https://huggingface.co/lmqg/bart-large-squad-qag)                                                                              | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | End2end QAG   | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |                          92.16 |                           63.79 |                            93.21 |                             66.71 |                         91.17 |                          61.32 |
| [`lmqg/t5-base-squad-qag`](https://huggingface.co/lmqg/t5-base-squad-qag)                                                                                    | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | End2end QAG   | [`t5-base`](https://huggingface.co/t5-base)                         |                          93.34 |                           65.78 |                            93.18 |                             65.96 |                         93.51 |                          65.68 |
| [`lmqg/t5-large-squad-qag`](https://huggingface.co/lmqg/t5-large-squad-qag)                                                                                  | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | End2end QAG   | [`t5-large`](https://huggingface.co/t5-large)                       |                          93.45 |                           66.05 |                            93.34 |                             66.34 |                         93.57 |                          65.84 |
| [`lmqg/t5-small-squad-qag`](https://huggingface.co/lmqg/t5-small-squad-qag)                                                                                  | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | End2end QAG   | [`t5-small`](https://huggingface.co/t5-small)                       |                          92.76 |                           64.59 |                            92.87 |                             65.3  |                         92.68 |                          63.99 |
| [`lmqg/t5-base-squad-qg-ae`](https://huggingface.co/lmqg/t5-base-squad-qg-ae)                                                                                | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-base`](https://huggingface.co/t5-base)                         |                          92.53 |                           64.23 |                            92.35 |                             64.33 |                         92.74 |                          64.23 |
| [`lmqg/t5-large-squad-qg-ae`](https://huggingface.co/lmqg/t5-large-squad-qg-ae)                                                                              | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-large`](https://huggingface.co/t5-large)                       |                          92.87 |                           64.67 |                            92.72 |                             64.82 |                         93.04 |                          64.63 |
| [`lmqg/t5-small-squad-qg-ae`](https://huggingface.co/lmqg/t5-small-squad-qg-ae)                                                                              | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Multitask QAG | [`t5-small`](https://huggingface.co/t5-small)                       |                          91.74 |                           63.23 |                            91.49 |                             63.26 |                         92.01 |                          63.29 |
| [`lmqg/bart-base-squad-qg`](https://huggingface.co/lmqg/bart-base-squad-qg), [`lmqg/bart-base-squad-ae`](https://huggingface.co/lmqg/bart-base-squad-ae)     | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Pipeline QAG  | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |                          92.84 |                           64.24 |                            92.75 |                             64.46 |                         92.95 |                          64.11 |
| [`lmqg/bart-large-squad-qg`](https://huggingface.co/lmqg/bart-large-squad-qg), [`lmqg/bart-large-squad-ae`](https://huggingface.co/lmqg/bart-large-squad-ae) | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Pipeline QAG  | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |                          93.23 |                           64.76 |                            93.13 |                             64.98 |                         93.35 |                          64.63 |
| [`lmqg/t5-base-squad-qg`](https://huggingface.co/lmqg/t5-base-squad-qg), [`lmqg/t5-base-squad-ae`](https://huggingface.co/lmqg/t5-base-squad-ae)             | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Pipeline QAG  | [`t5-base`](https://huggingface.co/t5-base)                         |                          92.75 |                           64.36 |                            92.59 |                             64.45 |                         92.93 |                          64.35 |
| [`lmqg/t5-small-squad-qg`](https://huggingface.co/lmqg/t5-small-squad-qg), [`lmqg/t5-small-squad-ae`](https://huggingface.co/lmqg/t5-small-squad-ae)         | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | Pipeline QAG  | [`t5-small`](https://huggingface.co/t5-small)                       |                          92.26 |                           63.83 |                            92.07 |                             63.92 |                         92.48 |                          63.82 |
| [`lmqg/bart-base-squad-qg`](https://huggingface.co/lmqg/bart-base-squad-qg)                                                                                  | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |                          95.49 |                           70.38 |                            95.55 |                             70.67 |                         95.44 |                          70.1  |
| [`lmqg/bart-large-squad-qg`](https://huggingface.co/lmqg/bart-large-squad-qg)                                                                                | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |                          95.54 |                           70.82 |                            95.59 |                             71.13 |                         95.49 |                          70.54 |
| [`lmqg/t5-base-squad-qg`](https://huggingface.co/lmqg/t5-base-squad-qg)                                                                                      | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`t5-base`](https://huggingface.co/t5-base)                         |                          95.42 |                           70.63 |                            95.48 |                             70.92 |                         95.37 |                          70.34 |
| [`lmqg/t5-large-squad-qg`](https://huggingface.co/lmqg/t5-large-squad-qg)                                                                                    | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`t5-large`](https://huggingface.co/t5-large)                       |                          95.57 |                           71.1  |                            95.62 |                             71.41 |                         95.51 |                          70.8  |
| [`lmqg/t5-small-squad-qg`](https://huggingface.co/lmqg/t5-small-squad-qg)                                                                                    | [`lmqg/qg_squad`](https://huggingface.co/datasets/lmqg/qg_squad) | QG            | [`t5-small`](https://huggingface.co/t5-small)                       |                          95.14 |                           69.79 |                            95.19 |                             70.09 |                         95.09 |                          69.51 | 


