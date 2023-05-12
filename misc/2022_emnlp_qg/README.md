# EMNLP 2022 QG-Bench Papaer
This repository contains the code to reproduce the models for the EMNLP 2022 QG-Bench paper,
[***Generative Language Models for Paragraph-Level Question Generation, EMNLP 2022 Main***](https://aclanthology.org/2022.emnlp-main.42/).
Please check [QG-Bench](https://github.com/asahi417/lm-question-generation/blob/master/QG_BENCH.md) for more details about the models and datasets.

## Contents
- [`nqg_baseline`](https://github.com/asahi417/lm-question-generation/blob/master/misc/2022_emnlp_qg/nqg_baseline): Get LSTM based QG baseline.
- [`qa_based_evaluation`](https://github.com/asahi417/lm-question-generation/blob/master/misc/2022_emnlp_qg/qa_based_evaluation): Run QA-based evaluation.
- [`qg_model_training`](https://github.com/asahi417/lm-question-generation/blob/master/misc/2022_emnlp_qg/qg_model_training): Train QG models.

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