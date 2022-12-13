# QAG
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
