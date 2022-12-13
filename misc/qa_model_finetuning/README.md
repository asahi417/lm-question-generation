# QA Model Fine-tuning
Abstractive QA model fine-tuning of encoder-decoder language models by [`lmqg`](https://github.com/asahi417/lm-question-generation).

- Usage
```python
from lmqg import TransformersQG
model = TransformersQG('lmqg/t5-base-tweetqa-qa')
answer = model.answer_q(
    list_context="One of the dumber and least respected of the political pundits is Chris Cillizza of the Washington Post @TheFix. Moron hates my poll numbers.",
    list_question="what does chris cillizza hate?"
)
print(answer)
>>> 'poll numbers'
```

- Leaderboard

| Model                                                                             | Data                                                                  |   AnswerExactMatch |   AnswerF1Score |   BERTScore |   METEOR |   MoverScore |   BLEU-1 |   BLEU-2 |   BLEU-3 |   BLEU-4 |   ROUGE-L | Language Model                                                      |
|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------|-------------------:|----------------:|------------:|---------:|-------------:|---------:|---------:|---------:|---------:|----------:|:--------------------------------------------------------------------|
| [`lmqg/t5-small-tweetqa-qa`](https://huggingface.co/lmqg/t5-small-tweetqa-qa)     | [`lmqg/qg_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |              38.49 |           56.12 |       92.19 |    27.89 |        74.57 |    45.54 |    37.38 |    29.91 |    23.73 |     49.86 | [`t5-small`](https://huggingface.co/t5-small)                       |
| [`lmqg/t5-base-tweetqa-qa`](https://huggingface.co/lmqg/t5-base-tweetqa-qa)       | [`lmqg/qg_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |              52.29 |           69.4  |       94.58 |    35.43 |        80.07 |    57.07 |    48.17 |    39.78 |    33.32 |     62.24 | [`t5-base`](https://huggingface.co/t5-base)                         |
| [`lmqg/t5-large-tweetqa-qa`](https://huggingface.co/lmqg/t5-large-tweetqa-qa)     | [`lmqg/qg_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |              54.45 |           71.1  |       94.8  |    36.5  |        80.79 |    58.53 |    49.65 |    41.43 |    35.02 |     64.13 | [`t5-large`](https://huggingface.co/t5-large)                       |
| [`lmqg/bart-base-tweetqa-qa`](https://huggingface.co/lmqg/bart-base-tweetqa-qa)   | [`lmqg/qg_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |              48.38 |           64.79 |       93.84 |    32.39 |        78.67 |    54.68 |    46.42 |    38.97 |    33.57 |     58.37 | [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)   |
| [`lmqg/bart-large-tweetqa-qa`](https://huggingface.co/lmqg/bart-large-tweetqa-qa) | [`lmqg/qg_tweetqa`](https://huggingface.co/datasets/lmqg/qag_tweetqa) |              50.54 |           68.58 |       94.37 |    34.86 |        79.66 |    59.01 |    49.88 |    41.7  |    35.95 |     61.82 | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) |
