# MT5 Finetuning on Summarization & Zeroshot Multilingual Evaluation
This is a library to finetune [mT5](https://arxiv.org/abs/2010.11934) models on summarization task.

```shell
pip install t5qg
```

```shell
t5qg-train -c mt5gen_outut/ckpt/test
```

nlg-eval --hypothesis=ckpt/dev.gen.txt --references=ckpt/dev.gold.txt --no-skipthoughts --no-glove 
