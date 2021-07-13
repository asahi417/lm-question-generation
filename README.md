# Question Generation with T5

```shell
pip install t5qg
```

```shell
t5qg-train -m t5-small -c t5qg_outut/ckpt/t5_small -b 32 -g 16 
```

nlg-eval --hypothesis=ckpt/dev.gen.txt --references=ckpt/dev.gold.txt --no-skipthoughts --no-glove 
