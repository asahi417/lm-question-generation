# Experiments Note
TODO: fix TydiQA

## Model Training
- T5
```shell
t5qg-train -m t5-small -c t5qg_output/ckpt/t5_small -b 32 -g 16 
t5qg-train -m t5-base -c t5qg_output/ckpt/t5_base -b 16 -g 32
t5qg-train -m t5-large -c t5qg_output/ckpt/t5_large -b 16 -g 32
t5qg-train -m google/mt5-small -c t5qg_output/ckpt/mt5_small -b 16 -g 32 
t5qg-train -m google/mt5-base -c t5qg_output/ckpt/mt5_base -b 8 -g 64
```

- BART
```shell
t5qg-train -m facebook/bart-base -c t5qg_output/ckpt/bart_base --task-type qg -b 32 -g 16
t5qg-train -m facebook/bart-large -c t5qg_output/ckpt/bart_large --task-type qg -b 8 -g 64
```

## Model Evaluation 
```shell
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_10/ -e t5qg_output/ckpt/t5_small/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_10/ -e t5qg_output/ckpt/t5_base/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_10/ -e t5qg_output/ckpt/mt5_small/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_base/epoch_10/ -e t5qg_output/ckpt/mt5_base/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/bart_base/epoch_10/ -e t5qg_output/ckpt/bart_base/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/bart_large/epoch_10/ -e t5qg_output/ckpt/bart_large/epoch_10/eval
```


```shell
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_1/ -e t5qg_output/ckpt/t5_large/epoch_1/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_2/ -e t5qg_output/ckpt/t5_large/epoch_2/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_3/ -e t5qg_output/ckpt/t5_large/epoch_3/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_4/ -e t5qg_output/ckpt/t5_large/epoch_4/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_5/ -e t5qg_output/ckpt/t5_large/epoch_5/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_6/ -e t5qg_output/ckpt/t5_large/epoch_6/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_large/epoch_7/ -e t5qg_output/ckpt/t5_large/epoch_7/eval
```

## NOTE
1. Train T5/mT5 on English SQuAD: {T5, mT5}-{small, base} model trainings are running on Hawk, while others on CANS.
