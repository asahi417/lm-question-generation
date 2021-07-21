# UNDER CONSTRUCTION!!

## TRAIN
```shell
t5qg-train -m t5-small -c t5qg_output/ckpt/t5_small -b 32 -g 16 
t5qg-train -m t5-base -c t5qg_output/ckpt/t5_base_qg -b 16 -g 32 --task-type qg
```

## Eval 
```shell
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_1/ -e t5qg_output/ckpt/t5_small/epoch_1/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_2/ -e t5qg_output/ckpt/t5_small/epoch_2/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_3/ -e t5qg_output/ckpt/t5_small/epoch_3/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_4/ -e t5qg_output/ckpt/t5_small/epoch_4/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_5/ -e t5qg_output/ckpt/t5_small/epoch_5/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_6/ -e t5qg_output/ckpt/t5_small/epoch_6/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_7/ -e t5qg_output/ckpt/t5_small/epoch_7/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_8/ -e t5qg_output/ckpt/t5_small/epoch_8/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_9/ -e t5qg_output/ckpt/t5_small/epoch_9/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_10/ -e t5qg_output/ckpt/t5_small/epoch_10/eval
```

```shell
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_1/ -e t5qg_output/ckpt/t5_base/epoch_1/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_2/ -e t5qg_output/ckpt/t5_base/epoch_2/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_3/ -e t5qg_output/ckpt/t5_base/epoch_3/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_4/ -e t5qg_output/ckpt/t5_base/epoch_4/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_5/ -e t5qg_output/ckpt/t5_base/epoch_5/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_6/ -e t5qg_output/ckpt/t5_base/epoch_6/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_7/ -e t5qg_output/ckpt/t5_base/epoch_7/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_8/ -e t5qg_output/ckpt/t5_base/epoch_8/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_9/ -e t5qg_output/ckpt/t5_base/epoch_9/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_base/epoch_10/ -e t5qg_output/ckpt/t5_base/epoch_10/eval
```

```shell
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_1/ -e t5qg_output/ckpt/mt5_small/epoch_1/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_2/ -e t5qg_output/ckpt/mt5_small/epoch_2/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_3/ -e t5qg_output/ckpt/mt5_small/epoch_3/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_4/ -e t5qg_output/ckpt/mt5_small/epoch_4/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_5/ -e t5qg_output/ckpt/mt5_small/epoch_5/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_6/ -e t5qg_output/ckpt/mt5_small/epoch_6/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_7/ -e t5qg_output/ckpt/mt5_small/epoch_7/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_8/ -e t5qg_output/ckpt/mt5_small/epoch_8/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_9/ -e t5qg_output/ckpt/mt5_small/epoch_9/eval
t5qg-eval -m ./t5qg_output/ckpt/mt5_small/epoch_10/ -e t5qg_output/ckpt/mt5_small/epoch_10/eval
```

```shell
MODEL="./ckpt/epoch_8" uvicorn app:app --reload
```

`http://127.0.0.1:8000/docs`

t5qg-train -m t5-large -c t5qg_output/ckpt/t5_large -b 8 -g 64


## NOTE
1. Train T5/mT5 on English SQuAD: {T5, mT5}-{small, base} model trainings are running on Hawk, while others on CANS.

# Run with docker
docker build -t t5qg/app:1.0 .