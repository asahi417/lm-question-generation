# Question Generation with T5

```shell
pip install t5qg
```

```shell
t5qg-train -m t5-small -c t5qg_output/ckpt/t5_small -b 32 -g 16 
```
 
```shell
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_10/ -e t5qg_output/ckpt/t5_small/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_10/ -e t5qg_output/ckpt/t5_small/epoch_10/eval
t5qg-eval -m ./t5qg_output/ckpt/t5_small/epoch_10/ -e t5qg_output/ckpt/t5_small/epoch_10/eval  
```


```shell
MODEL="./ckpt/epoch_8" uvicorn app:app --reload
```

`http://127.0.0.1:8000/docs`

t5qg-train -m t5-large -c t5qg_output/ckpt/t5_large -b 8 -g 64