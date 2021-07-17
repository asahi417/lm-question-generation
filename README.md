# T5 Question Generation
We release `t5qg`, a python library to finetune [T5](https://arxiv.org/pdf/1910.10683.pdf) on question generation and provide API to host the model prediction.
For the model training, we rely on the multitasking objective where the models are optimized 
for the question answering and the answer extraction in addition to the question generation
following [huggingface tutorial](https://github.com/patil-suraj/question_generation).

We extend the library to cover recently released multilingual T5, namely [mT5](https://arxiv.org/pdf/2010.11934.pdf).

### Get Started 🚀
```shell
git clone https://github.com/asahi417/t5-question-generation
cd t5-question-generation
pip install .
```

## CLI
- ***Model Training***: Finetune T5 model for question generation.
```shell
t5qg-train -c ckpt/test -m google/mt5-small -d squad
```
run `t5qg-train -h` to display all the options.

- ***Model Evaluation***: Get metric with [nlg-eval](https://github.com/Maluuba/nlg-eval) to assess the model
```shell
t5qg-eval -m ckpt/test/epoch_10/ -e ckpt/test/epoch_10/eval
```

## Rest API
We provide a rest API which hosts the model inference.
- ***From Command Line***
```shell
uvicorn app:app --reload
```
go to [`http://127.0.0.1/docs`](http://127.0.0.1/docs).

- ***Run with Docker***
```shell
docker build -t t5qg/app:latest .
docker run -p 80:80 t5qg/app:latest
```
go to [`http://127.0.0.1/docs`](http://127.0.0.1/docs).