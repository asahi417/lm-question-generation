# QA Model Fine-tuning with Ray Tuner
We provide an API endpoint that allows QA model fine-tuning with [Ray Tuner](https://docs.ray.io/en/latest/tune/index.html) on [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

```shell
lmqg-qae -m "distilbert-base-uncased" -d "lmqg/qa_squad" --output-dir "qa_model_ckpt"
```

See full argument by `lmqg-qae -h` or at the [source code](https://github.com/asahi417/lm-question-generation/blob/master/lmqg/qa_evaluation_tool/run_qa.py).
