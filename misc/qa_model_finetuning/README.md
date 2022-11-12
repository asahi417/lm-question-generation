# QA Model Fine-tuning with Ray Tuner
We provide an API endpoint that allows QA model fine-tuning with [Ray Tuner](https://docs.ray.io/en/latest/tune/index.html) on [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

```shell
lmqg-qae -m "distilbert-base-uncased" -d "squad" --output-dir "squad_ckpt" --split-test "validation" --eval-step 500
lmqg-qae -m "distilbert-base-uncased" -d "adversarial_qa" -n "adversarialQA" --output-dir "adversarial_qa_ckpt" --eval-step 100
```

See full argument by `lmqg-qae -h` or at the [source code](https://github.com/asahi417/lm-question-generation/blob/master/lmqg/qa_evaluation_tool/run_qa.py).
