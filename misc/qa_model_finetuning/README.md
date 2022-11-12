# QA Model Fine-tuning with Ray Tuner
We provide an API endpoint that allows QA model fine-tuning with [Ray Tuner](https://docs.ray.io/en/latest/tune/index.html) on [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

```shell
lmqg-qae -m "distilbert-base-uncased" -d "subjqa" -n "books" --output-dir "subjqa_books_ckpt"
lmqg-qae -m "distilbert-base-uncased" -d "subjqa" -n "electronics" --output-dir "subjqa_electronics_ckpt"
lmqg-qae -m "distilbert-base-uncased" -d "subjqa" -n "grocery" --output-dir "subjqa_grocery_ckpt"
lmqg-qae -m "distilbert-base-uncased" -d "subjqa" -n "movies" --output-dir "subjqa_movies_ckpt"
lmqg-qae -m "distilbert-base-uncased" -d "subjqa" -n "restaurants" --output-dir "subjqa_restaurants_ckpt"
lmqg-qae -m "distilbert-base-uncased" -d "subjqa" -n "tripadvisor" --output-dir "subjqa_tripadvisor_ckpt"

lmqg-qae -m "distilbert-base-uncased" -d "squad" --output-dir "squad_ckpt" --split-test "validation" --eval-step 500
```

See full argument by `lmqg-qae -h` or at the [source code](https://github.com/asahi417/lm-question-generation/blob/master/lmqg/qa_evaluation_tool/run_qa.py).
