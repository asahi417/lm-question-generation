# Model Finetuning/Evaluation
This is a collection of scripts we used to finetune/evaluate QG models.

## Finetuning with Hyperparameter Search & Evaluation
To run finetuning/evaluation on each QG dataset, see [model_finetuning.sh](model_finetuning.sh), which contains commands for each dataset.
We employ 3-stage hyperparameter optimization: (i) finetune the model over search space with small epoch, (ii) pick up the top-5 best models in terms of validation metric and continue finetuning,
(iii) pick up the best model within the final models and continue finetuning till the validation metric gets worse.

## Out-of-Domain Evaluation
To test QG models' domain adaptivity, we test one model in different domain (eg, evaluate SQuAD model in SubjQA's restaurant).
See [model_ood_evaluation.sh](model_ood_evaluation.sh), which is the script to run this test.

## Generate Summary
Generate summary files by
```shell
python summarize_evaluation_tweetqa.py
```

## Generate Dataset Statistics
```shell
python dataset_stats.py
```