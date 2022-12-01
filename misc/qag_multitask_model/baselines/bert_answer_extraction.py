import os
from lmqg.qa_evaluation_tool.run_qa import run_qa_evaluation

for language_model in ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]:
    run_qa_evaluation(
        dataset='lmqg/qa_squad',
        language_model=language_model,
        down_sample_size_validation=5000,
        down_sample_size_train=50000,
        eval_step=10000,
        n_trials=5,
        output_dir=f'lmqg_output/bert_answer_extractor/{os.path.basename(language_model)}',
        answer_extraction_mode=True,
        hf_model_alias_to_push=f"{os.path.basename(language_model)}-squad-answer-extraction",
        hf_organization_to_push='lmqg'
    )
