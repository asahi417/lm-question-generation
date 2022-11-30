from lmqg.qa_evaluation_tool.run_qa import run_qa_evaluation

language_model = "distilbert-base-uncased"
# language_model = "bert-base-uncased"
# model fine-tuning
run_qa_evaluation(
    dataset='lmqg/qa_squad',
    language_model=language_model,
    n_trials=5,
    eval_step=250,
    output_dir=f'lmqg_output/bert_answer_extractor/{language_model}',
    answer_extraction_mode=True,
    skip_test=True,
    hf_model_alias_to_push=f"{language_model}-squad-answer-extraction",
    hf_organization_to_push='lmqg'
)
