from lmqg.qa_evaluation_tool.run_qa import run_qa_evaluation

language_model = "distilbert-base-uncased"
# language_model = "bert-base-uncased"
# language_model = "roberta-base"
# model fine-tuning
run_qa_evaluation(
    dataset='lmqg/qa_squad',
    language_model=language_model,
    down_sample_size_validation=5000,
    down_sample_size_train=50000,
    eval_step=10000,
    n_trials=5,
    output_dir=f'lmqg_output/bert_answer_extractor/{language_model}',
    answer_extraction_mode=True,
    hf_model_alias_to_push=f"{language_model}-squad-answer-extraction",
    hf_organization_to_push='lmqg'
)

# for language_model in ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]:
#     config = AutoConfig.from_pretrained(language_model)
#     tokenizer = AutoTokenizer.from_pretrained(language_model, use_fast=True)
#     model = AutoModelForQuestionAnswering.from_pretrained(language_model, config=config)
