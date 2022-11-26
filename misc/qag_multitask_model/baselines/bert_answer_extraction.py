from lmqg.qa_evaluation_tool.run_qa import run_qa_evaluation

# model fine-tuning
run_qa_evaluation(
    dataset='lmqg/qa_squad',
    language_model="distilbert-base-uncased",
    n_trials=5,
    eval_step=250,
    output_dir='bert_answer_extraction_model',
    answer_extraction_mode=True,
    skip_test=True
)