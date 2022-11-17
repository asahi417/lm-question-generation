from datasets import load_dataset
from lmqg import TransformersQG

dataset = load_dataset("lmqg/qg_squad", split='test')
model = TransformersQG("lmqg/t5-small-squad-multitask")
sample_paragraph = dataset['paragraph'][:10]
sample_question = dataset['question'][:10]
sample_answer = dataset['answer'][:10]

ppl = model.get_perplexity(
    list_context=sample_paragraph,
    list_answer=sample_answer,
    list_question=sample_question,
    target_output='question'
)
print("question: ", ppl)


ppl = model.get_perplexity(
    list_context=sample_paragraph,
    list_answer=sample_answer,
    list_question=sample_question,
    target_output='answer'
)
print("answer:", ppl)
