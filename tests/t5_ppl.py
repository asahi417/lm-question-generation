from datasets import load_dataset
from lmqg import TransformersQG

dataset = load_dataset("lmqg/qg_squad", split='test')
model = TransformersQG("lmqg/t5-small-squad-multitask")
sample_paragraph = dataset['paragraph'][:10]
sample_question = dataset['question'][:10]
ppl = model.decoder_perplexity(
    sample_paragraph,
    sample_question,
    batch=3
)
print(ppl)
