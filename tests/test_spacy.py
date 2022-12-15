from lmqg.spacy_module import SpacyPipeline

with open('tests/test_input.txt') as f:
    samples = f.read().split('\n')

pipe = SpacyPipeline('en', 'ner')
for i in samples:
    out = pipe.keyword(i)
    print(str(out[0]))
    print(type(out[0]))
