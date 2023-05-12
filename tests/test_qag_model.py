from lmqg import TransformersQG

model = TransformersQG(model='asahi417/question-generation-squad-t5-small-multitask')
# model = TransformersQG(model='asahi417/question-generation-squad-t5-base-multitask')
model.eval()
with open('misc/squad_reference_files/paragraph-test.txt') as f:
    tmp = [i for i in f.read().split('\n') if len(i) > 0]
    tmp = sorted(list(set(tmp)))
for _i in tmp:
    print(_i)
    _a = model.generate_a(_i)
    print(_a)
    out = model.generate_qa(_i)
    for o in out:
        print(o)
    input()
