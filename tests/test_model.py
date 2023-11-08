import unittest
from lmqg import TransformersQG

model_qag = 'lmqg/t5-small-tweetqa-qag'
model_qa = 'lmqg/t5-small-tweetqa-qa'
model_qg = 'lmqg/t5-small-squad-qg'
model_qg_ae = 'lmqg/t5-small-squad-qg-ae'
model_ae = 'lmqg/t5-small-squad-ae'

max_length = 256
max_length_output = 64

with open('tests/test_input.txt', 'r') as f:
    sample_text = [i for i in f.read().split('\n') if len(i) > 0]
    sample_text = sorted(sample_text, key=len, reverse=False)[:5]


class Test(unittest.TestCase):

    def test_ae(self):
        _model_ae = TransformersQG(model_ae, max_length=max_length, max_length_output=max_length_output)
        _model_qg_ae = TransformersQG(model_qg_ae, max_length=max_length, max_length_output=max_length_output)
        print("######################")
        print("* AE Model")
        for s in sample_text:
            output_ae = _model_ae.generate_a(s)
            output_qg_ae = _model_qg_ae.generate_a(s)
            print(f"\t - Input: {s}\n\t - AE: {output_ae}\n\t - QG-AE: {output_qg_ae}\n\n")

    def test_qag_pipeline_lm(self):
        model = TransformersQG(model_qg, model_ae=model_ae, max_length=max_length, max_length_output=max_length_output)
        output = model.generate_qa(list_context=sample_text)
        print("######################")
        print('* QG Model with AE model')
        for i, o in zip(sample_text, output):
            print(f"\t - Input: {i}\n\t - QA: {o}\n\n")

    def test_qag_pipeline(self):
        model = TransformersQG(model_qg, max_length=max_length, max_length_output=max_length_output)
        output = model.generate_qa(list_context=sample_text)
        print("######################")
        print('* QG Model with keyword extraction')
        for i, o in zip(sample_text, output):
            print(f"\t - Input: {i}\n\t - QA: {o}\n\n")

        model = TransformersQG(model_qg, model_ae='ner', max_length=max_length, max_length_output=max_length_output)
        output = model.generate_qa(list_context=sample_text)
        print("######################")
        print('* QG Model with NER')
        for i, o in zip(sample_text, output):
            print(f"\t - Input: {i}\n\t - QA: {o}\n\n")

    def test_qag_multitask(self):
        model = TransformersQG(model_qg_ae, max_length=max_length, max_length_output=max_length_output)

        output = model.generate_qa(list_context=sample_text[0])
        print("######################")
        print('* QAG Model (single input)')
        print(f"\t - Input: {sample_text[0]}\n\t - QA: {output}\n\n")

        output = model.generate_qa(list_context=sample_text)
        print("######################")
        print('* QAG Model')
        for i, o in zip(sample_text, output):
            print(f"\t - Input: {i}\n\t - QA: {o}\n\n")

        print("######################")
        print('* QA Model')
        model = TransformersQG(model_qa, max_length=max_length, max_length_output=max_length_output)
        answers = model.answer_q(list_context=sample_text, list_question=[o[0][0] for o in output])
        for i, o, a in zip(sample_text, output, answers):
            print(f"\t - Input: {i}\n\t - Q: {o[0][0]}\n\t - A: {a} (original: {o[0][1]})\n")

    def test_qag_e2e(self):
        model = TransformersQG(model_qag, max_length=max_length, max_length_output=max_length_output)
        output = model.generate_qa(list_context=sample_text[0])
        print("######################")
        print('* QAG Model (single input)')
        print(f"\t - Input: {sample_text[0]}\n\t - QA: {output}\n\n")

        output = model.generate_qa(list_context=sample_text)
        print("######################")
        print('* QAG Model')
        for i, o in zip(sample_text, output):
            print(f"\t - Input: {i}\n\t - QA: {o}\n\n")


if __name__ == "__main__":
    unittest.main()
