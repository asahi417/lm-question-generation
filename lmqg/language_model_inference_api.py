""" Language model for Huggingface inference API. """
import requests
import json
import logging
from typing import List

import transformers
from .exceptions import APIError, HighlightNotFoundError, AnswerNotFoundError
from .spacy_module import SpacyPipeline
from .language_model import clean, TASK_PREFIX, ADDITIONAL_SP_TOKENS


def call_api(input_text,
             api_token,
             model,
             max_length: int = 64,
             do_sample: bool = True,
             num_beams: int = 1,
             top_p: float = 0.9,
             use_gpu: bool = False):
    data = json.dumps({
        "inputs": input_text,
        "parameters": {"do_sample": do_sample, "num_beams": num_beams, "max_length": max_length, "top_p": top_p},
        "options": {"wait_for_model": True, "use_gpu": use_gpu}
    })
    response = requests.request("POST", f"https://api-inference.huggingface.co/models/{model}",
                                headers={"Authorization": f"Bearer {api_token}"}, data=data)
    data = response.json()
    if type(data) is dict and 'error' in data.keys():
        raise APIError(data)
    return data


def highlight_sentence(input_text: str, highlight_span: str, prefix: str = None):
    position = input_text.find(highlight_span)
    if position == -1:
        HighlightNotFoundError(highlight_span, input_text)
    input_text = f"{input_text[:position]}{ADDITIONAL_SP_TOKENS['hl']} " \
                 f"{highlight_span} {ADDITIONAL_SP_TOKENS['hl']}{input_text[position + len(highlight_span):]}"
    if prefix is not None:
        input_text = f"{prefix}: {input_text}"
    return input_text


class TransformersQGInferenceAPI:

    def __init__(self,
                 qg_model: str,
                 language: str,
                 api_token: str,
                 answer_model: str = None,
                 cache_dir: str = None,
                 add_prefix_qg: bool = None,
                 add_prefix_answer: bool = None,
                 keyword_extraction_model: str = 'positionrank'):
        logging.info('initialize TransformersQGInferenceAPI:')
        logging.info(f'\n\tqg:{qg_model}\n\tqa:{answer_model}\n\tla:{language}')
        self.language = language
        self.qg_model = qg_model
        if add_prefix_qg is None:
            config_qg_model = transformers.AutoConfig.from_pretrained(self.qg_model, cache_dir=cache_dir)
            self.add_prefix_qg = config_qg_model.add_prefix
        else:
            self.add_prefix_qg = add_prefix_qg
        self.api_token = api_token
        self.spacy_module = SpacyPipeline(language, keyword_extraction_model)
        if answer_model is not None and answer_model != 'keyword_extraction':
            self.a_model = answer_model
            if add_prefix_answer is None:
                config_a_model = transformers.AutoConfig.from_pretrained(self.a_model, cache_dir=cache_dir)
                self.add_prefix_answer = config_a_model.add_prefix
            else:
                self.add_prefix_answer = add_prefix_answer
        else:
            self.a_model = self.add_prefix_answer = None

    def generate_qa(self,
                    input_text,
                    input_answer: List or str = None,
                    max_length: int = 64,
                    do_sample: bool = True,
                    num_beams: int = 1,
                    top_p: float = 0.9,
                    use_gpu: bool = False,
                    num_questions: int = 10):
        if input_answer is None or len(input_answer) == 0:
            logging.info(f"running answer extraction: {'keyword extraction' if self.a_model is None else self.a_model}")
            if self.a_model is None:  # keyword extraction
                input_answer = self.spacy_module.keyword(input_text, num_questions)
            else:
                input_answer = self.generate_a(
                    input_text, max_length=max_length, do_sample=do_sample, num_beams=num_beams, use_gpu=use_gpu,
                    top_p=top_p)
                input_answer = input_answer[:min(num_questions, len(input_answer))]
        logging.info("generate qg")
        input_answer = [input_answer] if type(input_answer) is str else input_answer

        if self.add_prefix_qg:
            prefix = TASK_PREFIX['qg']
        else:
            prefix = None
        batch = [highlight_sentence(input_text, i, prefix) for i in input_answer]
        output = call_api(input_text=batch, api_token=self.api_token, model=self.qg_model, max_length=max_length,
                          do_sample=do_sample, num_beams=num_beams, use_gpu=use_gpu, top_p=top_p)
        question = [i['generated_text'] for i in output]
        assert len(question) == len(input_answer), f"{question}, {input_answer}"
        output = [{'question': q, 'answer': a} for q, a in zip(question, input_answer)]
        logging.info(f"complete process: {len(output)} qa pairs generated")
        return output

    def generate_a(self,
                   input_text: str,
                   max_length: int = 64,
                   do_sample: bool = True,
                   num_beams: int = 1,
                   top_p: float = 0.9,
                   use_gpu: bool = False):
        # predict answer span in each sentence
        if self.add_prefix_answer:
            prefix = TASK_PREFIX['ae']
        else:
            prefix = None
        batch = [highlight_sentence(input_text, i, prefix) for i in self.spacy_module.sentence(input_text)]
        # API call
        result = call_api(
            input_text=batch, api_token=self.api_token, model=self.a_model, max_length=max_length,
            do_sample=do_sample, num_beams=num_beams, use_gpu=use_gpu, top_p=top_p)
        answer = list(filter(None, [clean(i['generated_text']) for i in result]))  # remove None
        answer = list(filter(lambda x: x in input_text, answer))  # remove answers not in input_text
        if len(answer) == 0:
            raise AnswerNotFoundError(input_text)
        return answer

